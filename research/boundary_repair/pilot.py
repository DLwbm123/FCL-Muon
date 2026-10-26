"""Isolated repair learner; the original UtilityPilot and optimizers stay unchanged."""
import math
import time
import torch
from torch.nn import functional as F
from fcl.math import ClientMuon
from fcl.checkpoint import rng_state,restore_rng
from research.aggregation_utility.pilot import UtilityPilot
from .math import metric,inner,norm2,repair,cross_loss


class BoundaryPilot(UtilityPilot):
    def enabled(self):
        return self.cfg['boundary_repair']['enabled'] and self.s['task']>0

    def initialize_task(self):
        super().initialize_task()
        if not self.enabled():return
        settings=self.cfg['boundary_repair'];saved={};scales={}
        if self.s['history'].committed!=set(range(self.s['task'])):raise ValueError('History includes wrong tasks')
        for name,a in self.model.adapters.items():
            q,inv,scale=metric(a.A,self.s['history'].raw[name].to(self.device),settings['epsilon'],settings['lambda'])
            saved[name]={'Q':q.cpu(),'inverse':inv.cpu()};scales[name]=scale
        self.s['repair_metric']={'task':self.s['task'],'layers':saved,'scales':scales}
        self._metric_cache=None
        self.charge('repair_Q_broadcast',self.cfg['clients']*sum(v['Q'].numel()*8 for v in saved.values()))
        self.event('repair_metric',history_tasks=sorted(self.s['history'].committed),scales=scales)

    def metrics(self):
        cache=getattr(self,'_metric_cache',None)
        if cache is None or cache[0]!=self.s['task']:
            saved=self.s['repair_metric'];assert saved['task']==self.s['task']
            qs=[];inverses=[]
            for name in self.model.adapters:
                v=saved['layers'][name];q=v['Q'].to(self.device);inv=v['inverse'].to(self.device)
                qs.extend([q,q]);inverses.extend([inv,inv])
            self._metric_cache=(self.s['task'],qs,inverses)
        return self._metric_cache[1:]

    @torch.no_grad()
    def diagnostic_losses(self,x,y,mask,base,corrected):
        state=rng_state();buffers={n:v.detach().clone() for n,v in self.model.named_buffers()}
        modes={m:m.training for m in self.model.modules()};bs=self.model.b_parameters();out={}
        try:
            for name,values in [('base',base),('corrected',corrected)]:
                restore_rng(state)
                for n,v in self.model.named_buffers():v.copy_(buffers[n])
                for b,v in zip(bs,values):b.copy_(v)
                z=self.model(x);t=self.s['task'];c=self.cfg['cpt']
                out[name]={'ce':float(F.cross_entropy(z[:,t*c:],y-t*c)),
                           'cross':float(cross_loss(z,y,t,c,mask)) if mask.any() else None}
        finally:
            for b,v in zip(bs,corrected):b.copy_(v)
            for n,v in self.model.named_buffers():v.copy_(buffers[n])
            for m,mode in modes.items():m.training=mode
            restore_rng(state)
        return out

    def repair_step(self,x,y,opt,headopt,lr,window):
        start=time.perf_counter();bs=self.model.b_parameters();w=self.model.rows[-1];params=bs+[w]
        t=self.s['task'];c=self.cfg['cpt'];z=self.model(x)
        loss=F.cross_entropy(z[:,t*c:],y-t*c)
        mask=(z[:,t*c:].detach().argmax(1)==y-t*c);active=bool(mask.any())
        cross=cross_loss(z,y,t,c,mask) if active else None
        gs=torch.autograd.grad(loss,params,retain_graph=active)
        hs=torch.autograd.grad(cross,params) if active else None
        if not torch.isfinite(loss) or any(not torch.isfinite(g).all() for g in gs):raise ArithmeticError('Nonfinite CE gradient')
        if hs is not None and (not torch.isfinite(cross) or any(not torch.isfinite(h).all() for h in hs)):raise ArithmeticError('Nonfinite competition gradient')
        before=[p.detach().clone() for p in params]
        for p,g in zip(params,gs):p.grad=g.detach()
        opt.step(lr);headopt.step()
        base=[(p.detach()-v) for p,v in zip(bs,before[:-1])];head_step=w.detach()-before[-1]
        base_values=[p.detach().clone() for p in bs]
        d=self._diagnostic;d['steps']+=1;d['images']+=len(y);d['mask_images']+=int(mask.sum())
        reasons=d['reasons'];info={'reason':'empty_mask','trigger':False}
        if active:
            d['extra_backward_calls']+=1;d['extra_backward_images']+=len(y)
            qs,inverses=self.metrics();g,h=gs[:-1],hs[:-1]
            delta,info=repair(g,h,base,inner([hs[-1]],[head_step]),qs,inverses,
                              self.cfg['boundary_repair']['kappa'],self.cfg['boundary_repair']['relative_denominator_tolerance'])
            with torch.no_grad():
                for p,v in zip(bs,delta):p.add_(v.to(p.dtype))
            actual=[p.detach()-b for p,b in zip(bs,base_values)]
            round_error=[a.double()-v for a,v in zip(actual,delta)]
            qnorm=float(norm2(actual,qs).clamp_min(0).sqrt());roundnorm=float(norm2(round_error,qs).clamp_min(0).sqrt())
            gres=float(inner(g,actual));residual=float(inner(h,actual))+info['joint_cross']
            gn=float(inner(g,g).sqrt());dn=float(inner(actual,actual).sqrt())
            normalized=abs(gres)/max(gn*dn,1e-300)
            # FP32 deployment tolerance is the explicit arithmetic error envelope,
            # recorded separately from the ideal FP64 orthogonality/cap properties.
            rounding_ce=float(sum((a.double().abs()*e.abs()).sum() for a,e in zip(g,round_error)))
            tol=self.cfg['boundary_repair']['geometry_tolerance']
            if abs(gres)>rounding_ce+tol*max(gn*dn,1e-20):raise ArithmeticError('CE orthogonality outside rounding envelope')
            if qnorm>self.cfg['boundary_repair']['kappa']*info['base_norm_Q']+roundnorm+tol*max(info['base_norm_Q'],1e-20):raise ArithmeticError('Global cap outside rounding envelope')
            if info['reason']=='full':
                rounding_h=float(sum((a.double().abs()*e.abs()).sum() for a,e in zip(h,round_error)))
                if residual>rounding_h+tol*max(abs(info['joint_cross']),1e-20):raise ArithmeticError('Full repair residual outside rounding envelope')
            # Momentum descent is -M; SSB's direction is tested using actual base displacement.
            d['momentum_help_ssb_hurt']+=int(inner(h,opt.state)>0 and info['base_cross']>0 and lr>0)
            vals={'base_cross':info['base_cross'],'head_cross':info['head_cross'],
                  'joint_cross_before':info['joint_cross'],'joint_cross_after':residual,
                  'remaining_positive_conflict':max(0.,residual),'history_cost':.5*qnorm*qnorm,
                  'amplitude_ratio':qnorm/max(info['base_norm_Q'],1e-300),
                  'normalized_ce_residual':normalized,'rounding_Q_norm':roundnorm,
                  'ideal_ce_residual':abs(info.get('ideal_ce_residual',0.))}
            for name,value in vals.items():
                d['sums'][name]=d['sums'].get(name,0.)+value
                d['maxima'][name]=max(d['maxima'].get(name,0.),abs(value))
            d['triggers']+=int(info['trigger']);d['nonzero_repairs']+=int(dn>0)
        reasons[info['reason']]=reasons.get(info['reason'],0)+1
        if window:
            corrected=[p.detach().clone() for p in bs]
            observation=self.diagnostic_losses(x,y,mask,base_values,corrected)
            observation.update(task=t,round=self.s['round'],client=self.s['next_client'],step=d['steps']-1,
                               before={'ce':float(loss.detach()),'cross':float(cross.detach()) if active else None},
                               mask_images=int(mask.sum()),images=len(y),repair=info)
            self.s.setdefault('finite_step_windows',[]).append(observation)
            d['diagnostic_forward_images']+=2*len(y);d['diagnostic_forward_calls']+=2
        if any(not torch.isfinite(p).all() for p in params):raise ArithmeticError('Nonfinite deployed parameter')
        d['repair_step_wall_seconds']+=time.perf_counter()-start
        return loss

    def train_client(self):
        if not self.enabled():return super().train_client()
        self._diagnostic={'steps':0,'images':0,'mask_images':0,'extra_backward_calls':0,'extra_backward_images':0,
                          'diagnostic_forward_images':0,'diagnostic_forward_calls':0,'momentum_help_ssb_hurt':0,
                          'triggers':0,'nonzero_repairs':0,'reasons':{},'sums':{},'maxima':{},'repair_step_wall_seconds':0.}
        client_started=time.perf_counter()
        s=self.s;k=s['next_client'];t=s['task'];r=s['round'];bs=self.model.b_parameters()
        global_b=[b.detach().clone() for b in bs];global_head=self.model.rows[-1].detach().clone()
        old_heads=[v.detach().clone() for v in self.model.rows[:-1]]
        ids=self.ids('train',t,k);n=len(ids)
        if n:
            dl=self.loader(self.raw[0],ids,self.cfg['batch'],True);count=self.cfg['epochs']*len(dl)
            momentum=[v.to(self.device) for v in s['client_states'][k]['m']]
            opt=ClientMuon(bs,momentum,[],[],False)
            headopt=torch.optim.AdamW([self.model.rows[-1]],lr=.001,weight_decay=0)
            steps=seen=0;loss_sum=0.
            for _ in range(self.cfg['epochs']):
                for x,y in dl:
                    self.budget();self.check_labels(y)
                    lr=self.cfg['lr']*.5*(1+math.cos(math.pi*(r*count+steps)/max(self.cfg['rounds']*count-1,1)))
                    self.model.zero_grad(set_to_none=True)
                    window=(k==0 and ((r==0 and steps in (0,count-1)) or (r==self.cfg['rounds']-1 and steps==count-1)))
                    loss=self.repair_step(self.train_tf(x.to(self.device)),y.to(self.device),opt,headopt,lr,window)
                    steps+=1;seen+=len(y);loss_sum+=float(loss.detach())*len(y)
            if steps!=count or seen!=n*self.cfg['epochs']:raise AssertionError('Optimizer budget')
            s['client_states'][k]={'m':[v.cpu() for v in momentum]}
            # This is the local endpoint encoder, BEFORE B FedAvg. Observation
            # restores RNG/modes so the extra pass does not change next-client RNG.
            gram=self.local_gram(k)
            if any(not torch.equal(a,b) for a,b in zip(old_heads,self.model.rows[:-1])):raise AssertionError('Old head changed')
            with torch.no_grad():
                for dst,b in zip(s['agg_B'],bs):dst.add_(b.detach().cpu(),alpha=n/len(self.ids('train',t)))
            s['heads'].append(self.model.rows[-1].detach().cpu().clone());s['grams'].append(gram);s['sizes'].append(n)
            rec={'task':t,'round':r,'client':k,'N':n,'visited':seen,'steps':steps,'lr_last':lr,'loss':loss_sum/seen}
            s['training_records'].append(rec);self.event('client',**rec)
            self._diagnostic.update(task=t,round=r,client=k,client_wall_seconds=time.perf_counter()-client_started,
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if self.device.type=='cuda' else 0,
                nonfinite_failures=0,numerical_checks_passed=True)
            s.setdefault('repair_records',[]).append(self._diagnostic)
            self.event('boundary_repair_client',**self._diagnostic)
            self.charge('B_upload',sum(b.numel()*4 for b in bs));self.charge('current_head_upload',global_head.numel()*4)
        with torch.no_grad():
            for b,v in zip(bs,global_b):b.copy_(v)
            self.model.rows[-1].copy_(global_head)
        s['next_client']+=1
        s['phase']='local_done' if s['next_client']==self.cfg['clients'] else 'clients'

