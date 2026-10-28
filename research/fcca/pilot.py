"""Global raw-risk aggregation with exact finite probe checks and archival guard.

No historical images, no local boundary repair, no head center normalization.
The archival response guard is explicitly NOT an old-label classification bound.
"""
import contextlib
import time
import torch
from torch.nn import functional as F
from fcl.checkpoint import rng_state,restore_rng
from research.aggregation_utility.pilot import UtilityPilot
from research.head_regmean.solver import aggregate_heads
from .math import directions,candidate,directional,historical_gram,historical_energy,propose,finite_acceptance


def probe_split(indices,task,client,settings):
    ids=sorted(indices);g=torch.Generator().manual_seed(settings['probe_seed']+10007*task+101*client)
    order=torch.randperm(len(ids),generator=g).tolist()
    take=min(len(ids),2*settings['probe_per_split'])
    chosen=[ids[i] for i in order[:take]]
    nfit=(take+1)//2
    return sorted(chosen[:nfit]),sorted(chosen[nfit:])


@contextlib.contextmanager
def probe_context(model):
    rng=rng_state();modes={m:m.training for m in model.modules()}
    flags={p:p.requires_grad for p in model.parameters()}
    buffers={n:b.detach().clone() for n,b in model.named_buffers()}
    try:
        model.eval()
        for p in flags:p.requires_grad_(False)
        for p in model.b_parameters():p.requires_grad_(True)
        yield
    finally:
        with torch.no_grad():
            for n,b in model.named_buffers():b.copy_(buffers[n])
        for p,v in flags.items():p.requires_grad_(v)
        for m,v in modes.items():m.training=v
        restore_rng(rng)


class FccaPilot(UtilityPilot):
    def enabled(self):return self.cfg['fcca']['enabled'] and self.s['task']>0

    def initialize_task(self):
        super().initialize_task()
        if self.enabled():
            t=self.s['task']
            self.s['fcca_probe_ids']=[probe_split(self.ids('train',t,k),t,k,self.cfg['fcca']) for k in range(self.cfg['clients'])]
            self.s['fcca_endpoints']=[]
            self.event('fcca_probes',counts=[[len(a),len(b)] for a,b in self.s['fcca_probe_ids']],
                       scope='current training only; selector-disjoint fit/check, not independent generalization holdout')

    def local_gram(self,k):
        # This inherited hook runs after local training and before the original
        # trainer restores the round-global B/head. Keep exact local endpoints.
        if self.enabled():
            self.s.setdefault('fcca_endpoints',[]).append({'client':k,'B':[b.detach().cpu().clone() for b in self.model.b_parameters()]})
        return super().local_gram(k)

    def history_metrics(self):
        if self.s['history'].committed!=set(range(self.s['task'])):raise ValueError('Wrong archival history scope')
        qs=[]
        for name,a in self.model.adapters.items():
            h=self.s['history'].raw[name].double();h=(h+h.T)/2
            scale=max(float(h.diag().mean()),1e-12);aa=a.A.detach().cpu().double()
            q=aa@h@aa.T/scale;q=(q+q.T)/2
            if not torch.isfinite(q).all():raise ArithmeticError('Nonfinite archive metric')
            qs.extend([q,q])
        return qs

    def measure(self,part,ds=None):
        """Every client's probe sees the SAME actual candidate B and fixed head.

        Fit uploads one FP32 raw-global CE gradient per active client. The
        server takes its dot products with all client directions (cross terms).
        """
        t=self.s['task'];c=self.cfg['cpt'];bs=self.model.b_parameters()
        counts=[len(self.ids('train',t,k)) if self.s['fcca_probe_ids'][k][part] else 0 for k in range(self.cfg['clients'])]
        denominator=sum(counts)
        if not denominator:return None
        total={name:0. for name in ('raw_ce','within_ce','centered_cross','raw_accuracy','centered_accuracy','within_accuracy','raw_current_to_old','centered_current_to_old')}
        means=torch.zeros(t+1,dtype=torch.float64);matrix=[];clients=[];weights=[];images=forwards=backwards=0
        with probe_context(self.model):
            for k,n in enumerate(counts):
                if not n:continue
                ids=self.s['fcca_probe_ids'][k][part];nk=len(ids);p=n/denominator
                local={name:0. for name in total};localmeans=torch.zeros(t+1,dtype=torch.float64)
                grads=[torch.zeros_like(b,device='cpu',dtype=torch.float64) for b in bs] if ds is not None else None
                for x,y in self.loader(self.raw[0],ids,self.cfg['batch']):
                    self.budget();self.check_labels(y);y=y.to(self.device)
                    with torch.set_grad_enabled(ds is not None):
                        z=self.model(self.test_tf(x.to(self.device)))
                        raw=F.cross_entropy(z,y,reduction='sum')
                        if ds is not None:
                            gg=torch.autograd.grad(raw/nk,bs)
                            for g,v in zip(grads,gg):g.add_(v.detach().cpu().double())
                            backwards+=1
                    zz=z.detach();blocks=zz.reshape(len(y),t+1,c);center=(blocks-blocks.mean(-1,keepdim=True)).reshape_as(zz)
                    local['raw_ce']+=float(raw.detach());local['within_ce']+=float(F.cross_entropy(zz[:,t*c:],y-t*c,reduction='sum'))
                    local['centered_cross']+=float(F.softplus(center[:,:t*c].max(1).values-center.gather(1,y[:,None]).squeeze(1)).sum())
                    rp=zz.argmax(1);cp=center.argmax(1)
                    local['raw_accuracy']+=int((rp==y).sum());local['centered_accuracy']+=int((cp==y).sum())
                    local['within_accuracy']+=int((zz[:,t*c:].argmax(1)==y-t*c).sum())
                    local['raw_current_to_old']+=int((rp<t*c).sum());local['centered_current_to_old']+=int((cp<t*c).sum())
                    localmeans+=blocks.mean(-1).double().sum(0).cpu();images+=len(y);forwards+=1
                for name in total:total[name]+=p*local[name]/nk
                means+=p*localmeans/nk
                if grads is not None:
                    uploaded=[g.float() for g in grads]
                    if any(not torch.isfinite(g).all() for g in uploaded):raise ArithmeticError('Nonfinite probe gradient')
                    matrix.append(directional(uploaded,ds));clients.append(k);weights.append(p)
                    self.charge('fcca_fit_gradient_upload',sum(g.numel()*g.element_size() for g in uploaded))
        if any(not torch.isfinite(torch.tensor(v)) for v in total.values()):raise ArithmeticError('Nonfinite probe risk')
        total.update(task_block_means=means.tolist(),probe_images=images,represented_training_images=denominator,
                     forward_calls=forwards,backward_calls=backwards,client_probe_counts=[len(v[part]) for v in self.s['fcca_probe_ids']])
        if ds is not None:
            matrix=torch.stack(matrix);w=torch.tensor(weights,dtype=torch.float64)
            global_derivative=w@matrix
            total.update(global_raw_derivative=global_derivative.tolist(),fit_clients=clients,fit_client_weights=weights,
                         cross_client_derivative_matrix=matrix.tolist())
        # Eight scalar means, (t+1) task means, and two int64 sample counts.
        self.charge('fcca_probe_scalar_upload',sum(v>0 for v in counts)*(8+(t+1)+2)*8)
        return total

    def fuse(self):
        if not self.enabled():return super().fuse()
        s=self.s;cfg=self.cfg['fcca'];start=time.perf_counter()
        bs=self.model.b_parameters();origin=[b.detach().cpu().clone() for b in bs]
        baseline=[b.clone() for b in s['agg_B']]
        head,_,cert=aggregate_heads(s['heads'],s['grams'],s['sizes'],True,alpha=.5)
        entries=s['fcca_endpoints'];endpoints=[r['B'] for r in entries]
        expected=[k for k in range(self.cfg['clients']) if self.ids('train',s['task'],k)]
        if [r['client'] for r in entries]!=expected or len(entries)!=len(s['sizes']):raise ValueError('Endpoint/client alignment')
        weights=torch.tensor(s['sizes'],dtype=torch.float64);weights/=weights.sum()
        ds=directions(endpoints,baseline,origin,weights);qs=self.history_metrics();gram=historical_gram(ds,qs)
        old_heads=[w.detach().clone() for w in self.model.rows[:-1]]
        def deploy(values):
            with torch.no_grad():
                for b,v in zip(bs,values):b.copy_(v)
                self.model.rows[-1].copy_(head)
        deploy(baseline)
        # New logical messages, on top of unchanged local-training communication.
        model_bytes=sum(b.numel()*b.element_size() for b in baseline)
        self.charge('fcca_reference_B_broadcast',len(entries)*model_bytes)
        self.charge('fcca_fixed_head_broadcast',len(entries)*head.numel()*head.element_size())
        base_fit=self.measure(0,ds);base_check=self.measure(1)
        if base_fit is None:raise ValueError('No fit probes')
        alpha,solve=propose(weights.numpy(),base_fit['global_raw_derivative'],gram.numpy(),cfg)
        proposed=candidate(baseline,ds,weights,alpha)
        base_energy=historical_energy(baseline,origin,qs);proposed_energy=historical_energy(proposed,origin,qs)
        actual_limit=base_energy*(1+cfg['history_relative_tolerance'])+cfg['history_absolute_tolerance']
        candidate_fit=candidate_check=None;accepted=False;reasons=[solve['reason']]
        if solve['success']:
            deploy(proposed)
            self.charge('fcca_candidate_B_broadcast',len(entries)*model_bytes)
            candidate_fit=self.measure(0);candidate_check=self.measure(1)
            accepted,reasons=finite_acceptance(base_fit,candidate_fit,base_check,candidate_check,cfg)
            if proposed_energy>actual_limit:
                accepted=False;reasons.append('actual_FP32_history_guard')
        deployed=proposed if accepted else baseline;deploy(deployed)
        if any(not torch.equal(a,b) for a,b in zip(old_heads,self.model.rows[:-1])):raise AssertionError('Old head changed')
        if not torch.equal(self.model.rows[-1].detach().cpu(),head):raise AssertionError('Candidate head mismatch')
        deployed_alpha=alpha.tolist() if accepted else weights.tolist()
        record={'task':s['task']+1,'round':s['round']+1,'accepted':accepted,'reasons':reasons,
                'clients':expected,'sample_weights':weights.tolist(),'proposed_weights':alpha.tolist(),'deployed_weights':deployed_alpha,
                'solver':solve,'history_baseline':base_energy,'history_candidate_actual_FP32':proposed_energy,'history_limit':actual_limit,
                'baseline_fit':base_fit,'baseline_check':base_check,'candidate_fit':candidate_fit,'candidate_check':candidate_check,
                'extra_forward_images':sum(v['probe_images'] for v in [base_fit,base_check,candidate_fit,candidate_check] if v),
                'extra_backward_images':base_fit['probe_images'],'extra_forward_calls':sum(v['forward_calls'] for v in [base_fit,base_check,candidate_fit,candidate_check] if v),
                'extra_backward_calls':base_fit['backward_calls'],'seconds':time.perf_counter()-start,
                'fixed_actual_HeadRegMean':True,'check_used_once':True,'old_class_guarantee':False}
        s.setdefault('fcca_records',[]).append(record)
        s['certificates'].append({'task':s['task'],'round':s['round'],**cert})
        self.event('fcca_aggregate',accepted=accepted,reasons=reasons,raw_fit=base_fit['raw_ce'],
                   raw_check=base_check['raw_ce'] if base_check else None,weights=deployed_alpha)
        s.update(round=s['round']+1,heads=[],grams=[],sizes=[],agg_B=[],fcca_endpoints=[],next_client=0,phase='round_start')
