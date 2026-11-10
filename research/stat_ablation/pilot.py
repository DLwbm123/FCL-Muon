import copy
import torch
from torch.nn import functional as F
from research.stat_head.pilot import HeadPilot
from research.head_regmean.pilot import observe
from research.adamw_t10.pilot import metrics,centered_logits
from .statistics import local_packet,merge,class_wire,dense_packet
from .response import federated_response


class AblationPilot(HeadPilot):
    functional_response=staticmethod(federated_response)

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.s['stat_branches']={mode:None for mode in self.cfg['branch_modes']}

    @property
    def response_options(self):
        return dict(protocol='unified',old=self.s.get('stat_anchor'),projection=self.R.to(self.device),
                    head_rank=self.cfg['head_rank'],transport=self.cfg['primary_mode'])

    def initialize_task(self):
        self.s['branch_anchors']=copy.deepcopy(self.s['stat_branches'])
        self.s['local_coordinate_packets']={};self.s['conditional_readouts']={}
        super().initialize_task()
        if self.s['task']+1 in self.cfg['diagnostic_tasks'] and self.s['task']>0:
            # Evaluator-only old TEST features: never consumed by fit/selection.
            z,y=self.evaluation_features(old_only=True)
            self.s['evaluator_old_anchor']=(z.cpu(),y.cpu())
        else:self.s['evaluator_old_anchor']=None

    def local_gram(self,k):
        if self.s['task']+1 in self.cfg['diagnostic_tasks'] and self.s['round']==self.cfg['rounds']-1:
            with observe(self.model):
                z,y=self._features(k)
                self.s['local_coordinate_packets'][k]=local_packet(z,z,y,self.cfg['cpt'],self.R,self.cfg['head_rank'])[0]
        return torch.empty(0)

    def fuse(self):
        s=self.s;bs=self.model.b_parameters();modes=self.cfg['branch_modes'];primary=self.cfg['primary_mode']
        with observe(self.model):
            if primary!='identity' and not s['anchor_features']:
                snapshot=[b.clone() for b in bs]
                try:
                    for b in bs:b.zero_()
                    for k in range(self.cfg['clients']):s['anchor_features'][k]=self._features(k)
                finally:
                    for b,v in zip(bs,snapshot):b.copy_(v)
            for b,v in zip(bs,s['agg_B']):b.copy_(v)
            total=sum(s['sizes'])
            self.model.rows[-1].copy_(sum(h*(n/total) for h,n in zip(s['heads'],s['sizes'])))
            packets=[];dense_packets=[];wires={mode:0 for mode in modes}
            diagnostic=s['task']+1 in self.cfg['diagnostic_tasks'] and s['round']==self.cfg['rounds']-1
            for k in range(self.cfg['clients']):
                z,y=self._features(k)
                z0,y0=(z,y) if primary=='identity' else s['anchor_features'][k]
                if not torch.equal(y,y0):raise ValueError('anchor mismatch')
                packet,wire,record=local_packet(z0,z,y,self.cfg['cpt'],self.R,self.cfg['head_rank'])
                if primary=='identity':packet={key:packet[key] for key in ('counts','means','scatter')}
                packets.append(packet)
                for mode in modes:wires[mode]+=len(class_wire(wire,z.shape[1],self.R.shape[1],self.cfg['cpt'],mode))
                if diagnostic:
                    dense=copy.copy(packet);exact,_=dense_packet(z,y,self.cfg['cpt'],self.R)
                    dense['scatter']=exact['scatter'];dense_packets.append(dense)
                record.update(task=s['task'],round=s['round'],client=k)
                s.setdefault('head_codec_records',[]).append(record)
            s.setdefault('branch_statistics_upload',{mode:0 for mode in modes})
            s['branch_heads']={};maps={}
            for mode in modes:
                stats,W,beta,cert,mapping=merge(packets,s['branch_anchors'][mode],self.R,mode=mode,return_map=True)
                if int(stats['counts'].sum())!=(s['task']+1)*len(self.ids('train',s['task'])):raise AssertionError('count duplication')
                s['stat_branches'][mode]={k:v.detach().cpu() for k,v in stats.items()}
                s['branch_heads'][mode]=(W.detach().cpu(),beta.detach().cpu())
                s['branch_statistics_upload'][mode]+=wires[mode];maps[mode]=tuple(v.cpu() for v in mapping)
                if mode==primary:s['certificates'].append(dict(task=s['task'],round=s['round'],**cert))
            s['unified_stats']=s['stat_branches'][primary]
            self.model.deploy(*s['branch_heads'][primary])
            self.charge('unified_statistics_upload',wires[primary])
            W,beta=s['branch_heads'][primary]
            self.charge('all_seen_head_download',self.cfg['clients']*(W.numel()+beta.numel())*4)
            if diagnostic:
                s['evaluator_maps']=maps
                s['conditional_readouts']={}
                for label,payload in [('dense_current_scatter',dense_packets),('local_current_coordinates',list(s['local_coordinate_packets'].values()))]:
                    if len(payload)!=self.cfg['clients']:raise ValueError('diagnostic missing local endpoint')
                    if label=='local_current_coordinates':
                        # Keep the SAME globally-fitted old-class transport in
                        # both coordinate comparisons; alter only new statistics.
                        transported=s['stat_branches'][primary]
                        count=self.cfg['cpt'];oldn=len(transported['counts'])-count
                        old_mean=transported['means'][:oldn];old_count=transported['counts'][:oldn]
                        global_new,_,_,_=merge(packets,None,self.R,mode='identity')
                        old_v=transported['scatter']-global_new['scatter']
                        pseudo={'counts':old_count,'means':old_mean,'scatter':old_v} if oldn else None
                        _,w,b,_=merge(payload,pseudo,self.R,mode='identity')
                    else:_,w,b,_=merge(payload,s['branch_anchors'][primary],self.R,mode=primary)
                    s['conditional_readouts'][label]=(w,b)
                stats=s['stat_branches'][primary];cov=stats['scatter']/stats['counts'].sum();d=len(cov)
                ridge=.1*cov.trace()/d+1e-6;means=stats['means'];eye=torch.eye(d,dtype=cov.dtype)
                for name,metric in [('diagonal_metric',torch.diag(cov.diag())),('isotropic_metric',eye*cov.trace()/d)]:
                    w=torch.linalg.solve(metric+ridge*eye,means.T).T;b=-.5*(w*means).sum(1)
                    s['conditional_readouts'][name]=(w,b)
                s['conditional_readouts']['no_bias']=(W,torch.zeros_like(beta))
        self.event('aggregate',branch_modes=modes)
        s.update(round=s['round']+1,heads=[],grams=[],sizes=[],agg_B=[],next_client=0,phase='round_start',local_coordinate_packets={})

    def evaluation_features(self,old_only=False):
        zs=[];ys=[];end=self.s['task'] if old_only else self.s['task']+1
        with observe(self.model):
            for i in range(end):
                for x,y in self.loader(self.raw[1],self.ids('test',i),self.cfg['batch']):
                    self.budget();zs.append(F.normalize(self.model.backbone.forward_features(self.test_tf(x.to(self.device)))[:,0],dim=1).cpu());ys.append(y.cpu())
        return torch.cat(zs),torch.cat(ys)

    def evaluate(self,save_artifacts=True):
        z,y=self.evaluation_features();s=self.s;c=self.cfg['cpt'];n=(s['task']+1)*c
        outputs={}
        for mode,(W,beta) in s['branch_heads'].items():
            logits=F.linear(z.to(self.device),W.float().to(self.device),beta.float().to(self.device)).cpu()
            cm=torch.bincount(y*n+logits.argmax(1),minlength=n*n).reshape(n,n)
            cc=torch.bincount(y*n+centered_logits(logits,c).argmax(1),minlength=n*n).reshape(n,n)
            prev=[item['branch_readouts'][mode]['per_task'] for item in s['stages'][:s['task']]]
            result=metrics(cm,c,prev)
            previous=[item['branch_readouts'][mode]['centered']['per_task'] for item in s['stages'][:s['task']]]
            result['centered']=metrics(cc,c,previous);outputs[mode]=result
        result=copy.deepcopy(outputs[self.cfg['primary_mode']]);result['branch_readouts']=outputs
        if s.get('conditional_readouts'):
            result['conditional_readouts']={name:float((F.linear(z,W.float(),b.float()).argmax(1)==y).double().mean()) for name,(W,b) in s['conditional_readouts'].items()}
        if s.get('evaluator_old_anchor') is not None:
            z0,y0=s['evaluator_old_anchor'];current=z[:len(z0)].double()
            if not torch.equal(y[:len(z0)],y0):raise ValueError('audit sample ordering changed')
            result['evaluator_only_transfer']={}
            old_classes=s['task']*c
            mu0=torch.stack([z0[y0==i].double().mean(0) for i in range(old_classes)])
            mur=torch.stack([current[y0==i].mean(0) for i in range(old_classes)])
            v0=sum((z0[y0==i].double()-mu0[i]).T@(z0[y0==i].double()-mu0[i]) for i in range(old_classes))
            vr=sum((current[y0==i]-mur[i]).T@(current[y0==i]-mur[i]) for i in range(old_classes))
            denom=float((current-z0.double()).square().sum())
            for mode,(T,b) in s['evaluator_maps'].items():
                pred=z0.double()@T.T+b
                result['evaluator_only_transfer'][mode]=dict(response_relative_error=float((pred-current).square().sum())/(denom+1e-30),response_denominator=denom,
                    mean_error=float((mu0@T.T+b-mur).norm(dim=1).mean()),
                    scatter_relative_error=float(((T@v0@T.T if mode=='full' else v0)-vr).norm()/vr.norm().clamp_min(1e-30)))
        return result
