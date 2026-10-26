"""Match local head responses in the deployed FedAvg encoder's coordinates."""
import torch
from torch.nn import functional as F
from fcl.checkpoint import atomic_save
from .old_response import OldResponsePilot
from .pilot import observe
from .solver import shrink_gram,sphere_solve,certify_sphere,objective


def shrink_cross(c,alpha=.5):
    # C=Z_local.T@Z_global is generally NOT symmetric. The same-coordinate
    # diagonal coupling corresponds to shared multiplicative feature noise.
    if c.ndim!=2 or c.shape[0]!=c.shape[1] or not 0<=alpha<=1:raise ValueError('Invalid paired cross moment')
    return alpha*c.double()+(1-alpha)*torch.diag(c.double().diag())


def aggregate_transport(heads,global_grams,cross_moments,sizes,alpha=.5):
    if not heads or not len(heads)==len(global_grams)==len(cross_moments)==len(sizes) or min(sizes)<=0:raise ValueError('Invalid paired endpoint collection')
    if any(not torch.isfinite(w).all() or bool((w.norm(dim=1)<=1e-12).any()) for w in heads):raise ValueError('Invalid cosine head')
    total=sum(sizes);h=sum(shrink_gram(g,alpha)/total for g in global_grams)
    q=sum(F.normalize(w.float(),dim=1).double()@shrink_cross(c,alpha)/total for w,c in zip(heads,cross_moments))
    u,lam,lower,info=sphere_solve(h,q)
    raw=torch.zeros_like(heads[0],dtype=torch.float32)
    for w,n in zip(heads,sizes):raw.add_(w.float(),alpha=n/total)
    deployed=(u*raw.double().norm(dim=1).clamp_min(1e-12)[:,None]).float()
    actual=F.normalize(deployed,dim=1).double();baseline=F.normalize(raw,dim=1).double()
    cert=certify_sphere(actual,h,q,lam,lower)
    info.update(deployed_fp32=cert,alpha=alpha,raw_moment_samples=total,
        baseline_objective=float(objective(baseline,h,q).sum()),
        regmean_objective=float(objective(actual,h,q).sum()),
        objective_note='Paired local-to-FedAvg-global normalized CLS response matching, with alpha shrink of global Gram and nonsymmetric local.T@global cross moment; unit-row constraint and raw FedAvg norms. Not the former local-coordinate objective; client target-only constant omitted.')
    if not cert['passed']:raise ArithmeticError(('Transport FP32 deployment failed',info))
    if info['regmean_objective']>info['baseline_objective']+1e-6*(1+abs(info['baseline_objective'])):raise ArithmeticError('Transport solve worse than feasible raw FedAvg')
    return deployed,raw,info


class TransportHeadPilot(OldResponsePilot):
    expected_first_round=None

    def local_gram(self,k):
        # The original learner calls this before restoring the round-start B.
        # No forward/RNG action here: retain the already-uploaded local B only.
        self.s.setdefault('transport_endpoints',{})[k]=[v.detach().cpu().clone() for v in self.model.b_parameters()]
        return None

    @torch.no_grad()
    def paired_moments(self,k,local_b,global_b):
        bs=self.model.b_parameters();d=self.model.backbone.num_features
        gl=torch.zeros(d,d,dtype=torch.float64,device=self.device);gg=torch.zeros_like(gl);cross=torch.zeros_like(gl)
        n=0;drift=0.
        def assign(values):
            for b,v in zip(bs,values):b.copy_(v)
        with observe(self.model):
            try:
                for x,y in self.loader(self.raw[0],self.ids('train',self.s['task'],k),self.cfg['batch']):
                    self.budget();self.check_labels(y);x=self.test_tf(x.to(self.device))
                    assign(global_b);zg=F.normalize(self.model.backbone.forward_features(x)[:,0],dim=1).double()
                    assign(local_b);zl=F.normalize(self.model.backbone.forward_features(x)[:,0],dim=1).double()
                    gg.add_(zg.T@zg);gl.add_(zl.T@zl);cross.add_(zl.T@zg)
                    n+=len(y);drift+=float((zl-zg).square().sum())
            finally:assign(global_b)
        if n!=len(self.ids('train',self.s['task'],k)) or not all(torch.isfinite(v).all() for v in [gl,gg,cross]):raise ArithmeticError('Invalid paired statistics')
        self.charge('transport_global_Gram_upload',gg.numel()*8)
        self.charge('transport_cross_moment_upload',cross.numel()*8)
        self.charge('transport_diagnostic_scalars_upload',(2*self.cfg['cpt']+1)*8)
        for kind in ['paired_local_endpoint','paired_deployed_global']:
            self.s['stats_records'].append({'task':self.s['task'],'round':self.s['round'],'client':k,'kind':kind,'samples':n})
        self.event('paired_statistics',client=k,samples=n,encoder_forward_images=2*n,feature_squared_drift_sum=drift)
        return gl.cpu(),gg.cpu(),cross.cpu(),drift

    def fuse(self):
        s=self.s;endpoints=s['transport_endpoints'];keys=sorted(endpoints)
        if len(keys)!=len(s['heads']) or [len(self.ids('train',s['task'],k)) for k in keys]!=s['sizes']:raise ValueError('Paired endpoint alignment failed')
        if any(not torch.isfinite(b).all() for b in s['agg_B']):raise ArithmeticError('Invalid FedAvg B')
        raw=torch.zeros_like(s['heads'][0],dtype=torch.float32)
        for w,n in zip(s['heads'],s['sizes']):raw.add_(w.float(),alpha=n/sum(s['sizes']))
        if self.expected_first_round is not None:
            ref=self.expected_first_round
            checks={'B_exact':len(ref['B'])==len(s['agg_B']) and all(torch.equal(x,y) for x,y in zip(ref['B'],s['agg_B'])),
                    'raw_FedAvg_head_exact':torch.equal(ref['raw_FedAvg_head'],raw)}
            report={'passed':all(checks.values()),'checks':checks,'task':s['task'],'round':s['round'],
                    'scope':'Before first changed fusion; existing old_response control, no control training repeated'}
            (self.out/'FIRST_ROUND_PAIR.json').write_text(__import__('json').dumps(report,indent=2))
            if not report['passed']:raise ArithmeticError(('First-round pairing failed; no automatic retry',report))
            s['transport_first_round_pair']=report;self.expected_first_round=None
        global_b=[v.to(self.device) for v in s['agg_B']]
        self.charge('transport_global_B_download',self.cfg['clients']*sum(v.numel()*4 for v in global_b))
        gg=[];cross=[];raw_const=shrunk_const=0.;drift=0.
        for k,w in zip(keys,s['heads']):
            gl,g,c,d=self.paired_moments(k,[v.to(self.device) for v in endpoints[k]],global_b)
            gg.append(g);cross.append(c);drift+=d;u=F.normalize(w.float(),dim=1).double()
            raw_const+=float(((u@gl)*u).sum())
            shrunk_const+=float(((u@shrink_gram(gl,.5))*u).sum())
        head,raw,cert=aggregate_transport(s['heads'],gg,cross,s['sizes'])
        total=sum(s['sizes']);actual=F.normalize(head,dim=1).double();base=F.normalize(raw,dim=1).double()
        h=sum(gg)/total;q=sum(F.normalize(w.float(),dim=1).double()@c for w,c in zip(s['heads'],cross))/total
        cert.update(paired_feature_squared_drift_mean=drift/total,
            raw_response_MSE=float((objective(actual,h,q).sum()+raw_const/total)/head.shape[0]),
            raw_FedAvg_response_MSE=float((objective(base,h,q).sum()+raw_const/total)/head.shape[0]),
            shrunk_response_MSE=(cert['regmean_objective']+shrunk_const/total)/head.shape[0],
            paired_encoder_forward_images=2*total,paired_samples=total)
        # raw MSE is descriptive: the fixed alpha.5 objective, not raw MSE, was minimized.
        with torch.no_grad():
            for b,v in zip(self.model.b_parameters(),global_b):b.copy_(v)
            self.model.rows[-1].copy_(head)
        s['certificates'].append({'task':s['task'],'round':s['round'],**cert})
        atomic_save({'task':s['task'],'round':s['round'],'B':s['agg_B'],'head':head,
            'raw_FedAvg_head':raw,'certificate':cert},self.out/f"round_t{s['task']}_r{s['round']+1}.pt")
        self.event('aggregate',certificate=cert)
        s.update(round=s['round']+1,heads=[],grams=[],sizes=[],agg_B=[],next_client=0,phase='round_start',transport_endpoints={})
