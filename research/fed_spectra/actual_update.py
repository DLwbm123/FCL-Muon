"""Shape-only server correction from actual local increments in a shared basis."""
import time
import torch
from .pilot import SpectraPilot


def global_shape_update(increments, weights, aggregate, eta, epsilon=1e-12,
                        degeneracy_tolerance=1e-10):
    """Interpolate spectral energy distributions, retaining aggregate norm/support.

    The target is measured along the aggregate's right singular directions;
    repeated singular values share a block-averaged target, avoiding arbitrary
    SVD coordinate choices. This is not a classification-loss proximal step.
    """
    if not 0 <= eta <= 1 or epsilon <= 0 or degeneracy_tolerance < 0:
        raise ValueError('Invalid spectral correction settings')
    base=aggregate.detach().double()
    ds=[d.detach().to(base) for d in increments]
    w=torch.as_tensor(weights,dtype=torch.float64,device=base.device)
    if (base.ndim != 2 or base.shape[0] < base.shape[1] or not ds or
        w.ndim != 1 or len(w) != len(ds) or (w < 0).any() or
        not torch.isfinite(w).all() or float(w.sum()) <= 0 or
        any(d.shape != base.shape or not torch.isfinite(d).all() for d in ds) or
        not torch.isfinite(base).all()):
        raise ValueError('Invalid actual increments, aggregate or weights')
    w=w/w.sum()
    energies=torch.stack([d.square().sum() for d in ds])
    mean_energy=float((w*energies).sum());norm=float(base.norm())
    record={'eta':eta,'input_norm':norm,'client_mean_squared_norm':mean_energy,
            'zero_clients':int((energies==0).sum()),'identity_reason':None}
    if norm == 0 or mean_energy == 0 or norm <= epsilon*mean_energy**.5:
        record.update(identity_reason='zero_or_numerical_cancellation',output_norm=norm,
                      relative_correction=0.,norm_relative_error=0.)
        return aggregate.clone(),record
    # Zero clients carry no shape; normalize weights over nonzero contributors.
    active=(energies>0)&(w>0)
    shape=sum(float(p)*(d.T@d)/e for d,p,e,keep in zip(ds,w,energies,active) if keep)
    shape=shape/w[active].sum()
    u,s,vh=torch.linalg.svd(base,full_matrices=False)
    supported=s > epsilon*s[0]
    target=(vh@shape@vh.T).diag().clamp_min(0)
    groups=[];start=0
    while start < len(s):
        end=start+1
        while end < len(s) and float(s[start]-s[end]) <= degeneracy_tolerance*float(s[0]):
            end+=1
        if end-start>1:
            target[start:end]=target[start:end].mean();groups.append([start,end])
        start=end
    p=(s[supported]/norm).square();p=p/p.sum()
    q=target[supported]+epsilon;q=q/q.sum()
    energy=torch.softmax((1-eta)*p.log()+eta*q.log(),dim=0)
    new_s=torch.zeros_like(s);new_s[supported]=energy.sqrt()*norm
    # A literal bypass makes eta=0 exactly match the existing FP32 endpoint.
    value=base if eta==0 else (u*new_s)@vh
    if eta==0:new_s=s.clone();energy=p
    out_norm=float(value.norm());norm_error=abs(out_norm-norm)/norm
    if not torch.isfinite(value).all() or norm_error>1e-10:
        raise ArithmeticError('Global shape correction changed the update norm')
    record.update(singular_values_before=s.tolist(),singular_values_after=new_s.tolist(),
        energy_before=p.tolist(),target_energy=q.tolist(),energy_after=energy.tolist(),
        repeated_singular_blocks=groups,retained_rank=int(supported.sum()),
        target_energy_outside_support=float(target[~supported].sum()),
        output_norm=out_norm,norm_relative_error=norm_error,
        effective_energy_rank_before=float(torch.exp(-(p*p.log()).sum())),
        effective_energy_rank_after=float(torch.exp(-(energy*energy.log()).sum())),
        singular_scale_min=float((new_s[supported]/s[supported]).min()),
        singular_scale_max=float((new_s[supported]/s[supported]).max()),
        relative_correction=float((value-base).norm())/norm,
        cosine_with_fedavg=float((value*base).sum())/(norm*out_norm),
        identity_reason='eta_zero' if eta==0 else None)
    return value,record


class ActualUpdatePilot(SpectraPilot):
    """Reuse RRS, CE/KL, Adam resets, head solve, evaluation and recovery."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.cfg['method']!='actual_update_global':raise ValueError('Wrong actual-update method')
        self.s.update(actual_update_endpoints=[],actual_update_rounds=[])

    def local_gram(self,k):
        self.s['actual_update_endpoints'].append(
            {'client':k,'B':[b.detach().cpu().clone() for b in self.model.b_parameters()]})
        return super().local_gram(k)

    def fuse(self):
        s=self.s;settings=self.cfg['actual_update'];started=time.perf_counter()
        t,r=s['task'],s['round'];entries=s['actual_update_endpoints']
        expected=[k for k in range(self.cfg['clients']) if self.ids('train',t,k)]
        if [v['client'] for v in entries]!=expected:raise ValueError('Actual endpoint/client mismatch')
        origin=[b.detach().cpu().clone() for b in self.model.b_parameters()]
        baseline=[b.clone() for b in s['agg_B']];deployed=[];layers={}
        for i,name in enumerate(self.model.adapters):
            records=[]
            for part in range(2):
                index=2*i+part
                ds=[v['B'][index].double()-origin[index].double() for v in entries]
                old=baseline[index].double()-origin[index].double()
                new,record=global_shape_update(ds,s['sizes'],old,**settings['operator'])
                value=baseline[index].clone() if record['identity_reason'] else (baseline[index].double()+new-old).float()
                actual=value.double()-origin[index].double()
                norm_error=abs(float(actual.norm())-float(old.norm()))/max(float(old.norm()),1e-30)
                if not torch.isfinite(value).all() or norm_error>settings['deployed_norm_tolerance']:
                    raise ArithmeticError('Deployed FP32 shape correction violated norm tolerance')
                record.update(actual_FP32_norm=float(actual.norm()),actual_FP32_norm_relative_error=norm_error,
                              actual_FP32_relative_correction=float((actual-old).norm())/max(float(old.norm()),1e-30))
                deployed.append(value);records.append(record)
            layers[name]=records
        super().fuse()
        with torch.no_grad():
            for b,v in zip(self.model.b_parameters(),deployed):b.copy_(v)
        s['actual_update_rounds'].append({'task':t+1,'round':r+1,'layers':layers,'seconds':time.perf_counter()-started})
        s['actual_update_endpoints']=[]
        self.event('actual_update_aggregate',eta=settings['operator']['eta'],
                   maximum_deployed_norm_relative_error=max(v['actual_FP32_norm_relative_error'] for rs in layers.values() for v in rs))
