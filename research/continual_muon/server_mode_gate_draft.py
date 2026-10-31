"""Unregistered task-history and client-dispersion spectral gate; no real-data run."""
import math
import torch


def _psd(x, reference_scale=None):
    if x.ndim!=2 or x.shape[0]!=x.shape[1] or not torch.isfinite(x).all():raise ValueError('invalid moment matrix')
    x=(x.double()+x.double().T)/2
    vals,vecs=torch.linalg.eigh(x)
    scale=float(vals.abs().max()) if reference_scale is None else float(reference_scale)
    if not math.isfinite(scale) or scale<0:raise ValueError('invalid PSD reference scale')
    tolerance=1e-10*max(scale,1e-30)
    if float(vals.min()) < -tolerance:raise ValueError('materially indefinite moment matrix')
    return (vecs*vals.clamp_min(0))@vecs.T


def aggregate_update_dispersion(updates,weights):
    """Weighted true local-update Gram dispersion; no raw observations returned."""
    if not updates or len(updates)!=len(weights) or any(x.ndim!=2 or x.shape!=updates[0].shape for x in updates):
        raise ValueError('invalid update shapes')
    if any(not math.isfinite(w) or w<0 for w in weights) or abs(sum(weights)-1)>1e-12:
        raise ValueError('invalid weights')
    if any(not torch.isfinite(v).all() for v in updates):raise ValueError('invalid update')
    avg=sum(w*u.double() for w,u in zip(weights,updates))
    # Direct centered form avoids cancellation from E[X^T X]-E[X]^T E[X].
    cov=sum(w*((u.double()-avg).T@(u.double()-avg)) for w,u in zip(weights,updates))
    return avg,_psd(cov)


def dispersion_from_moments(mean_update, second_moment):
    """Server-side E[D^T D]-E[D]^T E[D] with FP64 PSD validation."""
    if (mean_update.ndim!=2 or second_moment.shape!=(mean_update.shape[1],mean_update.shape[1]) or
            not torch.isfinite(mean_update).all() or not torch.isfinite(second_moment).all()):
        raise ValueError('invalid accumulated moments')
    mu=mean_update.double();second=second_moment.double();outer=mu.T@mu
    scale=max(float(torch.linalg.matrix_norm(second,ord=2)),float(torch.linalg.matrix_norm(outer,ord=2)),1e-30)
    return _psd(second-outer,reference_scale=scale)


def gated_update(update,history,current,dispersion,tolerance=1e-10):
    """Filter true Muon aggregate in shared basis; history is only a proxy."""
    if update.ndim!=2 or not torch.isfinite(update).all() or not math.isfinite(tolerance) or tolerance<0:
        raise ValueError('invalid update or tolerance')
    rank=update.shape[1]
    if any(x.shape!=(rank,rank) for x in (history,current,dispersion)):raise ValueError('moment shape mismatch')
    H,S,C=(_psd(x) for x in (history,current,dispersion))
    def fallback(reason,**extra):return update.clone(),{'accepted':False,'reason':reason,**extra}
    def normalize(x,required):
        scale=float(torch.trace(x)/rank)
        if scale<=1e-30:return None if required else torch.zeros_like(x)
        return x/scale
    H=normalize(H,True);S=normalize(S,True);C=normalize(C,False)
    if H is None or S is None:return fallback('empty_required_moment')
    total=torch.eye(rank,dtype=H.dtype,device=H.device)+H+S+C
    eigen,vectors=torch.linalg.eigh(total)
    if float(eigen.min())<=0:raise ArithmeticError('invalid gate metric')
    inv=(vectors*eigen.rsqrt())@vectors.T
    gate=(inv@S@inv);gate=(gate+gate.T)/2
    gvals=torch.linalg.eigvalsh(gate)
    if float(gvals.min()) < -1e-9 or float(gvals.max()) > 1+1e-9:raise ArithmeticError('gate is not contraction')
    base=update.double();candidate=(base@gate).to(update.dtype);new=candidate.double()
    old_hist=float(((base@H)*base).sum());new_hist=float(((new@H)*new).sum())
    old_norm=float(torch.linalg.vector_norm(base));new_norm=float(torch.linalg.vector_norm(new))
    receipt={'history_before':old_hist,'history_after':new_hist,'norm_before':old_norm,'norm_after':new_norm,
             'gate_min_eigenvalue':float(gvals.min()),'gate_max_eigenvalue':float(gvals.max())}
    if (not torch.isfinite(candidate).all() or new_hist>old_hist+tolerance*max(old_hist,1.) or
            new_norm>old_norm*(1+1e-6)+1e-12):return fallback('deployed_budget',**receipt)
    return candidate,{'accepted':True,'reason':'accepted',**receipt}


def gated_parameter(origin,baseline,history,current,dispersion,tolerance=1e-10,gate_history=None):
    """Recheck the actual FP32 parameter after origin+filtered-update addition."""
    if origin.shape!=baseline.shape or origin.dtype!=baseline.dtype:raise ValueError('incompatible parameters')
    update=baseline.double()-origin.double()
    filtered,receipt=gated_update(update,history if gate_history is None else gate_history,current,dispersion,tolerance=tolerance)
    if not receipt['accepted']:return baseline.clone(),receipt
    actual=(origin.double()+filtered.double()).to(origin.dtype)
    if torch.equal(actual,baseline):return baseline.clone(),{'accepted':False,'reason':'no_physical_change'}
    H=_psd(history);scale=float(torch.trace(H)/H.shape[0])
    if scale<=1e-30:return baseline.clone(),{'accepted':False,'reason':'empty_required_moment'}
    H=H/scale;new=actual.double()-origin.double()
    old_hist=float(((update@H)*update).sum());new_hist=float(((new@H)*new).sum())
    old_norm=float(torch.linalg.vector_norm(update));new_norm=float(torch.linalg.vector_norm(new))
    record={'history_before':old_hist,'history_after':new_hist,'norm_before':old_norm,'norm_after':new_norm,
            'gate_min_eigenvalue':receipt['gate_min_eigenvalue'],'gate_max_eigenvalue':receipt['gate_max_eigenvalue']}
    if (not torch.isfinite(actual).all() or new_hist>old_hist+tolerance*max(old_hist,1.) or
            new_norm>old_norm*(1+1e-6)+1e-12):
        return baseline.clone(),{'accepted':False,'reason':'post_addition_budget',**record}
    return actual,{'accepted':True,'reason':'accepted',**record}
