"""Unregistered history-whitened Muon update consensus; synthetic development only."""
import math
import torch
from fcl.math import mulora_ssb
from research.continual_muon.server_mode_gate_draft import _psd


def aggregate(origin,baseline,local_updates,weights,history,*,use_history=True,tolerance=1e-10):
    """Polar consensus for genuine client deltas; physical FP32 gate and fallback.

    `history` is a past-response proxy, not an old-class safety guarantee.
    The method itself, data, budgets and statistical gates are not registered.
    """
    if (origin.ndim!=2 or origin.shape!=baseline.shape or origin.dtype!=baseline.dtype or
            not local_updates or len(local_updates)!=len(weights) or
            any(x.shape!=origin.shape for x in local_updates) or
            history.shape!=(origin.shape[1],origin.shape[1])):
        raise ValueError('invalid shape')
    if (any(not math.isfinite(w) or w<0 for w in weights) or abs(sum(weights)-1)>1e-12 or
            not math.isfinite(tolerance) or tolerance<0):raise ValueError('invalid weights or tolerance')
    if any(not torch.isfinite(x).all() for x in (origin,baseline,history,*local_updates)):
        raise ValueError('nonfinite input')
    def fallback(reason,**info):return baseline.clone(),{'accepted':False,'reason':reason,**info}
    old=baseline.double()-origin.double();old_norm=float(torch.linalg.vector_norm(old))
    if old_norm<=1e-12:return fallback('zero_global_update',ssb_calls=0)
    expected=sum(w*x.double() for w,x in zip(weights,local_updates))
    mean_error=float(torch.linalg.vector_norm(expected-old))
    if mean_error>1e-6*max(float(torch.linalg.vector_norm(expected)),1.):
        raise ValueError('client deltas do not match aggregated update')
    h=_psd(history);scale=float(torch.trace(h)/h.shape[0])
    if scale<=1e-30:return fallback('empty_history',ssb_calls=0)
    h=h/scale
    if use_history:
        eigen,vectors=torch.linalg.eigh(torch.eye(h.shape[0],dtype=h.dtype,device=h.device)+h)
        whiten=(vectors*eigen.rsqrt())@vectors.T
    else:whiten=torch.eye(h.shape[0],dtype=h.dtype,device=h.device)
    consensus=sum(w*mulora_ssb((x.double()@whiten).float()).double() for w,x in zip(weights,local_updates))
    if not torch.isfinite(consensus).all():raise ArithmeticError('nonfinite consensus')
    if float(torch.linalg.vector_norm(consensus))<=1e-7:
        return fallback('zero_consensus',ssb_calls=len(local_updates),mean_error=mean_error)
    polar=mulora_ssb(consensus.float()).double();pn=float(torch.linalg.vector_norm(polar))
    if not math.isfinite(pn) or pn<=1e-12:return fallback('zero_polar',ssb_calls=len(local_updates)+1)
    ideal=polar*(old_norm/pn)
    candidate=(origin.double()+ideal).to(origin.dtype)
    if torch.equal(candidate,baseline):return fallback('no_physical_change',ssb_calls=len(local_updates)+1)
    actual=candidate.double()-origin.double();actual_norm=float(torch.linalg.vector_norm(actual))
    hist_old=float(((old@h)*old).sum());hist_ideal=float(((ideal@h)*ideal).sum())
    hist_new=float(((actual@h)*actual).sum())
    rounding=float(torch.linalg.vector_norm(actual-ideal))
    h_operator=float(torch.linalg.matrix_norm(h,ord=2))
    roundoff_bound=h_operator*(2*float(torch.linalg.vector_norm(ideal))*rounding+rounding**2)
    agreement=float((actual*old).sum())
    info={'mean_error':mean_error,'ssb_calls':len(local_updates)+1,'norm_before':old_norm,'norm_after':actual_norm,
          'history_before':hist_old,'history_ideal':hist_ideal,'history_after':hist_new,
          'physical_roundoff_norm':rounding,'roundoff_history_bound':roundoff_bound,'alignment_dot':agreement}
    budget=tolerance*max(hist_old,1.)
    if (not torch.isfinite(candidate).all() or abs(actual_norm/old_norm-1)>1e-6 or agreement<=0 or
            hist_ideal>hist_old+budget or hist_new>hist_old+budget+roundoff_bound+1e-15):
        return fallback('deployed_budget',**info)
    return candidate,{'accepted':True,'reason':'accepted',**info}


def online_whitener(history,use_history=True):
    """Return the candidate's common-basis transform; no historical data access."""
    h=_psd(history);scale=float(torch.trace(h)/h.shape[0])
    if scale<=1e-30:return None
    h=h/scale
    if not use_history:return torch.eye(h.shape[0],dtype=h.dtype,device=h.device)
    eigen,vectors=torch.linalg.eigh(torch.eye(h.shape[0],dtype=h.dtype,device=h.device)+h)
    return (vectors*eigen.rsqrt())@vectors.T


def finish_online(origin,baseline,mean_update,consensus,history,*,local_ssb_calls,tolerance=1e-10):
    """Finalize and audit online aggregate; values are aggregate only, not clients."""
    if (origin.ndim!=2 or origin.shape!=baseline.shape or origin.shape!=mean_update.shape or
            origin.shape!=consensus.shape or origin.dtype!=baseline.dtype or
            history.shape!=(origin.shape[1],origin.shape[1])):raise ValueError('invalid shapes')
    if any(not torch.isfinite(x).all() for x in (origin,baseline,mean_update,consensus,history)):
        raise ValueError('nonfinite aggregate')
    if local_ssb_calls<0 or not math.isfinite(tolerance) or tolerance<0:raise ValueError('invalid cost or budget')
    def fallback(reason,**info):return baseline.clone(),{'accepted':False,'reason':reason,**info}
    old=baseline.double()-origin.double();old_norm=float(torch.linalg.vector_norm(old))
    if old_norm<=1e-12:return fallback('zero_global_update',ssb_calls=local_ssb_calls)
    err=float(torch.linalg.vector_norm(mean_update.double()-old))
    if err>1e-6*max(float(torch.linalg.vector_norm(mean_update)),1.):
        raise ValueError('online client deltas do not match actual aggregate')
    h=_psd(history);scale=float(torch.trace(h)/h.shape[0])
    if scale<=1e-30:return fallback('empty_history',ssb_calls=local_ssb_calls)
    h=h/scale
    if float(torch.linalg.vector_norm(consensus))<=1e-7:
        return fallback('zero_consensus',ssb_calls=local_ssb_calls,mean_error=err)
    polar=mulora_ssb(consensus.float()).double();pn=float(torch.linalg.vector_norm(polar))
    if not math.isfinite(pn) or pn<=1e-12:return fallback('zero_polar',ssb_calls=local_ssb_calls+1)
    ideal=polar*(old_norm/pn);candidate=(origin.double()+ideal).to(origin.dtype)
    if torch.equal(candidate,baseline):return fallback('no_physical_change',ssb_calls=local_ssb_calls+1)
    actual=candidate.double()-origin.double();actual_norm=float(torch.linalg.vector_norm(actual))
    hist_old=float(((old@h)*old).sum());hist_ideal=float(((ideal@h)*ideal).sum());hist_new=float(((actual@h)*actual).sum())
    rounding=float(torch.linalg.vector_norm(actual-ideal));h_operator=float(torch.linalg.matrix_norm(h,ord=2))
    bound=h_operator*(2*float(torch.linalg.vector_norm(ideal))*rounding+rounding**2)
    alignment=float((actual*old).sum());budget=tolerance*max(hist_old,1.)
    info={'mean_error':err,'ssb_calls':local_ssb_calls+1,'norm_before':old_norm,'norm_after':actual_norm,
          'history_before':hist_old,'history_ideal':hist_ideal,'history_after':hist_new,
          'physical_roundoff_norm':rounding,'roundoff_history_bound':bound,'alignment_dot':alignment}
    if (not torch.isfinite(candidate).all() or abs(actual_norm/old_norm-1)>1e-6 or alignment<=0 or
            hist_ideal>hist_old+budget or hist_new>hist_old+budget+bound+1e-15):
        return fallback('deployed_budget',**info)
    return candidate,{'accepted':True,'reason':'accepted',**info}
