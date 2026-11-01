"""Unregistered constrained spectral proposal; synthetic development only."""
import torch


def constrained_candidate(pool, baseline, history, moment, *, iterations=40, tolerance=1e-5):
    """Search top eigenspaces of M-lambda*H subject to the original trace budget.

    A feasible result is a proposal, not a proof of global optimality or old-task safety.
    """
    original = baseline.detach().clone()
    if pool.ndim != 2 or baseline.ndim != 2 or moment.ndim != 2:
        raise ValueError('invalid tensor rank')
    d, m = pool.shape; rank = baseline.shape[0]
    if (baseline.shape[1] != d or not 0 < rank < m <= d or history.shape != (d,d) or
        moment.shape != (m,m) or iterations < 1 or
        any(not torch.isfinite(x).all() for x in (pool, baseline, history, moment))):
        raise ValueError('invalid inputs')
    p, a, h, mom = [x.double() for x in (pool, baseline, history, moment)]
    eye = torch.eye(m, dtype=p.dtype, device=p.device)
    if (float(torch.linalg.matrix_norm(p.T@p-eye)) > tolerance or
        float(torch.linalg.matrix_norm(a@a.T-torch.eye(rank,device=p.device,dtype=p.dtype))) > tolerance or
        float(torch.linalg.matrix_norm(p@(p.T@a.T)-a.T)) > tolerance):
        return original, {'accepted':False, 'reason':'invalid_pool_or_basis'}
    for x in (h, mom):
        scale = max(float(torch.linalg.matrix_norm(x)), 1e-12)
        if float(torch.linalg.matrix_norm(x-x.T)) > tolerance*scale:
            return original, {'accepted':False, 'reason':'nonsymmetric_input'}
    h=(h+h.T)*.5; mom=(mom+mom.T)*.5; hp=p.T@h@p
    for x in (hp, mom):
        eig=torch.linalg.eigvalsh(x); scale=max(float(eig.abs().max()),1e-12)
        if float(eig.min()) < -tolerance*scale:
            return original, {'accepted':False, 'reason':'non_psd_input'}
    z0=p.T@a.T
    limit=float(torch.trace(a@h@a.T)); base=float(torch.trace(z0.T@mom@z0))
    history_slack=tolerance*max(abs(limit),1e-12)
    best=None
    def inspect(lam):
        nonlocal best
        values, vec=torch.linalg.eigh(mom-lam*hp)
        scale=max(float(values.abs().max()),1e-12)
        # The formal AdamW update depends on basis coordinates, not only the projector.
        if float(values[-rank]-values[-rank-1]) <= 1e-7*scale:
            return False
        if rank>1 and float(values[-rank:].diff().abs().min()) <= 1e-7*scale:
            return False
        z=vec[:,-rank:].flip(1)
        piv=z.abs().argmax(0); sign=torch.sign(z[piv,torch.arange(rank,device=z.device)])
        actual=((p@(z*sign)).T).to(baseline.dtype).double()
        r=float(torch.trace(actual@h@actual.T)); u=float(torch.trace((p.T@actual.T).T@mom@(p.T@actual.T)))
        orth=float(torch.linalg.matrix_norm(actual@actual.T-torch.eye(rank,device=p.device,dtype=p.dtype)))
        feasible=r <= limit+history_slack and orth <= tolerance
        if feasible and u > base + 1e-7*max(abs(base),1e-12) and (best is None or u > best[0]):
            best=(u,r,actual.to(baseline.dtype),lam)
        return feasible
    low=0.; high=1.
    if not inspect(0.):
        for _ in range(32):
            if inspect(high): break
            high*=2
        else:
            return original, {'accepted':False,'reason':'no_feasible_spectral_candidate'}
        for _ in range(iterations):
            mid=(low+high)*.5
            if inspect(mid): high=mid
            else: low=mid
    if best is None:
        return original, {'accepted':False,'reason':'no_strict_gain'}
    utility,risk,candidate,lam=best
    return candidate, {'accepted':True,'baseline_utility':base,'utility':utility,
                       'history_budget':limit,'history_response':risk,'multiplier':lam}


def endpoint_moment(endpoints, counts, mode):
    """Return mean-update or sample-weighted individual-update energy."""
    if endpoints.ndim != 4:
        raise ValueError('expected clients x branches x outputs x coordinates')
    count=torch.as_tensor(counts,dtype=torch.float64,device=endpoints.device)
    if (count.shape!=(endpoints.shape[0],) or not count.numel() or not torch.isfinite(count).all() or
            (count<0).any() or count.sum()<=0 or not torch.isfinite(endpoints).all()):
        raise ValueError('invalid endpoint weights')
    weights=count/count.sum(); e=endpoints.double()
    if mode=='mean':
        mean=torch.einsum('k,kbod->bod',weights,e)
        return torch.einsum('bod,boe->de',mean,mean)
    if mode=='individual':
        return torch.einsum('k,kbod,kboe->de',weights,e,e)
    raise ValueError('unknown energy mode')


def propose_from_trials(model, pools, histories, trials, mode):
    """Propose per-layer bases using private trial endpoints and fixed energy mode."""
    names=list(model.adapters)
    if not names or set(names)!=set(pools) or set(names)!=set(histories) or not trials:
        raise ValueError('invalid trial or layer set')
    expected=[p for name in names for p in (model.adapters[name].Bq,model.adapters[name].Bv)]
    actual=model.b_parameters()
    if len(actual)!=len(expected) or any(a is not b for a,b in zip(actual,expected)):
        raise ValueError('unexpected adapter order')
    if any(len(t['B'])!=len(expected) or t['samples']<=0 for t in trials):
        raise ValueError('incomplete client endpoints')
    counts=[t['samples'] for t in trials]
    bases,records={},{}
    for j,name in enumerate(names):
        endpoints=torch.stack([torch.stack((t['B'][2*j],t['B'][2*j+1])) for t in trials])
        moment=endpoint_moment(endpoints,counts,mode)
        bases[name],records[name]=constrained_candidate(pools[name],model.adapters[name].A,histories[name],moment)
    return bases,records
