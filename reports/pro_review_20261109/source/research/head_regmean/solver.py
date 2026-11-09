"""Head-only RegMean and its unit-row cosine adaptation (no label fitting).

For local input raw Grams G_k and effective rows U_k, minimize
sum_k tr((U-U_k) G_k (U-U_k)^T), optionally ||U[c]||=1.
Off-diagonal shrinkage is applied before forming this objective.
"""
import torch
from torch.nn import functional as F


def shrink_gram(g, alpha=.5):
    if not 0 <= alpha <= 1:
        raise ValueError('alpha outside [0,1]')
    g = (g.double() + g.double().T) / 2
    return alpha*g + (1-alpha)*torch.diag(g.diag())


def objective(u, h, q):
    # Additive client-only constant omitted consistently.
    return ((u @ h)*u).sum(1) - 2*(u*q).sum(1)


def sphere_solve(h, q):
    """Global solution of each equality-constrained quadratic via secular roots.

KKT: (H+lambda I)u=q, ||u||=1, H+lambda I PSD.
The minimum-eigenspace hard case is solved explicitly, including H=q=0.
"""
    h=(h.double()+h.double().T)/2; q=q.double()
    eigen, v=torch.linalg.eigh(h)
    scale=max(float(eigen.abs().max()), float(q.abs().max()), 1e-30)
    tol=1e-12*scale
    if float(eigen.min()) < -tol:
        raise ValueError('Gram is not positive semidefinite')
    c=q@v; delta=eigen-eigen[0]; minimum=delta<=tol
    answer=[]; multipliers=[]; cases=[]
    for row in c:
        base=torch.zeros_like(row)
        base[~minimum]=row[~minimum]/delta[~minimum]
        hard=bool(row[minimum].norm()<=tol and base.square().sum()<=1)
        if hard:
            j=int(torch.where(minimum)[0][0])
            base[j]=(1-base.square().sum()).clamp_min(0).sqrt()
            x=base; lam=-eigen[0]; cases.append('hard')
        else:
            # Work in t=lambda+lambda_min >=0 to avoid root cancellation.
            lo=0.; hi=max(float(row.norm()), scale)
            def norm2(t):
                denom=delta+t
                if bool(((denom==0)&(row!=0)).any()):return float('inf')
                return float((torch.where(denom>0,row/denom,torch.zeros_like(row))**2).sum())
            while norm2(hi)>1:hi*=2
            for _ in range(160):
                mid=(lo+hi)/2
                if norm2(mid)>1:lo=mid
                else:hi=mid
            t=(lo+hi)/2
            x=row/(delta+t); lam=t-eigen[0]; cases.append('secular')
        answer.append(x@v.T); multipliers.append(lam)
    u=torch.stack(answer); lam=torch.as_tensor(multipliers,dtype=h.dtype,device=h.device)
    lower=-(q*u).sum(1)-lam
    certificate=certify_sphere(u,h,q,lam,lower)
    if not certificate['passed']:raise ArithmeticError(('FP64 sphere solve failed',certificate))
    return u,lam,lower,{'cases':cases,'fp64':certificate}


def certify_sphere(u,h,q,lam,lower):
    h=h.double();q=q.double();u=u.double()
    norm2=u.square().sum(1)
    stationarity=u@h+lam[:,None]*u-q
    scale=(h.norm()*u.norm(dim=1)+q.norm(dim=1)+lam.abs()).clamp_min(1e-30)
    rel=stationarity.norm(dim=1)/scale
    gap=objective(u,h,q)-lower
    gap_scale=((u@h)*u).sum(1).abs()+2*(u*q).sum(1).abs()+lower.abs()
    gap_tol=1e-10+1e-6*gap_scale
    psd=torch.linalg.eigvalsh(h)[0]+lam
    psd_tol=1e-10+1e-10*h.norm()
    return {'norm_squared_error_max':float((norm2-1).abs().max()),
            'stationarity_relative_max':float(rel.max()),
            'dual_gap_abs_max':float(gap.abs().max()),
            'dual_gap_per_row':gap.tolist(),'dual_gap_tolerance_per_row':gap_tol.tolist(),
            'shifted_min_eigenvalue_min':float(psd.min()),
            'passed':bool(((norm2-1).abs()<=2e-6).all() and (rel<=2e-6).all()
                          and (gap.abs()<=gap_tol).all() and (psd>=-psd_tol).all())}


def aggregate_heads(heads, raw_grams, sizes, cosine, alpha=.5):
    if len(heads)!=len(raw_grams) or len(heads)!=len(sizes):raise ValueError('length mismatch')
    if min(sizes)<=0:raise ValueError('empty client')
    weights=[x.double() for x in heads]
    if cosine:
        if any(bool((w.norm(dim=1)==0).any()) for w in weights):raise ValueError('zero cosine row')
        # Match the effective FP32 rows used by actual client forward passes.
        effective=[F.normalize(w.float(),dim=1).double() for w in weights]
    else:effective=weights
    gs=[shrink_gram(g,alpha)/sum(sizes) for g in raw_grams]
    # Raw sums already carry sample counts: do not weight them again by N_k.
    h=sum(gs);q=sum(w@g for w,g in zip(effective,gs))
    raw_average=torch.zeros_like(heads[0],dtype=torch.float32)
    for w,n in zip(heads,sizes):raw_average.add_(w.float(),alpha=n/sum(sizes))
    if cosine:
        u,lam,lower,info=sphere_solve(h,q)
        norms=raw_average.double().norm(dim=1).clamp_min(1e-12)
        deployed=(u*norms[:,None]).float()
        actual=F.normalize(deployed,dim=1).double()
        info['deployed_fp32']=certify_sphere(actual,h,q,lam,lower)
        baseline=F.normalize(raw_average,dim=1).double()
        if not info['deployed_fp32']['passed']:raise ArithmeticError(('FP32 deployment failed',info))
    else:
        u=torch.linalg.solve(h,q.T).T
        deployed=u.float();actual=deployed.double();baseline=raw_average.double()
        residual=actual@h-q
        rel=float(residual.norm()/(h.norm()*actual.norm()+q.norm()).clamp_min(1e-30))
        gap=((actual-u)@h*(actual-u)).sum()
        info={'deployed_fp32':{'stationarity_relative':rel,'objective_gap':float(gap),'passed':rel<2e-6},
              'condition_number':float(torch.linalg.cond(h))}
        if not info['deployed_fp32']['passed']:raise ArithmeticError(('Linear solve failed',info))
    info.update(alpha=alpha,raw_moment_samples=sum(sizes),gram_min_eigenvalue=float(torch.linalg.eigvalsh(h)[0]),
                baseline_objective=float(objective(baseline,h,q).sum()),
                regmean_objective=float(objective(actual,h,q).sum()),
                objective_note='client-only constant omitted; cosine divide true logit squared error by scale^2; alpha shrinkage changes the raw empirical objective')
    if info['regmean_objective']>info['baseline_objective']+1e-6*(1+abs(info['baseline_objective'])):
        raise ArithmeticError('RegMean objective worse than feasible FedAvg candidate')
    return deployed,raw_average,info
