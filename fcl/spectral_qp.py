"""Spectral-ball convex QP. Every certificate is evaluated on the returned step."""
import time
from dataclasses import dataclass
import torch


@dataclass
class SolverConfig:
    max_iter:int=200
    gap_rtol:float=1e-4
    gap_atol:float=1e-10
    warm_start:bool=True


def checked_psd(h):
    h=(h+h.T)/2
    values,vectors=torch.linalg.eigh(h)
    scale=max(float(values.abs().max()),1e-30)
    tol=64*torch.finfo(h.dtype).eps*h.shape[0]*scale
    if float(values.min()) < -tol:raise ValueError('Historical Hessian is materially non-PSD')
    correction=max(0.,-float(values.min()))
    if correction:h=(vectors*values.clamp_min(0))@vectors.T
    return h,values.clamp_min(0),correction


def compact_polar(g):
    u,s,v=torch.linalg.svd(g,full_matrices=False)
    tol=max(g.shape)*torch.finfo(g.dtype).eps*(float(s.max()) if s.numel() else 0)
    keep=s>tol
    return u[:,keep]@v[keep],tol


def spectral_project(x,radius):
    u,s,v=torch.linalg.svd(x,full_matrices=False)
    return (u*s.clamp_max(radius))@v


def objective(g,h,d):return (g*d).sum()+.5*((d@h)*d).sum()


def certificate(g,h,d,eta,config):
    k=g+d@h
    singular=torch.linalg.svdvals(k)
    scale=max(float(eta*torch.linalg.svdvals(g).sum()),1e-30)
    gap=float((k*d).sum()+eta*singular.sum())
    norm=float(torch.linalg.matrix_norm(d,ord=2)) if d.numel() else 0.
    phi=float(objective(g,h,d))
    numeric=64*torch.finfo(g.dtype).eps*scale
    return {'objective':phi,'gap':gap,'relative_gap':gap/scale,'relative_scale':scale,
            'operator_norm':norm,'feasibility_residual':max(0.,norm-eta),
            'converged':gap<=config.gap_atol+config.gap_rtol*scale and gap>=-numeric and norm<=eta+64*torch.finfo(g.dtype).eps*max(eta,1e-30),
            'zero_step':not bool(torch.count_nonzero(d)),'numeric_precision':str(g.dtype)}


@torch.no_grad()
def solve_spectral_qp(g,h,eta,previous_delta=None,config=None):
    config=config or SolverConfig();started=time.perf_counter()
    if g.dtype not in [torch.float32,torch.float64]:raise TypeError('QP requires FP32/FP64')
    if eta<0 or not torch.isfinite(g).all() or not torch.isfinite(h).all():raise ValueError('Invalid QP input')
    h,ev,correction=checked_psd(h)
    warm_used=False;warm_obj=None;rank_tol=0.;iterations=0
    if eta==0 or not torch.count_nonzero(g):d=torch.zeros_like(g)
    elif not torch.count_nonzero(h):
        polar,rank_tol=compact_polar(g);d=-eta*polar
    else:
        q,t=torch.linalg.qr(g,mode='reduced')
        x=torch.zeros_like(t);best=x.clone();best_phi=0.
        if config.warm_start and previous_delta is not None:
            warm=spectral_project(q.T@previous_delta.to(g),eta)
            warm_obj=float(objective(t,h,warm))
            if warm_obj<=0:x=warm;best=x.clone();best_phi=warm_obj;warm_used=True
        polar,rank_tol=compact_polar(t)
        candidate=-eta*polar
        if float(objective(t,h,candidate))<best_phi:best=candidate;best_phi=float(objective(t,h,best))
        # The unconstrained optimum is an exact candidate only if H is positive definite.
        if float(ev.min())>float(ev.max())*torch.finfo(h.dtype).eps*32:
            free=-torch.linalg.solve(h,t.T).T
            candidate=spectral_project(free,eta)
            phi=float(objective(t,h,candidate))
            if phi<best_phi:best=candidate;best_phi=phi
        x=best.clone();y=x.clone();accel=1.;lip=float(ev.max())
        threshold=config.gap_atol+config.gap_rtol*max(float(eta*torch.linalg.svdvals(t).sum()),1e-30)
        for iterations in range(config.max_iter+1):
            k=t+best@h
            gap=float((k*best).sum()+eta*torch.linalg.svdvals(k).sum())
            if gap<=threshold:break
            if iterations==config.max_iter:break
            nxt=spectral_project(y-(t+y@h)/lip,eta)
            phi=float(objective(t,h,nxt))
            if phi>float(objective(t,h,x)):
                y=x;accel=1.;nxt=spectral_project(y-(t+y@h)/lip,eta);phi=float(objective(t,h,nxt))
            if phi<best_phi:best=nxt.clone();best_phi=phi
            new_accel=(1+(1+4*accel*accel)**.5)/2
            y=nxt+(accel-1)/new_accel*(nxt-x);x=nxt;accel=new_accel
        d=q@best
    diagnostic=certificate(g,h,d,eta,config)
    failure=diagnostic['objective']>config.gap_atol+64*torch.finfo(g.dtype).eps*diagnostic['relative_scale']
    if failure:d=torch.zeros_like(d);diagnostic=certificate(g,h,d,eta,config)
    diagnostic.update(iterations=iterations,warm_start_used=warm_used,warm_start_objective=warm_obj,
                      numerical_failure=failure,rank_tolerance=rank_tol,psd_correction=correction,elapsed_seconds=time.perf_counter()-started)
    return d,diagnostic


@torch.no_grad()
def solve_frobenius_qp(g,h,rho):
    if rho<0:raise ValueError('Negative Frobenius radius')
    h,values,_=checked_psd(h)
    values,vectors=torch.linalg.eigh(h);values=values.clamp_min(0)
    gv=g@vectors
    if rho==0 or not torch.count_nonzero(g):return torch.zeros_like(g),{'mu':0.,'iterations':0}
    tol=torch.finfo(g.dtype).eps*max(float(values.max()),1e-30)*32
    positive=values>tol
    free=torch.zeros_like(gv);free[:,positive]=-gv[:,positive]/values[positive]
    if not torch.count_nonzero(gv[:,~positive]) and float(free.norm())<=rho:return free@vectors.T,{'mu':0.,'iterations':0}
    low=0.;high=max(float(g.norm())/rho,float(values.max()),1e-30)
    for _ in range(100):
        if float((gv/(values+high)).norm())<=rho:break
        high*=2
    for it in range(100):
        mid=(low+high)/2
        if float((gv/(values+mid)).norm())>rho:low=mid
        else:high=mid
        if high-low<max(high,1e-30)*1e-12:break
    return (-gv/(values+high))@vectors.T,{'mu':high,'iterations':it+1}
