"""Spectral-ball convex QP. Every certificate is evaluated on the returned step."""
import time
import math
from dataclasses import dataclass
import torch


@dataclass
class SolverConfig:
    max_iter:int=200
    gap_rtol:float=1e-4
    gap_atol:float=1e-10
    warm_start:bool=True
    reduced_dtype:str='float64'
    inner_gap_rtol:float=1e-6
    inner_gap_atol:float=1e-12


def _svd(x):
    # CUDA's default Jacobi driver can lose several e-6 of orthogonality in
    # FP32. Use the QR-based driver for radius-sensitive projection/polar steps.
    return torch.linalg.svd(x,full_matrices=False,**({'driver':'gesvd'} if x.is_cuda else {}))


def _svdvals(x):
    return torch.linalg.svdvals(x,**({'driver':'gesvd'} if x.is_cuda else {}))


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
    # Audit the stored matrix in FP64: an input-FP32 rank heuristic can discard
    # enough small, nonzero singular mass to violate the requested oracle gap.
    u,s,v=_svd(g.double())
    tol=max(g.shape)*torch.finfo(s.dtype).eps*(float(s.max()) if s.numel() else 0)
    keep=s>tol
    return (u[:,keep]@v[keep]).to(g.dtype),tol


def spectral_project(x,radius):
    u,s,v=_svd(x)
    return (u*s.clamp_max(radius))@v


def objective(g,h,d):return (g*d).sum()+.5*((d@h)*d).sum()


@torch.no_grad()
def certificate(g,h,d,eta,config):
    # Audit the stored problem/step in FP64, rather than trusting the FP32 SVD
    # and cancellation-prone reductions used to construct a boundary iterate.
    # Keep the existing input-precision feasibility allowance and gap tolerances.
    input_dtype=g.dtype
    gd,hd,dd=g.double(),h.double(),d.double()
    k=gd+dd@hd
    scale=max(float(eta*_svdvals(gd).sum()),1e-30)
    gap=float((k*dd).sum()+eta*_svdvals(k).sum())
    norm=float(_svdvals(dd).max()) if dd.numel() else 0.
    phi=float(objective(gd,hd,dd))
    numeric=64*torch.finfo(input_dtype).eps*scale
    gap_limit=config.gap_atol+config.gap_rtol*scale
    radius_allowance=64*torch.finfo(input_dtype).eps*max(eta,1e-30)
    gap_ok=-numeric<=gap<=gap_limit
    feasible=norm<=eta+radius_allowance
    return {'objective':phi,'gap':gap,'relative_gap':gap/scale,'relative_scale':scale,
            'operator_norm':norm,'feasibility_residual':max(0.,norm-eta),
            'gap_limit':gap_limit,'negative_gap_allowance':numeric,'feasibility_allowance':radius_allowance,
            'gap_satisfied':gap_ok,'feasible_within_tolerance':feasible,
            'strictly_feasible':norm<=eta,'gap_is_bound_for_feasible_point':norm<=eta,
            'converged':gap_ok and feasible,
            'zero_step':not bool(torch.count_nonzero(dd)),'numeric_precision':str(input_dtype),
            'certificate_precision':'torch.float64','certificate_version':'fp64-endpoints-v2'}


def actual_update_certificate(g,h,before,after,eta,config):
    # Cast the two stored endpoints BEFORE subtraction. In particular, tiny
    # applied steps and cancellation must not be replaced by the proposal.
    # Reproduce the solver's declared symmetry/tiny-negative-eigenvalue repair
    # from the same original input, so both certificates describe the same QP.
    canonical,_,correction=checked_psd(h)
    result=certificate(g,canonical,after.double()-before.double(),eta,config)
    result['certificate_psd_correction']=correction
    return result


@torch.no_grad()
def solve_spectral_qp(g,h,eta,previous_delta=None,config=None):
    config=config or SolverConfig();started=time.perf_counter()
    if g.dtype not in [torch.float32,torch.float64]:raise TypeError('QP requires FP32/FP64')
    if not math.isfinite(eta) or eta<0 or not torch.isfinite(g).all() or not torch.isfinite(h).all():raise ValueError('Invalid QP input')
    if config.reduced_dtype not in ['float64','input']:raise ValueError('Unknown reduced solve precision')
    if config.max_iter<0 or min(config.gap_rtol,config.gap_atol,config.inner_gap_rtol,config.inner_gap_atol)<0:
        raise ValueError('Invalid solver tolerances/budget')
    h,_,correction=checked_psd(h)
    work_dtype=torch.float64 if config.reduced_dtype=='float64' else g.dtype
    inner=SolverConfig(gap_rtol=min(config.gap_rtol,config.inner_gap_rtol),
                       gap_atol=min(config.gap_atol,config.inner_gap_atol))
    warm_used=False;warm_obj=None;rank_tol=0.;iterations=0
    reduced_info=None;returned='special_case';best_objective=None
    if eta==0 or not torch.count_nonzero(g):d=torch.zeros_like(g)
    elif not torch.count_nonzero(h):
        polar,rank_tol=compact_polar(g);d=-eta*polar
    else:
        q,t=torch.linalg.qr(g,mode='reduced')
        # QR remains in the input precision; the r-dimensional reference solve
        # and objective/selection arithmetic are explicitly FP64 by default.
        qw=q.to(work_dtype);t=t.to(work_dtype);hw=h.to(work_dtype)
        best=None;best_phi=float('inf');certified=None;cert_phi=float('inf')

        def consider(x,label):
            nonlocal best,best_phi,certified,cert_phi
            ri=certificate(t,hw,x,eta,inner)
            di=(qw@x).to(g.dtype)
            pi=certificate(g,h,di,eta,config)
            # The best objective and a certified candidate are different ledgers.
            # Inspect EVERY candidate, even when its objective is slightly higher.
            if ri['feasible_within_tolerance'] and pi['feasible_within_tolerance']:
                item=(x.clone(),di,ri,pi,label)
                if pi['objective']<best_phi:best=item;best_phi=pi['objective']
                nonincrease=pi['objective']<=config.gap_atol+pi['negative_gap_allowance']
                if pi['converged'] and nonincrease and pi['objective']<cert_phi:
                    certified=item;cert_phi=pi['objective']
            return ri['converged'] and pi['converged']

        consider(torch.zeros_like(t),'zero')
        if config.warm_start and previous_delta is not None:
            warm=spectral_project(qw.T@previous_delta.to(device=g.device,dtype=work_dtype),eta)
            warm_obj=float(objective(t.double(),hw.double(),warm.double()))
            if warm_obj<=0:consider(warm,'warm');warm_used=True
        polar,rank_tol=compact_polar(t)
        consider(-eta*polar,'linear_oracle')
        ev=torch.linalg.eigvalsh(hw)
        if float(ev.min())>float(ev.max())*torch.finfo(work_dtype).eps*32:
            free=-torch.linalg.solve(hw,t.T).T
            consider(spectral_project(free,eta),'projected_unconstrained')
        x=best[0].clone();y=x.clone();accel=1.;lip=float(ev.max())
        for iterations in range(config.max_iter+1):
            if certified is not None and certified[2]['converged']:break
            if iterations==config.max_iter:break
            nxt=spectral_project(y-(t+y@hw)/lip,eta)
            phi=float(objective(t.double(),hw.double(),nxt.double()))
            if phi>float(objective(t.double(),hw.double(),x.double())):
                y=x;accel=1.;nxt=spectral_project(y-(t+y@hw)/lip,eta)
            consider(nxt,'projected_gradient')
            new_accel=(1+(1+4*accel*accel)**.5)/2
            y=nxt+(accel-1)/new_accel*(nxt-x);x=nxt;accel=new_accel
        selected=certified if certified is not None else best
        _,d,reduced_info,_,returned=selected;best_objective=best_phi
    diagnostic=certificate(g,h,d,eta,config)
    failure=diagnostic['objective']>config.gap_atol+64*torch.finfo(g.dtype).eps*diagnostic['relative_scale']
    if failure:
        d=torch.zeros_like(d);diagnostic=certificate(g,h,d,eta,config)
        reduced_info=None;returned='numerical_failure_zero_fallback'
    proposed_info=diagnostic.copy()
    diagnostic.update(iterations=iterations,warm_start_used=warm_used,warm_start_objective=warm_obj,
                      numerical_failure=failure,rank_tolerance=rank_tol,psd_correction=correction,
                      svd_driver='gesvd' if g.is_cuda else 'CPU LAPACK',
                      reduced_precision=str(work_dtype),polar_precision='torch.float64',
                      reduced_certificate=reduced_info,proposed_certificate=proposed_info,
                      returned_candidate=returned,best_feasible_objective=best_objective,
                      inner_gap_rtol=inner.gap_rtol,inner_gap_atol=inner.gap_atol,
                      iteration_limit_reached=iterations==config.max_iter,
                      elapsed_seconds=time.perf_counter()-started)
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
