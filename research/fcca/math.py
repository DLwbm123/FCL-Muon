"""Signed cross-client derivatives and a bounded server weight proposal."""
import numpy as np
import torch


def directions(endpoints, baseline, origin, weights):
    """Affine directions anchored to the original FP32 FedAvg, including its rounding.

    sum p_k D_k = baseline - origin.  With sum alpha=1, the proposed endpoint
    is baseline + sum (alpha-p)_k D_k. All returned tensors are CPU FP64.
    """
    p=torch.as_tensor(weights,dtype=torch.float64)
    result=[]
    for i,(b,o) in enumerate(zip(baseline,origin)):
        stack=torch.stack([e[i].detach().cpu().double() for e in endpoints])
        mean=torch.einsum('k,kdr->dr',p,stack)
        result.append(stack-mean+b.detach().cpu().double()-o.detach().cpu().double())
    return result


def candidate(baseline, ds, weights, alpha):
    p=torch.as_tensor(weights,dtype=torch.float64);a=torch.as_tensor(alpha,dtype=torch.float64)
    if torch.equal(p,a):return [b.detach().cpu().clone() for b in baseline]
    return [(b.detach().cpu().double()+torch.einsum('k,kdr->dr',a-p,d)).to(b.dtype)
            for b,d in zip(baseline,ds)]


def directional(gradients, ds):
    """Keep signs: one client's uploaded gradient against EVERY client direction."""
    return sum(torch.einsum('dr,kdr->k',g.detach().cpu().double(),d) for g,d in zip(gradients,ds))


def historical_gram(ds, metrics):
    k=ds[0].shape[0];out=torch.zeros(k,k,dtype=torch.float64)
    for d,q in zip(ds,metrics):
        out+=(d@q).flatten(1)@d.flatten(1).T
    return (out+out.T)/2


def historical_energy(endpoint, origin, metrics):
    return float(sum(((a.detach().cpu().double()-b.detach().cpu().double())@q*
                      (a.detach().cpu().double()-b.detach().cpu().double())).sum()
                     for a,b,q in zip(endpoint,origin,metrics)))


def propose(weights, derivative, gram, settings):
    """Convex proposal, not a certificate of nonlinear or old-class loss.

    Minimize a standardized signed first-order raw CE change plus chi-square
    distance to sample weights, subject to simplex/box/history constraints.
    A failed numerical solve returns the exact original weights.
    """
    from scipy.optimize import minimize
    p=np.asarray(weights,dtype=np.float64);v=np.asarray(derivative,dtype=np.float64)
    h=np.asarray(gram,dtype=np.float64);h=(h+h.T)/2
    if not (np.isfinite(p).all() and np.isfinite(v).all() and np.isfinite(h).all()):
        raise ArithmeticError('Nonfinite aggregation input')
    if len(p)==0 or np.any(p<=0) or abs(p.sum()-1)>1e-10:raise ValueError('Invalid client weights')
    eigen=np.linalg.eigvalsh(h)
    if eigen[0]<-1e-9*max(float(np.linalg.norm(h)),1e-30):raise ArithmeticError('Non-PSD history Gram')
    base=max(0.,float(p@h@p));limit=base*(1+settings['history_relative_tolerance'])+settings['history_absolute_tolerance']
    scale=max(limit,1e-30);hs=h/scale
    centered=v-p@v;std=float(np.sqrt(p@(centered*centered)))
    info={'baseline_history_energy':base,'history_limit':limit,'derivative_scale':std,
          'solver':'scipy SLSQP, FP64, verified constraints','success':False}
    if std<=settings['derivative_floor']:
        return p.copy(),dict(info,reason='no_client_direction_contrast')
    vnorm=centered/std;rho=settings['minimum_weight_ratio'];lower=rho*p;upper=(2-rho)*p
    penalty=settings['trust_penalty']
    def objective(a):return float(vnorm@(a-p)+.5*penalty*np.sum((a-p)**2/p))
    def jac(a):return vnorm+penalty*(a-p)/p
    result=minimize(objective,p.copy(),jac=jac,method='SLSQP',bounds=list(zip(lower,upper)),
                    constraints=[{'type':'eq','fun':lambda a:float(a.sum()-1),'jac':lambda a:np.ones_like(a)},
                                 {'type':'ineq','fun':lambda a:float(limit/scale-a@hs@a),
                                  'jac':lambda a:-2*hs@a}],
                    options={'maxiter':settings['solver_iterations'],'ftol':settings['solver_ftol'],'disp':False})
    a=np.asarray(result.x);tol=settings['weight_tolerance']
    feasible=(np.isfinite(a).all() and abs(a.sum()-1)<=tol and
              np.all(a>=lower-tol) and np.all(a<=upper+tol) and
              float(a@h@a)<=limit+settings['history_absolute_tolerance'])
    predicted=float(v@(a-p)) if np.isfinite(a).all() else None
    info.update(iterations=int(result.nit),message=str(result.message),feasible=bool(feasible),
                predicted_fit_raw_ce_change=predicted,objective=float(result.fun),
                proposed_weights=a.tolist() if np.isfinite(a).all() else None)
    if not result.success or not feasible:return p.copy(),dict(info,reason='solver_rejected')
    if predicted>=-settings['derivative_floor'] or objective(a)>=-settings['solver_ftol']:
        return p.copy(),dict(info,reason='no_predicted_gain')
    return a,dict(info,reason='candidate',success=True)


def finite_acceptance(base_fit, cand_fit, base_check, cand_check, settings):
    """Exactly one candidate; independent selector-check split, never re-search."""
    reasons=[]
    for split,base,cand in [('fit',base_fit,cand_fit),('check',base_check,cand_check)]:
        if base is None or cand is None:
            reasons.append(split+'_empty');continue
        if cand['raw_ce']>base['raw_ce']-settings['minimum_relative_raw_gain']*max(base['raw_ce'],1e-12):
            reasons.append(split+'_raw_gain')
        tolerance=settings['within_ce_absolute_tolerance']+settings['within_ce_relative_tolerance']*base['within_ce']
        if cand['within_ce']>base['within_ce']+tolerance:reasons.append(split+'_within_ce')
    return not reasons,reasons
