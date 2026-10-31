"""Unregistered actual-update server geometry; synthetic development only."""
import itertools
import math
import torch


def _project_cone(q, normals):
    """Project a vector onto up to three homogeneous halfspaces by active sets."""
    if q.ndim != 1 or len(normals) != 3 or any(g.shape != q.shape for g in normals):
        raise ValueError('invalid cone shapes')
    q = q.double(); gs = [g.double() for g in normals]
    if not torch.isfinite(q).all() or any(not torch.isfinite(g).all() for g in gs):
        raise ValueError('nonfinite cone input')
    tol = 1e-12 * max(float(torch.linalg.vector_norm(q)), 1.)
    ns = [float(torch.linalg.vector_norm(g)) for g in gs]
    def feasible(x):
        return all(float(torch.dot(g, x)) <= tol * max(n, 1.) for g, n in zip(gs, ns))
    candidates = [(q.clone(), ())] if feasible(q) else []
    for count in range(1, 4):
        for active in itertools.combinations(range(3), count):
            G = torch.stack([gs[i] for i in active])
            gram = G @ G.T
            if int(torch.linalg.matrix_rank(gram)) != count:continue
            multipliers = torch.linalg.solve(gram, G @ q)
            if bool((multipliers < -1e-12).any()):continue
            v = q - G.T @ multipliers
            if feasible(v):candidates.append((v, active))
    if not candidates:return torch.zeros_like(q), {'solved':False, 'active':[]}
    best, active = min(candidates, key=lambda x:float(torch.linalg.vector_norm(x[0]-q)))
    return best, {'solved':True, 'active':list(active), 'normal_dots':[float(g @ best) for g in gs]}


def actual_step(updates, proxy_directions, all_gradients, current_gradients, histories,
                radius=.1, history_tolerance=1e-10):
    """Constrain the actual correction, never a tangent later remapped to a sphere.

    Historical forms are response proxies. Inputs and budget are research-draft
    choices, not preregistered or validated on private data.
    """
    arrays = (updates,proxy_directions,all_gradients,current_gradients)
    if not updates or any(len(v)!=len(updates) for v in (*arrays[1:],histories)):
        raise ValueError('matrix count mismatch')
    if not math.isfinite(radius) or radius <= 0 or not math.isfinite(history_tolerance) or history_tolerance < 0:
        raise ValueError('invalid budget')
    for i,u in enumerate(updates):
        if u.ndim!=2 or any(v[i].shape!=u.shape for v in arrays[1:]) or histories[i].shape!=(u.shape[1],u.shape[1]):
            raise ValueError('matrix shape mismatch')
    if any(not torch.isfinite(x).all() for a in (*arrays,histories) for x in a):
        raise ValueError('nonfinite input')
    flat=lambda values:torch.cat([x.detach().double().reshape(-1) for x in values])
    u,q,ga,gc=(flat(v) for v in arrays)
    def fallback(reason,**info):return [x.clone() for x in updates],{'accepted':False,'reason':reason,**info}
    if float(torch.linalg.vector_norm(u))==0:return fallback('zero_update')
    hs=[];history_grad=[];base=0.
    for old,h in zip(updates,histories):
        h=(h.double()+h.double().T)/2
        if float(torch.linalg.eigvalsh(h).min()) < -1e-8:raise ValueError('indefinite history')
        scale=float(torch.trace(h)/h.shape[0])
        if scale<=0:return fallback('empty_history')
        h=h/scale;old=old.double();hs.append(h);history_grad.append(2*old@h)
        base+=float(((old@h)*old).sum())
    gh=flat(history_grad)
    d,cone=_project_cone(q,[ga,gc,gh]);dn=float(torch.linalg.vector_norm(d))
    if not cone['solved'] or dn<=1e-12:return fallback('empty_or_unsolved_cone',cone=cone)
    d=d/dn
    pieces=[];start=0
    for old in updates:
        n=old.numel();pieces.append(d[start:start+n].reshape_as(old));start+=n
    linear=sum(float(2*((old.double()@h)*v).sum()) for old,h,v in zip(updates,hs,pieces))
    quadratic=sum(float(((v@h)*v).sum()) for v,h in zip(pieces,hs))
    if linear >= -1e-14:return fallback('no_history_descent',cone=cone,history_linear=linear)
    cap=radius*float(torch.linalg.vector_norm(u))
    if quadratic>0:cap=min(cap,.9*(-linear/quadratic))
    if cap<=1e-12:return fallback('zero_step',cone=cone)
    result=[(old.double()+cap*v).to(old.dtype) for old,v in zip(updates,pieces)]
    delta=flat(result)-u
    all_dot=float(ga@delta);cur_dot=float(gc@delta)
    used=float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(u))
    hist=sum(float(((v.double()@h)*v.double()).sum()) for v,h in zip(result,hs))
    receipt={'cone':cone,'all_directional':all_dot,'current_directional':cur_dot,
             'baseline_history':base,'proposal_history':hist,'relative_step_norm':used}
    if (used>radius+1e-6 or all_dot>=-1e-12 or cur_dot>1e-12 or
            hist>base+history_tolerance*max(base,1.)):
        return fallback('deployed_budget',**receipt)
    return result,{'accepted':True,'reason':'accepted',**receipt}


def deployed_parameters(origins, baselines, corrected_updates, all_gradients, current_gradients,
                        histories, radius=.1, tolerance=1e-10):
    """Recheck after FP32 origin+update, falling back to exact baseline parameters."""
    inputs=(origins,baselines,corrected_updates,all_gradients,current_gradients)
    if not origins or any(len(v)!=len(origins) for v in (*inputs[1:],histories)):
        raise ValueError('deployment matrix count mismatch')
    if any(any(v.shape!=origins[i].shape for v in [baselines[i],corrected_updates[i],all_gradients[i],current_gradients[i]])
           for i in range(len(origins))):raise ValueError('deployment matrix shape mismatch')
    if any(not torch.isfinite(x).all() for arr in (*inputs,histories) for x in arr):raise ValueError('nonfinite deployment')
    base_updates=[b.double()-o.double() for o,b in zip(origins,baselines)]
    new=[(o+v.to(o.dtype)).to(o.dtype) for o,v in zip(origins,corrected_updates)]
    changes=[n.double()-b.double() for n,b in zip(new,baselines)]
    real_updates=[n.double()-o.double() for n,o in zip(new,origins)]
    norm=lambda arr:math.sqrt(sum(float((v*v).sum()) for v in arr))
    base_norm=norm(base_updates)
    def fallback(reason,**kwargs):return [b.clone() for b in baselines],{'accepted':False,'reason':reason,**kwargs}
    if base_norm==0:return fallback('zero_base_update')
    if not math.isfinite(radius) or radius<=0 or not math.isfinite(tolerance) or tolerance<0:raise ValueError('invalid budget')
    adot=sum(float((g.double()*d).sum()) for g,d in zip(all_gradients,changes))
    cdot=sum(float((g.double()*d).sum()) for g,d in zip(current_gradients,changes))
    relative=norm(changes)/base_norm;base_hist=next_hist=0.
    for u,v,h in zip(base_updates,real_updates,histories):
        h=(h.double()+h.double().T)/2
        if h.shape!=(u.shape[1],u.shape[1]) or float(torch.linalg.eigvalsh(h).min()) < -1e-8:
            raise ValueError('invalid history')
        scale=float(torch.trace(h)/h.shape[0])
        if scale<=0:return fallback('empty_history')
        h=h/scale;base_hist+=float(((u@h)*u).sum());next_hist+=float(((v@h)*v).sum())
    rec={'all_directional':adot,'current_directional':cdot,'relative_step_norm':relative,
         'baseline_history':base_hist,'proposal_history':next_hist}
    if relative>radius+1e-6 or adot>=-1e-12 or cdot>1e-12 or next_hist>base_hist+tolerance*max(base_hist,1.):
        return fallback('post_addition_budget',**rec)
    return new,{'accepted':True,'reason':'accepted',**rec}
