"""Joint, history-weighted boundary repair; no per-layer projection/clipping."""
import torch


def inner(xs, ys):
    return sum((x.double()*y.double()).sum() for x,y in zip(xs,ys))


def metric(a,h,epsilon=1e-12,weight=1.):
    h=(h.double()+h.double().T)/2
    scale=h.diag().mean().clamp_min(epsilon)
    k=a.double()@h@a.double().T
    q=torch.eye(a.shape[0],device=a.device,dtype=torch.float64)+weight*(k+k.T)/(2*scale)
    if not torch.isfinite(q).all():raise ArithmeticError('Nonfinite historical metric')
    chol=torch.linalg.cholesky(q)
    return q,torch.cholesky_inverse(chol),float(scale)


def norm2(xs,qs):
    return inner(xs,[x.double()@q for x,q in zip(xs,qs)])


def repair(g,h,base,head_effect,qs,inverses,kappa=.25,relative_tolerance=1e-10):
    """FP64 ideal correction. base and head_effect use actual FP32 displacements."""
    zero=[torch.zeros_like(x,dtype=torch.float64) for x in g]
    cross_b=inner(h,base);joint=cross_b+head_effect
    if not torch.isfinite(joint):raise ArithmeticError('Nonfinite joint proposal')
    base_norm=norm2(base,qs).clamp_min(0).sqrt()
    info={'base_cross':float(cross_b),'head_cross':float(head_effect),'joint_cross':float(joint),
          'trigger':bool(joint>0),'base_norm_Q':float(base_norm),'scale':0.,'reason':'no_conflict'}
    if joint<=0:return zero,info
    th=[x.double()@q for x,q in zip(h,inverses)];hth=inner(h,th)
    if hth==0:info['reason']='no_B_cross_gradient';return zero,info
    tg=[x.double()@q for x,q in zip(g,inverses)];gtg=inner(g,tg)
    if gtg==0:
        if any(torch.count_nonzero(x) for x in g):raise ArithmeticError('Nonzero g with zero metric norm')
        tangent=th
    else:
        if not torch.isfinite(gtg) or gtg<0:raise ArithmeticError('Invalid CE metric norm')
        coefficient=inner(g,th)/gtg
        tangent=[a-coefficient*b for a,b in zip(th,tg)]
    denom=inner(h,tangent)
    info['relative_denominator']=float(denom/hth)
    if not torch.isfinite(denom) or denom<=0 or denom<=relative_tolerance*hth:
        info['reason']='degenerate_projection';return zero,info
    delta=[-joint/denom*x for x in tangent]
    full_norm=norm2(delta,qs).clamp_min(0).sqrt()
    scale=min(1.,float(kappa*base_norm/full_norm)) if full_norm>0 else 0.
    delta=[scale*x for x in delta]
    info.update(scale=scale,reason='full' if scale==1 else 'capped',ideal_norm_Q=float(full_norm*scale),
                ideal_ce_residual=float(inner(g,delta)),ideal_remaining_cross=float(joint+inner(h,delta)))
    if any(not torch.isfinite(x).all() for x in delta):raise ArithmeticError('Nonfinite correction')
    return delta,info


def cross_loss(logits,y,task,cpt,mask):
    from torch.nn import functional as F
    centered=(logits.reshape(len(y),task+1,cpt)-logits.reshape(len(y),task+1,cpt).mean(-1,keepdim=True)).reshape_as(logits)
    margin=centered[:,:task*cpt].max(1).values-centered.gather(1,y[:,None]).squeeze(1)
    return F.softplus(margin[mask]).mean()
