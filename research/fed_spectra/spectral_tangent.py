"""Loss-directed singular-value rotation with a fixed geometric step ceiling."""
import math
import torch


def tangent_candidate(origin,baseline,reference,gradient,epsilon=1e-12,degeneracy_tolerance=1e-10):
    """Keep singular vectors, update norm and support; never enlarge the reference angle."""
    if not origin.shape==baseline.shape==reference.shape==gradient.shape or baseline.ndim!=2:
        raise ValueError('Mismatched matrix inputs')
    if baseline.shape[0]<baseline.shape[1] or epsilon<=0 or degeneracy_tolerance<0:raise ValueError('Invalid settings')
    o,b,ref,g=[v.detach().double() for v in (origin,baseline,reference,gradient)]
    if not all(torch.isfinite(v).all() for v in (o,b,ref,g)):raise ArithmeticError('Nonfinite inputs')
    d=b-o;v=ref-o;n=float(d.norm())
    record={'identity_reason':None,'input_norm':n}
    def identity(reason):
        record.update(identity_reason=reason,reference_angle=0.,used_angle=0.,norm_relative_error=0.,
                      actual_FP32_norm_relative_error=0.,predicted_delta=0.,actual_FP32_relative_correction=0.)
        return baseline.clone(),record
    if n==0:return identity('zero_baseline')
    parallel=float((d*v).sum())/n
    theta=math.atan2(float((v-(parallel/n)*d).norm()),parallel)
    if theta<=epsilon:return identity('zero_reference_angle')
    u,s,vh=torch.linalg.svd(d,full_matrices=False)
    active=s>epsilon*s[0]
    modes=(u.T@g@vh.T).diag().clone()
    # Equal-singular-value blocks can rotate arbitrarily; use their mean trace.
    groups=[];start=0
    while start<len(s):
        end=start+1
        while end<len(s) and float(s[start]-s[end])<=degeneracy_tolerance*float(s[0]):end+=1
        if end-start>1:modes[start:end]=modes[start:end].mean();groups.append([start,end])
        start=end
    modes[~active]=0.;ss=s.clone();ss[~active]=0.
    tangent=modes-float((modes*ss).sum()/ss.square().sum())*ss
    norm_tangent=float(tangent.norm())
    if norm_tangent<=epsilon*max(float(modes.norm()),1e-30):return identity('zero_spectral_tangent')
    h=-tangent/norm_tangent
    bounds=[math.atan2(float(si),-n*float(hi)) for si,hi,keep in zip(ss,h,active) if keep and hi<0]
    cap=min(bounds) if bounds else math.pi
    angle=min(theta,cap)
    new_s=ss*math.cos(angle)+n*h*math.sin(angle)
    # A boundary coefficient may round slightly negative at the analytic cap.
    if float(new_s.min()) < -1e-12*n:raise ArithmeticError('Negative singular coefficient')
    new_s.clamp_min_(0.)
    new=(u*new_s)@vh
    mathematical_error=abs(float(new.norm())-n)/n
    if mathematical_error>1e-10:raise ArithmeticError('Tangent norm mismatch')
    out=(b+new-d).float();actual=out.double()-o
    deployed_error=abs(float(actual.norm())-n)/n
    if deployed_error>5e-4 or not torch.isfinite(out).all():raise ArithmeticError('FP32 norm mismatch')
    record.update(reference_angle=theta,used_angle=angle,positivity_angle_cap=cap,angle_capped=angle<theta-1e-12,
        singular_values_before=s.tolist(),singular_coefficients_after=new_s.tolist(),
        singular_values_after=torch.linalg.svdvals(actual).tolist(),retained_rank=int(active.sum()),
        repeated_blocks=groups,mode_gradient=modes.tolist(),tangent_gradient_norm=norm_tangent,
        norm_relative_error=mathematical_error,actual_FP32_norm_relative_error=deployed_error,
        actual_FP32_relative_correction=float((out.double()-b).norm())/n,
        tangent_first_order_slope=-n*norm_tangent,
        predicted_delta=float((g*(out.double()-b)).sum()))
    return out,record
