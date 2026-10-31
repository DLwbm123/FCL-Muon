"""Unregistered teacher-response Muon trust step; synthetic development only."""
import math
import torch
from fcl.math import mulora_ssb


def propose(origins,baselines,response_gradients,response_loss,*,local_polar=None):
    """One global loss-calibrated correction at actual FedAvg anchor.

    response_loss/gradients are calculated on the same current-task inputs.
    This proxy cannot establish old-class safety; caller must separately audit
    actual teacher response and current loss before accepting the candidate.
    """
    if not origins or len(origins)!=len(baselines) or len(origins)!=len(response_gradients):
        raise ValueError('matrix count mismatch')
    if local_polar is not None and (len(local_polar)!=len(origins) or any(q.shape!=o.shape for q,o in zip(local_polar,origins)) or any(not torch.isfinite(q).all() for q in local_polar)):
        raise ValueError('invalid local polar consensus')
    if not math.isfinite(response_loss) or response_loss < -1e-12:
        raise ValueError('invalid response loss')
    for o,b,g in zip(origins,baselines,response_gradients):
        if o.ndim!=2 or o.shape!=b.shape or o.shape!=g.shape or o.dtype!=b.dtype:
            raise ValueError('matrix shape or dtype mismatch')
        if any(not torch.isfinite(v).all() for v in (o,b,g)):raise ValueError('nonfinite input')
    def fallback(reason,**info):return [v.clone() for v in baselines],{'proposed':False,'reason':reason,**info}
    loss=max(response_loss,0.)
    if loss<=1e-12:return fallback('zero_response',ssb_calls=0)
    updates=[b.double()-o.double() for o,b in zip(origins,baselines)]
    norm=lambda vs:math.sqrt(sum(float((v.double()**2).sum()) for v in vs))
    base_norm=norm(updates);grad_norm=norm(response_gradients)
    if base_norm<=1e-12:return fallback('zero_global_update',ssb_calls=0)
    if grad_norm<=1e-12:return fallback('zero_response_gradient',ssb_calls=0)
    # Project each matrix gradient with the existing approximate Muon operator.
    directions=[mulora_ssb(g.float()).double() for g in response_gradients]
    if any(not torch.isfinite(v).all() for v in directions):raise ArithmeticError('nonfinite SSB direction')
    direction_norm=norm(directions)
    product=sum(float((g.double()*d).sum()) for g,d in zip(response_gradients,directions))
    calls=len(directions)
    if direction_norm<=1e-12 or product<=1e-12:
        return fallback('no_descent_direction',ssb_calls=calls)
    # The same-anchor client factor is a training proxy, not a multistep bias.
    agreement=1.
    if local_polar is not None:
        local_norm=norm(local_polar)
        if local_norm<=1e-12:return fallback('zero_client_consensus',ssb_calls=calls)
        dot=sum(float((d*local.double()).sum()) for d,local in zip(directions,local_polar))
        agreement=max(-1.,min(1.,dot/(direction_norm*local_norm)))
        if agreement<=0:return fallback('conflicting_client_consensus',ssb_calls=calls,consensus_cosine=agreement)
    # Polyak-like response step; ||correction|| is capped by ||U||.
    step=min(loss/product,base_norm/direction_norm)*agreement
    if not math.isfinite(step) or step<=0:return fallback('zero_step',ssb_calls=calls)
    candidates=[(b.double()-step*d).to(b.dtype) for b,d in zip(baselines,directions)]
    if all(torch.equal(v,b) for v,b in zip(candidates,baselines)):
        return fallback('no_physical_change',ssb_calls=calls)
    delta=[v.double()-b.double() for v,b in zip(candidates,baselines)]
    physical_dot=sum(float((g.double()*d).sum()) for g,d in zip(response_gradients,delta))
    ratio=norm(delta)/base_norm
    info={'ssb_calls':calls,'response_loss':loss,'ideal_descent_product':product,
          'step':step,'consensus_cosine':agreement,'physical_response_dot':physical_dot,'correction_to_update_norm':ratio}
    if (any(not torch.isfinite(v).all() for v in candidates) or physical_dot>=-1e-12 or ratio>1+1e-6):
        return fallback('physical_budget',**info)
    return candidates,{'proposed':True,'reason':'proposed',**info}
