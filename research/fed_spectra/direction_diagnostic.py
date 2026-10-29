"""A fixed-angle reflection and disjoint confirmation probes; no strength search."""
import torch


def confirmation_probes(groups,seed,selection):
    result=[]
    for client,(ids,used) in enumerate(zip(groups,selection)):
        pool=sorted(set(ids)-set(used))
        g=torch.Generator().manual_seed(61030+1009*seed+31*client)
        result.append([pool[i] for i in torch.randperm(len(pool),generator=g).tolist()[:32]])
    return result


def reflected_proposal(origin,baseline,positive):
    """Reflect each proposed update about its FedAvg axis, retaining norm/angle."""
    if not len(origin)==len(baseline)==len(positive) or not origin:raise ValueError('Mismatched endpoints')
    result=[];records=[]
    for o,b,p in zip(origin,baseline,positive):
        if not o.shape==b.shape==p.shape:raise ValueError('Mismatched endpoint shapes')
        o,b,p=o.double(),b.double(),p.double()
        if not all(torch.isfinite(v).all() for v in (o,b,p)):raise ArithmeticError('Nonfinite endpoint')
        u=b-o;v=p-o;n=float(u.norm())
        if n==0:
            if float(v.norm())!=0:raise ValueError('Nonzero proposal for zero baseline')
            negative=b.float();norm_error=0.;distance_error=0.
        else:
            reflected=2*float((u*v).sum()/u.square().sum())*u-v
            negative=(o+reflected).float();actual=negative.double()-o
            norm_error=abs(float(actual.norm())-n)/n
            distance_error=abs(float((negative.double()-b).norm())-float((p-b).norm()))/n
            if norm_error>5e-4 or distance_error>5e-4:raise ArithmeticError('Reflection FP32 mismatch')
        result.append(negative)
        records.append({'norm_relative_error':norm_error,'correction_length_relative_error':distance_error})
    return result,records


def choose_direction(grads,baseline,positive,negative):
    dots={key:sum(float((g.double()*(v.double()-b.double())).sum()) for g,b,v in zip(grads,baseline,values))
          for key,values in [('positive',positive),('negative',negative)]}
    if not all(torch.isfinite(torch.tensor(v)) for v in dots.values()):raise ArithmeticError('Nonfinite directional derivative')
    return ('negative' if dots['negative']<dots['positive']-1e-12 else 'positive'),dots
