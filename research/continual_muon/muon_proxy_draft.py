"""Unregistered cross-client Muon terminal-state proxy; synthetic development only."""
import math
import torch
from fcl.math import mulora_ssb


def discrepancy(momenta, weights, operator=mulora_ssb):
    if not momenta or len(momenta)!=len(weights) or any(x.shape!=momenta[0].shape or x.ndim!=2 for x in momenta):
        raise ValueError('momentum shape mismatch')
    if any(not math.isfinite(w) or w<0 for w in weights) or abs(sum(weights)-1)>1e-12:
        raise ValueError('invalid client weights')
    if any(not torch.isfinite(x).all() for x in momenta):raise ValueError('nonfinite momentum')
    avg=sum(w*x.detach().double() for w,x in zip(weights,momenta))
    transformed=operator(avg.float()).double()
    individual=sum(w*operator(x.detach().float()).double() for w,x in zip(weights,momenta))
    if not torch.isfinite(transformed).all() or not torch.isfinite(individual).all():
        raise ValueError('nonfinite operator output')
    return transformed-individual


def history_damped_direction(momenta, weights, history):
    """Returns a proxy in common current basis; not a multistep gradient bias."""
    q=discrepancy(momenta,weights);h=(history.double()+history.double().T)/2
    if h.shape!=(q.shape[1],q.shape[1]) or not torch.isfinite(h).all():raise ValueError('invalid history')
    if float(torch.linalg.eigvalsh(h).min()) < -1e-8:raise ValueError('indefinite history')
    scale=float(torch.trace(h)/h.shape[0])
    if scale<=0:return torch.zeros_like(q), {'reason':'empty_history'}
    h=h/scale
    direction=-torch.linalg.solve(torch.eye(h.shape[0],dtype=h.dtype,device=h.device)+h,q.T).T
    return direction, {'reason':'computed','proxy_norm':float(torch.linalg.vector_norm(q))}
