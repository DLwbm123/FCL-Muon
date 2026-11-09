"""Functional uncorrected ClientMuon/NS5 and zero-weight-decay AdamW steps."""
import torch
from fcl.math import mulora_ssb


def muon_step(values, momenta, gradients, lr):
    if not (len(values) == len(momenta) == len(gradients)):
        raise ValueError('Muon state mismatch')
    next_m = [m * .9 + g.float() * .1 for m, g in zip(momenta, gradients)]
    return [p - lr * mulora_ssb(m) for p, m in zip(values, next_m)], next_m


def adamw_step(values, state, gradients, rates):
    if not (len(values) == len(gradients) == len(rates)):
        raise ValueError('AdamW state mismatch')
    if state is None:
        step = 0
        moment = [torch.zeros_like(p) for p in values]
        variance = [torch.zeros_like(p) for p in values]
    else:
        step, moment, variance = state
        if len(moment) != len(values) or len(variance) != len(values):
            raise ValueError('AdamW moment mismatch')
    step += 1
    moment = [.9*m + .1*g for m, g in zip(moment, gradients)]
    variance = [.999*v + .001*g.square() for v, g in zip(variance, gradients)]
    out = []
    for p, m, v, lr in zip(values, moment, variance, rates):
        mh = m / (1 - .9**step)
        vh = v / (1 - .999**step)
        # Preserve forward values while avoiding nonfinite higher derivatives at zero.
        root = torch.where(vh > 0, vh.clamp_min(torch.finfo(vh.dtype).tiny).sqrt(), torch.zeros_like(vh))
        out.append(p - lr * mh / (root + 1e-8))
    return out, (step, moment, variance)
