"""Fixed-anchor weighted training gradient and a frozen correction's directional derivative."""
import math
import torch
from torch.nn import functional as F
from fcl.checkpoint import restore_rng, rng_state


def directional(pilot, client_splits, correction, part):
    if part not in (0, 1):
        raise ValueError('invalid split')
    groups = pilot.manifest['partitions']['train'][2]
    if len(client_splits) != len(groups):
        raise ValueError('client count mismatch')
    weights = [len(g) for g in groups]
    denominator = sum(w for w, pair in zip(weights, client_splits) if pair[0] and pair[1])
    if not denominator:
        raise ValueError('no client has both probes')
    model = pilot.model;params = model.b_parameters()
    if len(params) != len(correction) or any(p.shape != d.shape for p, d in zip(params, correction)):
        raise ValueError('correction shape mismatch')
    if any(not torch.isfinite(d).all() for d in correction):
        raise ValueError('nonfinite correction')
    modes = {m: m.training for m in model.modules()}
    buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
    parameters = list(model.parameters())
    old_grad = [None if v.grad is None else v.grad.detach().clone() for v in parameters]
    random_state = rng_state()
    total = [torch.zeros_like(p, dtype=torch.float64) for p in params]
    samples = 0
    try:
        model.eval()
        for k, pair in enumerate(client_splits):
            ids = pair[part]
            count = 0
            for x, y in pilot.loader(pilot.raw[0], ids, pilot.cfg['batch']):
                pilot.budget();pilot.check_labels(y)
                y = y.to(pilot.device)
                logits = model(pilot.test_tf(x.to(pilot.device))).double()
                loss = F.cross_entropy(logits, y, reduction='sum')
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite common-anchor loss')
                gradients = torch.autograd.grad(loss, params)
                with torch.no_grad():
                    for acc, grad in zip(total, gradients):
                        acc.add_(grad.detach().double(), alpha=weights[k] / (denominator * len(ids)))
                count += len(y);samples += len(y)
            if count != len(ids):
                raise ValueError('common-anchor sample count mismatch')
        dot = sum(float((g * d.to(device=g.device, dtype=torch.float64)).sum()) for g, d in zip(total, correction))
        g2 = sum(float((g*g).sum()) for g in total)
        d2 = sum(float((d.double()*d.double()).sum()) for d in correction)
        if not all(math.isfinite(v) for v in (dot, g2, d2)):
            raise ValueError('nonfinite directional result')
        return {'directional_derivative': dot, 'gradient_norm': math.sqrt(g2),
                'correction_norm': math.sqrt(d2),
                'cosine': dot / math.sqrt(g2*d2) if g2 and d2 else None,
                'samples': samples, 'weight_coverage': denominator / sum(weights)}
    finally:
        with torch.no_grad():
            for name, value in model.named_buffers():value.copy_(buffers[name])
        for parameter, value in zip(parameters, old_grad):parameter.grad = value
        for module, mode in modes.items():module.training = mode
        restore_rng(random_state)
