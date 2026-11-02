"""Unregistered one-step ordered-basis proposal with exact baseline fallback.

The caller supplies a paired response evaluator and must isolate data and RNG.
History is a proxy, not a guarantee of old-class performance.
"""
import math
import torch


def propose_step(bases, histories, evaluate, step, *, progress=None):
    if not math.isfinite(step) or step <= 0 or not bases or set(bases) != set(histories):
        raise ValueError('invalid proposal inputs')
    variables = {k: v.detach().clone().requires_grad_(True) for k, v in bases.items()}
    for k, a in variables.items():
        h = histories[k]
        if (a.ndim != 2 or h.shape != (a.shape[1], a.shape[1]) or
            not torch.isfinite(a).all() or not torch.isfinite(h).all()):
            raise ValueError('invalid basis or history')
        eye = torch.eye(a.shape[0], device=a.device, dtype=a.dtype)
        if not torch.allclose(a @ a.T, eye, atol=1e-5, rtol=1e-5):
            raise ValueError('baseline basis is not row orthonormal')
    before = evaluate(variables)
    if not torch.isfinite(before['loss']) or not torch.isfinite(before['current']):
        raise ValueError('nonfinite baseline response')
    if progress is not None:
        progress('outer_backward_attempts', 1)
    grads = torch.autograd.grad(before['loss'], list(variables.values()))
    tangent = {}
    for (k, a), g in zip(variables.items(), grads):
        sym = (g @ a.T + a @ g.T) / 2
        tangent[k] = g - sym @ a
    norm = torch.sqrt(sum(t.square().sum() for t in tangent.values()))
    if not torch.isfinite(norm) or float(norm) == 0:
        return {k: a.detach().clone() for k, a in bases.items()}, {'accepted': False, 'reason': 'zero_or_nonfinite_direction'}
    candidate = {}
    for k, a in variables.items():
        updated = a.detach() - step * tangent[k].detach() / norm.detach()
        q, r = torch.linalg.qr(updated.T, mode='reduced')
        diagonal = r.diagonal()
        if not torch.isfinite(q).all() or bool((diagonal.abs() < 1e-10).any()):
            return {n: v.detach().clone() for n, v in bases.items()}, {'accepted': False, 'reason': 'degenerate_retraction'}
        candidate[k] = (q * torch.sign(diagonal).unsqueeze(0)).T.contiguous()
        h = histories[k].double()
        old = torch.trace(a.detach().double() @ h @ a.detach().double().T)
        new = torch.trace(candidate[k].double() @ h @ candidate[k].double().T)
        if not torch.isfinite(new) or new > old:
            return {n: v.detach().clone() for n, v in bases.items()}, {'accepted': False, 'reason': 'history_budget'}
    after = evaluate(candidate)
    if not torch.isfinite(after['loss']) or not torch.isfinite(after['current']):
        raise ValueError('nonfinite candidate response')
    if not (after['loss'] < before['loss'].detach() and after['current'] <= before['current'].detach()):
        return {n: v.detach().clone() for n, v in bases.items()}, {'accepted': False, 'reason': 'response_gate'}
    return candidate, {'accepted': True, 'reason': 'response_improved'}
