"""Unregistered FP64 implicit derivative of the existing sphere head solver.

This is not a replacement for the deployed FP32 Head-RegMean path. It rejects
hard or singular KKT systems instead of claiming a derivative there.
"""
import torch
from research.head_regmean.solver import sphere_solve, aggregate_heads, shrink_gram
from torch.nn import functional as F


class UnsupportedHeadDerivative(ValueError):
    """The certified head exists, but its implicit derivative is unsupported."""


class _SphereImplicit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, q):
        if h.dtype != torch.float64 or q.dtype != torch.float64:
            raise ValueError('FP64 inputs required')
        u, lam, _, info = sphere_solve(h, q)
        if any(case != 'secular' for case in info['cases']):
            raise UnsupportedHeadDerivative('hard case has no asserted derivative')
        h = (h + h.T) / 2
        d = h.shape[0]
        kkt = torch.zeros((q.shape[0], d+1, d+1), dtype=h.dtype, device=h.device)
        kkt[:, :d, :d] = h.unsqueeze(0) + lam[:, None, None] * torch.eye(d, dtype=h.dtype, device=h.device)
        kkt[:, :d, d] = u
        kkt[:, d, :d] = u
        condition = torch.linalg.cond(kkt)
        if not torch.isfinite(condition).all() or bool((condition > 1e10).any()):
            raise UnsupportedHeadDerivative('singular or ill-conditioned KKT derivative')
        ctx.save_for_backward(kkt, u)
        return u

    @staticmethod
    def backward(ctx, grad_output):
        kkt, u = ctx.saved_tensors
        rhs = torch.cat((grad_output, grad_output.new_zeros((len(u), 1))), dim=1)
        v = torch.linalg.solve(kkt.transpose(-1, -2), rhs.unsqueeze(-1)).squeeze(-1)[:, :-1]
        vh = v.T @ u
        return -(vh + vh.T) / 2, v


def sphere_implicit(h, q):
    return _SphereImplicit.apply(h, q)


def certified_head_surrogate(heads, grams, sizes, alpha=.5):
    """Certified production forward; ideal-FP64 implicit surrogate backward.

    The gradient is a surrogate through FP32 quantization and is not the
    mathematical derivative of the complete deployed implementation.
    """
    if not heads or len(heads) != len(grams) or len(heads) != len(sizes) or min(sizes) <= 0:
        raise ValueError('invalid client inputs')
    actual, _, info = aggregate_heads([x.detach() for x in heads],
                                       [x.detach() for x in grams], sizes, True, alpha)
    scaled_grams = [shrink_gram(g, alpha) / sum(sizes) for g in grams]
    h = sum(scaled_grams)
    effective = [F.normalize(w.float(), dim=1).double() for w in heads]
    q = sum(w @ g for w, g in zip(effective, scaled_grams))
    average = sum(w.float() * (n / sum(sizes)) for w, n in zip(heads, sizes))
    ideal = sphere_implicit(h, q) * average.double().norm(dim=1, keepdim=True).clamp_min(1e-12)
    return actual.detach() + (ideal.float() - ideal.float().detach()), info
