"""Historical patch-input response energy; diagnostic, not a safety certificate."""
import torch


def response_energy(a, bq, bv, history):
    """Energy of the two actual Q/V residuals on the recorded input Gram.

    Inputs represent deployed FP32 A/B; contraction is FP64. The historical
    Gram contains patch tokens only, so this does not measure CLS responses.
    """
    if a.ndim != 2 or bq.ndim != 2 or bv.ndim != 2 or history.ndim != 2:
        raise ValueError('invalid dimensions')
    rank, width = a.shape
    if history.shape != (width, width) or bq.shape[1] != rank or bv.shape[1] != rank:
        raise ValueError('incompatible dimensions')
    if any(not torch.isfinite(x).all() for x in (a, bq, bv, history)):
        raise ValueError('nonfinite response')
    aa, qq, vv, hh = (x.detach().double() for x in (a, bq, bv, history))
    hh = (hh + hh.T) * .5
    reduced = aa @ hh @ aa.T
    value = ((qq.T @ qq + vv.T @ vv) * reduced).sum()
    if not torch.isfinite(value) or float(value) < -1e-8:
        raise ArithmeticError('invalid historical energy')
    return max(float(value), 0.)


def within_reference(candidate, reference, *, relative=1e-5, absolute=1e-8):
    """Check the fixed numerical tolerance against the matched reference."""
    import math
    if (not math.isfinite(candidate) or not math.isfinite(reference) or
            candidate < 0 or reference < 0 or relative < 0 or absolute < 0):
        raise ValueError('invalid energy or tolerance')
    return candidate <= reference + max(absolute, relative * reference)
