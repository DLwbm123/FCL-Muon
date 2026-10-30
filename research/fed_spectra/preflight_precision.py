"""Layer-scale FP32 check against a separate high-precision loss oracle."""
import torch


def gradient_error(actual,reference):
    if not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        raise ArithmeticError('Nonfinite preflight gradient')
    error=(actual.double()-reference.double()).norm().item()
    scale=reference.double().norm().item()
    relative=error/max(scale,1e-30)
    if relative>1e-4:raise AssertionError('FP32 gradient exceeds relative L2 tolerance')
    return {'relative_L2_error':relative,'max_absolute_error':float((actual-reference).abs().max())}
