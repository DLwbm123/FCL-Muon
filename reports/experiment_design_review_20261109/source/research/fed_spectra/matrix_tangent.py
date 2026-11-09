"""Training-only diagnostic of directions excluded by fixed singular vectors."""
import math
import torch


def tangent_candidate(origin, baseline, reference, gradient, epsilon=1e-12):
    if not origin.shape == baseline.shape == reference.shape == gradient.shape or baseline.ndim != 2:
        raise ValueError('Mismatched matrix inputs')
    if epsilon <= 0:
        raise ValueError('Invalid epsilon')
    o, b, ref, g = [v.detach().double() for v in (origin, baseline, reference, gradient)]
    if not all(torch.isfinite(v).all() for v in (o, b, ref, g)):
        raise ArithmeticError('Nonfinite inputs')
    d = b-o
    n = float(d.norm())
    record = {'identity_reason': None, 'input_norm': n}
    def identity(reason):
        record.update(identity_reason=reason, reference_angle=0., used_angle=0.,
                      norm_relative_error=0., actual_FP32_norm_relative_error=0.,
                      predicted_delta=0., actual_FP32_relative_correction=0.)
        return baseline.clone(), record
    if n == 0:
        return identity('zero_baseline')
    v = ref-o
    parallel = float((v*d).sum())/n
    theta = math.atan2(float((v-parallel*d/n).norm()), parallel)
    if theta <= epsilon:
        return identity('zero_reference_angle')
    tangent = g-(g*d).sum()/d.square().sum()*d
    tn = float(tangent.norm())
    if tn <= epsilon*max(float(g.norm()), 1e-30):
        return identity('zero_matrix_tangent')
    new = math.cos(theta)*d-math.sin(theta)*n*tangent/tn
    error = abs(float(new.norm())-n)/n
    out = (b+new-d).float()
    actual_error = abs(float((out.double()-o).norm())-n)/n
    if error > 1e-10 or actual_error > 5e-4 or not torch.isfinite(out).all():
        raise ArithmeticError('Matrix tangent norm mismatch')
    record.update(reference_angle=theta, used_angle=theta, norm_relative_error=error,
                  actual_FP32_norm_relative_error=actual_error, tangent_gradient_norm=tn,
                  gradient_norm=float(g.norm()), tangent_first_order_slope=-n*tn,
                  actual_FP32_relative_correction=float((out.double()-b).norm())/n,
                  predicted_delta=float((g*(out.double()-b)).sum()))
    return out, record


def gradient_coverage(origin, baseline, fit_gradient, check_gradient, candidate):
    d = baseline.double()-origin.double()
    g, c = fit_gradient.double(), check_gradient.double()
    n2 = float(d.square().sum())
    if n2 == 0:
        return {'zero_update': True}
    gf = g-(g*d).sum()/n2*d
    gc = c-(c*d).sum()/n2*d
    u, s, vh = torch.linalg.svd(d, full_matrices=False)
    modes = (u.T@g@vh.T).diag()
    spectral = (u*modes)@vh-(g*d).sum()/n2*d
    # This diagnostic is used on the full-rank, non-repeated frozen endpoints only.
    gaps = s[:-1]-s[1:]
    if float(s[-1]) <= 1e-12*float(s[0]) or bool((gaps <= 1e-10*s[0]).any()):
        raise ArithmeticError('Endpoint outside nondegenerate coverage diagnostic')
    norm_fit, norm_check = float(gf.norm()), float(gc.norm())
    return {'zero_update': False, 'spectral_fraction_of_tangent_energy': float(spectral.square().sum())/max(norm_fit**2,1e-60),
            'fit_tangent_norm': norm_fit, 'confirmation_tangent_norm': norm_check,
            'fit_confirmation_tangent_cosine': float((gf*gc).sum())/max(norm_fit*norm_check,1e-60),
            'confirmation_predicted_delta': float((c*(candidate.double()-baseline.double())).sum())}
