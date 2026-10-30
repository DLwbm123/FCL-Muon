"""Geometry for a private, read-only Muon boundary observation."""
import numpy as np


def measure(old_m, old_a, new_a, ssb):
    m, a, b = (np.asarray(x, dtype=np.float64) for x in (old_m, old_a, new_a))
    if (m.ndim != 2 or a.ndim != 2 or b.ndim != 2 or
            m.shape[1] != a.shape[0] or a.shape != b.shape):
        raise ValueError('incompatible momentum and row-basis shapes')
    if not all(np.isfinite(x).all() for x in (m, a, b)):
        raise ValueError('nonfinite input')
    eye = np.eye(a.shape[0])
    orth_old = float(np.linalg.norm(a @ a.T - eye))
    orth_new = float(np.linalg.norm(b @ b.T - eye))
    if max(orth_old, orth_new) > 1e-4:
        raise ValueError('basis is not row-orthonormal')
    source = m @ a
    projected_m = (m @ a) @ b.T
    projected = projected_m @ b
    residual = source - projected
    norm2 = float(np.sum(source * source))
    proj2 = float(np.sum(projected * projected))
    residual2 = float(np.sum(residual * residual))
    pyth_error = abs(norm2 - proj2 - residual2) / max(norm2, 1e-30)
    orth_error = float(np.linalg.norm(residual @ b.T)) / max(np.sqrt(norm2), 1e-30)
    old_update = np.asarray(ssb(m.astype(np.float32)), dtype=np.float64) @ a
    new_update = np.asarray(ssb(projected_m.astype(np.float32)), dtype=np.float64) @ b
    if not np.isfinite(old_update).all() or not np.isfinite(new_update).all():
        raise ValueError('nonfinite SSB output')
    old_norm = float(np.linalg.norm(old_update))
    new_norm = float(np.linalg.norm(new_update))
    cosine = float(np.sum(old_update * new_update) / (old_norm * new_norm)) if old_norm and new_norm else None
    return {'source_norm_squared': norm2, 'projected_norm_squared': proj2,
            'residual_norm_squared': residual2,
            'retained_energy': proj2 / norm2 if norm2 else None,
            'relative_residual': np.sqrt(residual2 / norm2) if norm2 else None,
            'pythagorean_relative_error': pyth_error, 'residual_orthogonality_error': orth_error,
            'old_basis_orthogonality_error': orth_old, 'new_basis_orthogonality_error': orth_new,
            'old_ssb_weight_norm': old_norm, 'new_ssb_weight_norm': new_norm,
            'ssb_weight_cosine': cosine}


def last_momentum_proxy(momenta, weights, ssb):
    """A same-basis last-state proxy; not a bias estimate for the training trajectory."""
    values = np.asarray(momenta, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float64)
    if (values.ndim != 3 or weights.shape != (len(values),) or
            not np.isfinite(values).all() or not np.isfinite(weights).all() or
            np.any(weights < 0) or not np.isclose(weights.sum(), 1., atol=1e-12, rtol=0)):
        raise ValueError('invalid last states or sample weights')
    mean = np.einsum('k,kij->ij', weights, values, dtype=np.float64).astype(np.float32)
    transformed_mean = np.asarray(ssb(mean), dtype=np.float64)
    mean_transformed = sum(w * np.asarray(ssb(v), dtype=np.float64) for w, v in zip(weights, values))
    if not np.isfinite(transformed_mean).all() or not np.isfinite(mean_transformed).all():
        raise ValueError('nonfinite SSB output')
    difference = float(np.linalg.norm(transformed_mean - mean_transformed))
    reference = float(np.linalg.norm(mean_transformed))
    return {'difference_norm': difference, 'mean_transformed_norm': reference,
            'transformed_mean_norm': float(np.linalg.norm(transformed_mean)),
            'relative_difference': difference / reference if reference else None}
