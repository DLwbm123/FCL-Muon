"""Experimental task-boundary state transport, not a trained method."""
import math
import torch


def candidate(momenta, old_bases, new_bases):
    """Project each Q/V momentum in weight space onto its new shared row basis."""
    if not old_bases or len(old_bases) != len(new_bases) or len(momenta) != 2 * len(old_bases):
        raise ValueError('expected two momentum matrices per basis')
    result, original_energy, kept_energy = [], 0., 0.
    for j, momentum in enumerate(momenta):
        old, new = old_bases[j // 2].double(), new_bases[j // 2].double()
        m = momentum.double()
        if (old.ndim != 2 or old.shape != new.shape or m.ndim != 2 or
                m.shape[1] != old.shape[0] or not torch.isfinite(m).all() or
                not torch.isfinite(old).all() or not torch.isfinite(new).all()):
            raise ValueError('invalid momentum or basis')
        eye = torch.eye(old.shape[0], dtype=old.dtype, device=old.device)
        if max(float(torch.linalg.matrix_norm(old @ old.T - eye)),
               float(torch.linalg.matrix_norm(new @ new.T - eye))) > 1e-4:
            raise ValueError('row basis is not orthonormal')
        x = m @ old
        projected = x @ new.T
        original_energy += float((x * x).sum())
        y = projected @ new
        kept_energy += float((y * y).sum())
        result.append(projected.to(dtype=momentum.dtype, device=momentum.device))
    return result, kept_energy / original_energy if original_energy else None


def select(momenta, gradients, retained, minimum_retained=.05):
    """Return projected state only when a current-task training probe aligns."""
    if len(momenta) != len(gradients) or not momenta:
        raise ValueError('invalid momentum or probe list')
    dot = m2 = g2 = 0.
    for momentum, gradient in zip(momenta, gradients):
        if (momentum.shape != gradient.shape or not torch.isfinite(momentum).all() or
                not torch.isfinite(gradient).all()):
            raise ValueError('invalid training probe')
        m, g = momentum.double(), gradient.double()
        dot += float((m * g).sum())
        m2 += float((m * m).sum())
        g2 += float((g * g).sum())
    cosine = dot / math.sqrt(m2 * g2) if m2 and g2 else None
    accepted = bool(retained is not None and math.isfinite(retained) and
                    retained >= minimum_retained and cosine is not None and cosine >= 0.)
    selected = [v.clone() if accepted else torch.zeros_like(v) for v in momenta]
    return selected, {'accepted': accepted, 'retained_energy': retained, 'probe_cosine': cosine}
