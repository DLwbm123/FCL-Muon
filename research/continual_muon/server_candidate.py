"""Experimental history-budgeted SSB proxy correction; not a trained method."""
import math
import torch
from fcl.math import mulora_ssb


def candidate(fedavg_update, momenta, weights, history_metric, angle=0.1, damping=1.0, tolerance=1e-10):
    """Return a norm-preserving candidate, or an exact FedAvg fallback.

    Each momentum must be expressed in the same current task basis as the update.
    The history matrix is only a response proxy, not an old-data safety guarantee.
    """
    if (fedavg_update.ndim != 2 or not momenta or len(weights) != len(momenta) or
            any(m.shape != fedavg_update.shape for m in momenta) or
            history_metric.shape != (fedavg_update.shape[1], fedavg_update.shape[1]) or
            not math.isfinite(angle) or angle < 0 or not math.isfinite(damping) or damping < 0 or
            not math.isfinite(tolerance) or tolerance < 0):
        raise ValueError('invalid server candidate inputs')
    tensors = [fedavg_update, history_metric, *momenta]
    if any(not torch.isfinite(x).all() for x in tensors):
        raise ValueError('nonfinite server candidate input')
    if any(not math.isfinite(w) or w < 0 for w in weights) or abs(sum(weights) - 1.) > 1e-12:
        raise ValueError('invalid client sample weights')
    u = fedavg_update.double()
    history = (history_metric.double() + history_metric.double().T) / 2
    if float(torch.linalg.eigvalsh(history).min()) < -1e-8:
        raise ValueError('history metric is not positive semidefinite')
    scale = float(torch.trace(history) / history.shape[0])
    if scale <= 0:
        return fedavg_update.clone(), {'accepted': False, 'reason': 'empty_history'}
    h = history / scale
    norm = torch.linalg.vector_norm(u)
    if float(norm) == 0 or angle == 0:
        return fedavg_update.clone(), {'accepted': False, 'reason': 'zero_update_or_angle'}
    mean_m = sum(w * m.double() for w, m in zip(weights, momenta))
    phi_mean = mulora_ssb(mean_m.float()).double()
    mean_phi = sum(w * mulora_ssb(m.float()).double() for w, m in zip(weights, momenta))
    # This is a proxy built from final local momenta, not a full-trajectory bias.
    discrepancy = phi_mean - mean_phi
    direction = -torch.linalg.solve(torch.eye(h.shape[0], device=h.device, dtype=h.dtype) + damping * h,
                                    discrepancy.T).T
    tangent = direction - (direction * u).sum() / norm.square() * u
    tangent_norm = torch.linalg.vector_norm(tangent)
    if float(tangent_norm) <= 1e-12:
        return fedavg_update.clone(), {'accepted': False, 'reason': 'zero_tangent'}
    proposal = (u + angle * norm * tangent / tangent_norm) / math.sqrt(1. + angle * angle)
    baseline_history = float(((u @ h) * u).sum())
    proposed_history = float(((proposal @ h) * proposal).sum())
    allowed = proposed_history <= baseline_history + tolerance * max(baseline_history, 1.)
    if not allowed:
        return fedavg_update.clone(), {'accepted': False, 'reason': 'history_budget',
                                        'baseline_history': baseline_history,
                                        'proposal_history': proposed_history}
    deployed = proposal.to(fedavg_update.dtype)
    deployed64 = deployed.double()
    deployed_history = float(((deployed64 @ h) * deployed64).sum())
    deployed_norm_error = float(abs(torch.linalg.vector_norm(deployed64) / norm - 1.))
    if (deployed_history > baseline_history + tolerance * max(baseline_history, 1.) or
            deployed_norm_error > 1e-6):
        return fedavg_update.clone(), {'accepted': False, 'reason': 'deployed_precision_budget',
                'baseline_history': baseline_history, 'proposal_history': proposed_history,
                'deployed_history': deployed_history, 'deployed_relative_norm_error': deployed_norm_error}
    return deployed, {'accepted': True, 'reason': 'accepted',
            'baseline_history': baseline_history, 'proposal_history': proposed_history,
            'deployed_history': deployed_history, 'deployed_relative_norm_error': deployed_norm_error,
            'proxy_norm': float(torch.linalg.vector_norm(discrepancy)),
            'precast_relative_norm_error': float(abs(torch.linalg.vector_norm(proposal) / norm - 1.))}
