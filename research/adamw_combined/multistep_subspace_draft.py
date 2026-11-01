"""Synthetic-stage subspace proposal from expanded-coordinate client endpoints."""
import torch


def propose_subspace(pool, baseline, history, endpoints, counts, tolerance=1e-5, relative_gap=1e-7):
    """Propose a global rank-r basis from weighted multi-step endpoint second moments.

    pool is d×m, baseline is r×d, endpoints are client×branch×out×m.
    This uses individual client endpoints or a securely aggregated second moment.
    No safety or performance claim follows from the historical trace check.
    """
    original = baseline.detach().clone()
    if pool.ndim != 2 or baseline.ndim != 2 or endpoints.ndim != 4:
        raise ValueError('invalid tensor rank')
    d, m = pool.shape; rank = baseline.shape[0]
    n = endpoints.shape[0]
    count = torch.as_tensor(counts, device=pool.device, dtype=torch.float64)
    if (baseline.shape[1] != d or not 0 < rank < m <= d or endpoints.shape[-1] != m or
            history.shape != (d, d) or count.shape != (n,) or not n or
            not torch.isfinite(pool).all() or not torch.isfinite(baseline).all() or
            not torch.isfinite(history).all() or not torch.isfinite(endpoints).all() or
            not torch.isfinite(count).all() or (count < 0).any() or count.sum() <= 0):
        raise ValueError('invalid inputs')
    p, a, h, e = pool.double(), baseline.double(), history.double(), endpoints.double()
    eye = torch.eye(m, device=p.device, dtype=p.dtype)
    if float(torch.linalg.matrix_norm(p.T @ p-eye)) > tolerance:
        return original, 'pool_not_orthonormal'
    if float(torch.linalg.matrix_norm(a @ a.T-torch.eye(rank, device=p.device, dtype=p.dtype))) > tolerance:
        return original, 'baseline_not_orthonormal'
    if float(torch.linalg.matrix_norm(p @ (p.T @ a.T)-a.T)) > tolerance:
        return original, 'baseline_not_in_pool'
    hn = float(torch.linalg.matrix_norm(h))
    if float(torch.linalg.matrix_norm(h-h.T)) > tolerance*max(hn, 1e-12):
        return original, 'history_not_symmetric'
    hp = p.T @ ((h+h.T)*.5) @ p
    hvals = torch.linalg.eigvalsh(hp)
    if float(hvals.min()) < -tolerance*max(float(hvals.abs().max()), 1e-12):
        return original, 'history_not_psd_in_pool'
    moment = torch.einsum('k,kbom,kbon->mn', count/count.sum(), e, e)
    values, vectors = torch.linalg.eigh((moment+moment.T)*.5)
    scale = max(float(values.abs().max()), 1e-12)
    if float(values[-rank]-values[-rank-1]) <= relative_gap*scale:
        return original, 'spectral_boundary_tie'
    if rank > 1 and float(values[-rank:].diff().abs().min()) <= relative_gap*scale:
        return original, 'selected_eigenvalue_tie'
    z = vectors[:, -rank:].flip(1)
    pivots = z.abs().argmax(dim=0)
    signs = torch.sign(z[pivots, torch.arange(rank, device=z.device)])
    candidate = (p @ (z*signs)).T.to(baseline.dtype)
    cd = candidate.double()
    if float(torch.linalg.matrix_norm(cd @ cd.T-torch.eye(rank, device=p.device, dtype=p.dtype))) > tolerance:
        return original, 'deployment_not_orthonormal'
    original_risk = float(torch.trace(a @ h @ a.T))
    candidate_risk = float(torch.trace(cd @ h @ cd.T))
    if candidate_risk > original_risk + tolerance*max(abs(original_risk), 1e-12):
        return original, 'history_budget_exceeded'
    return candidate, 'accepted'


def propose_from_trials(model, pools, histories, trials):
    """Build per-layer candidates from one wide-model trial's private endpoints."""
    names = list(model.adapters)
    if not names or set(names) != set(pools) or set(names) != set(histories) or not trials:
        raise ValueError('invalid trial or layer set')
    expected = [p for name in names for p in (model.adapters[name].Bq, model.adapters[name].Bv)]
    actual = model.b_parameters()
    if len(actual) != len(expected) or any(a is not b for a, b in zip(actual, expected)):
        raise ValueError('unexpected adapter parameter order')
    if any(len(t['B']) != len(expected) or t['samples'] <= 0 for t in trials):
        raise ValueError('incomplete client endpoints')
    counts = [t['samples'] for t in trials]
    candidates, reasons = {}, {}
    for j, name in enumerate(names):
        endpoints = torch.stack([torch.stack((t['B'][2*j], t['B'][2*j+1])) for t in trials])
        candidates[name], reasons[name] = propose_subspace(
            pools[name], model.adapters[name].A, histories[name], endpoints, counts)
    return candidates, reasons
