"""Draft global tangent-cone server operator; not an evaluated method."""
import math
import torch


def project_cone(q, normals, tolerance=1e-12):
    """Euclidean projection onto at most two homogeneous halfspaces in FP64."""
    if q.ndim != 1 or len(normals) != 2 or any(g.shape != q.shape for g in normals):
        raise ValueError('expected one vector and two matching constraint normals')
    q = q.double();g = [v.double() for v in normals]
    if not torch.isfinite(q).all() or any(not torch.isfinite(v).all() for v in g):
        raise ValueError('nonfinite cone input')
    norms = [float(torch.linalg.vector_norm(v)) for v in g]
    eps = [tolerance * max(float(torch.linalg.vector_norm(q)) * n, 1.) for n in norms]
    def feasible(x):
        return all(float(torch.dot(v, x)) <= e for v, e in zip(g, eps))
    solutions = []
    if feasible(q):solutions.append((q.clone(), []))
    for i in range(2):
        n2 = float(torch.dot(g[i], g[i]))
        if n2 <= 1e-30:continue
        multiplier = float(torch.dot(q, g[i])) / n2
        if multiplier < 0:continue
        x = q - multiplier * g[i]
        if feasible(x):solutions.append((x, [i]))
    gram = torch.stack([torch.stack([torch.dot(a, b) for b in g]) for a in g])
    determinant = float(torch.linalg.det(gram))
    if determinant > 1e-14 * max(float(gram[0, 0] * gram[1, 1]), 1e-300):
        multiplier = torch.linalg.solve(gram, torch.stack([torch.dot(q, x) for x in g]))
        if bool((multiplier >= -tolerance).all()):
            x = q - sum(multiplier[i] * g[i] for i in range(2))
            if feasible(x):solutions.append((x, [0, 1]))
    if not solutions:
        return torch.zeros_like(q), {'solved': False, 'reason': 'numerical_fallback'}
    best, active = min(solutions, key=lambda pair: float(torch.linalg.vector_norm(pair[0] - q)))
    return best, {'solved': True, 'active_constraints': active,
                  'normal_dots': [float(torch.dot(v, best)) for v in g]}


def guarded_candidate(updates, directions, all_gradients, current_gradients, histories,
                      angle=.1, history_tolerance=1e-10):
    """One fixed global candidate with deployment-time checks and exact fallback."""
    arrays = (updates, directions, all_gradients, current_gradients)
    if not updates or any(len(v) != len(updates) for v in (*arrays[1:], histories)):
        raise ValueError('server matrix count mismatch')
    for i, u in enumerate(updates):
        if (u.ndim != 2 or any(v[i].shape != u.shape for v in arrays[1:]) or
                histories[i].shape != (u.shape[1], u.shape[1])):
            raise ValueError('server matrix shape mismatch')
    if not math.isfinite(angle) or angle <= 0 or not math.isfinite(history_tolerance) or history_tolerance < 0:
        raise ValueError('invalid fixed budget')
    if any(not torch.isfinite(x).all() for values in (*arrays, histories) for x in values):
        raise ValueError('nonfinite server input')
    flat = lambda values: torch.cat([v.detach().double().reshape(-1) for v in values])
    u, q, ga, gc = (flat(v) for v in arrays)
    norm2 = torch.dot(u, u)
    def fallback(reason, **details):
        return [x.clone() for x in updates], {'accepted': False, 'reason': reason, **details}
    if float(norm2) == 0:return fallback('zero_update')
    tangent = lambda x: x - torch.dot(x, u) / norm2 * u
    q, ga_t, gc_t = tangent(q), tangent(ga), tangent(gc)
    direction, cone = project_cone(q, [ga_t, gc_t])
    dnorm = torch.linalg.vector_norm(direction)
    if not cone['solved'] or float(dnorm) <= 1e-12:return fallback('empty_or_unsolved_cone')
    z = (u + angle * torch.sqrt(norm2) * direction / dnorm) / math.sqrt(1 + angle*angle)
    result, start = [], 0
    for old in updates:
        count = old.numel();result.append(z[start:start+count].reshape_as(old).to(old.dtype));start += count
    deployed = flat(result);delta = deployed - u
    norm_error = float(abs(torch.linalg.vector_norm(deployed) / torch.sqrt(norm2) - 1))
    all_dot, current_dot = float(torch.dot(ga, delta)), float(torch.dot(gc, delta))
    base_history = proposed_history = 0.
    for old, new, history in zip(updates, result, histories):
        h = (history.double() + history.double().T) / 2
        if float(torch.linalg.eigvalsh(h).min()) < -1e-8:
            raise ValueError('history proxy not positive semidefinite')
        scale = float(torch.trace(h) / h.shape[0])
        if scale <= 0:return fallback('empty_history')
        h = h / scale;old64, new64 = old.double(), new.double()
        base_history += float(((old64 @ h) * old64).sum())
        proposed_history += float(((new64 @ h) * new64).sum())
    receipt = {'all_directional': all_dot, 'current_directional': current_dot,
               'baseline_history': base_history, 'proposal_history': proposed_history,
               'deployed_relative_norm_error': norm_error, 'cone': cone}
    if (norm_error > 1e-6 or all_dot >= -1e-12 or current_dot > 1e-12 or
            proposed_history > base_history + history_tolerance * max(base_history, 1.)):
        return fallback('deployed_budget', **receipt)
    return result, {'accepted': True, 'reason': 'accepted', **receipt}
