"""Fixed reference-anchored subspace chart for paired route comparison."""
import math
import torch


def reference_frame(pool, reference):
    """Orthonormalize pool while preserving reference's ordered frame first."""
    if pool.ndim != 2 or reference.ndim != 2:
        raise ValueError('expected matrices')
    d, width = pool.shape
    rank, ref_d = reference.shape
    if d != ref_d or not 0 < rank < width <= d:
        raise ValueError('incompatible pool and reference')
    p, a = pool.double(), reference.double()
    if not torch.isfinite(p).all() or not torch.isfinite(a).all():
        raise ValueError('nonfinite reference frame')
    # Orthonormalize the reference by positive Cholesky; unlike an eigensolver,
    # this keeps the same ordered frame for both optimizers.
    chol = torch.linalg.cholesky(a @ a.T)
    rref = torch.linalg.solve_triangular(chol, a, upper=False)
    old = rref.T
    rest = p[:, rank:] - old @ (old.T @ p[:, rank:])
    rest = rest - old @ (old.T @ rest)
    q, upper = torch.linalg.qr(rest, mode='reduced')
    diagonal = upper.diagonal()
    if len(diagonal) != width - rank or bool((diagonal.abs() < 1e-10).any()):
        raise ArithmeticError('candidate pool complement is rank deficient')
    q = q * torch.sign(diagonal).unsqueeze(0)
    e = torch.cat((old, q), dim=1)
    z0 = torch.eye(width, dtype=e.dtype, device=e.device)[:, :rank]
    n = torch.eye(width, dtype=e.dtype, device=e.device)[:, rank:]
    return e, z0, n


def graph_basis(e, z0, n, x):
    if (e.ndim != 2 or z0.ndim != 2 or n.ndim != 2 or x.ndim != 2
            or z0.shape[0] != e.shape[1] or n.shape[0] != e.shape[1]
            or x.shape != (n.shape[1], z0.shape[1])):
        raise ValueError('incompatible chart dimensions')
    if not torch.isfinite(x).all():
        raise ValueError('nonfinite graph coordinate')
    ident = torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
    lower = torch.linalg.cholesky(ident + x.T @ x)
    y = z0 + n @ x
    z = torch.linalg.solve_triangular(lower, y.T, upper=False).T
    return z.T @ e.T


def projector_distance(a, b):
    if a.shape != b.shape:
        raise ValueError('incompatible subspaces')
    qa = torch.linalg.qr(a.double().T, mode='reduced').Q
    qb = torch.linalg.qr(b.double().T, mode='reduced').Q
    return torch.linalg.matrix_norm(qa @ qa.T - qb @ qb.T)


def repair_candidate(e, z0, n, proposed_x, history, reference, lambdas, trust_radius, orthogonal_tolerance):
    """Finite spectral repair; never enlarge the fixed reference budget."""
    if not math.isfinite(trust_radius) or trust_radius <= 0 or not lambdas:
        raise ValueError('invalid finite search limits')
    if any(not math.isfinite(v) or v < 0 for v in lambdas):
        raise ValueError('invalid multiplier')
    h = history.double()
    if h.shape != (e.shape[0], e.shape[0]) or not torch.isfinite(h).all():
        raise ValueError('invalid history')
    h = (h + h.T) / 2
    if not torch.isfinite(proposed_x).all():
        raise ValueError('invalid proposal')
    base = reference.detach().to(dtype=torch.float32).double()
    budget = torch.trace(base @ h @ base.T)
    s = e.T @ h @ e
    values = torch.linalg.eigvalsh((s + s.T) / 2)
    rank = z0.shape[1]
    lowest = values[:rank].sum()
    candidate = graph_basis(e, z0, n, proposed_x)
    proposal_trace = torch.trace(candidate.float().double() @ h @ candidate.float().double().T)
    summary = {'budget': float(budget), 'pool_minimum': float(lowest),
               'budget_slack': float(budget-lowest), 'proposed_trace': float(proposal_trace),
               'proposed_excess': float(proposal_trace-budget), 'proposed_norm': float(proposed_x.norm())}
    z = e.T @ candidate.T
    pc = z @ z.T
    checks = []
    for lam in lambdas:
        eigenvalues, vec = torch.linalg.eigh((pc - lam * s + (pc - lam * s).T) / 2)
        if float(eigenvalues[-rank] - eigenvalues[-rank-1]) <= 1e-10:
            checks.append({'lambda': lam, 'feasible': False, 'reason': 'degenerate_projector'})
            continue
        p = vec[:, -rank:] @ vec[:, -rank:].T
        c = z0.T @ p @ z0
        try:
            if float(torch.linalg.cond(c)) > 1e10:
                raise ArithmeticError('ill-conditioned graph')
            x = torch.linalg.solve(c.T, (n.T @ p @ z0).T).T
            if not torch.isfinite(x).all() or float(x.norm()) > trust_radius:
                raise ArithmeticError('outside trust region')
            actual = graph_basis(e, z0, n, x).float()
            identity = torch.eye(rank, device=actual.device, dtype=actual.dtype)
            err = (actual @ actual.T - identity).abs().max()
            exposure = torch.trace(actual.double() @ h @ actual.double().T)
            valid = (torch.isfinite(exposure) and float(err) <= orthogonal_tolerance and exposure <= budget)
            checks.append({'lambda': lam, 'trace': float(exposure), 'excess': float(exposure-budget),
                           'orthogonal_error': float(err), 'feasible': bool(valid)})
            if valid:
                return x, {'accepted': True, **summary, 'attempts': checks}
        except (RuntimeError, ArithmeticError) as error:
            checks.append({'lambda': lam, 'feasible': False, 'reason': str(error)})
    return None, {'accepted': False, **summary, 'attempts': checks}
