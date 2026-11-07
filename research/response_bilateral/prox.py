"""Differentiable B-only proximal map. Statistical/linear algebra is FP64."""
import math
import torch


PSD_RTOL = 1e-10


def finite(name, x):
    if not torch.isfinite(x).all():
        raise ArithmeticError('nonfinite ' + name)


def symmetric(x):
    return (x + x.T) * .5


def history_spectrum(residual, rank_cap):
    h = residual.detach().double()
    finite('historical residual', h)
    u, sigma, vh = torch.linalg.svd(h, full_matrices=False)
    top = float(sigma[0]) if sigma.numel() else 0.
    threshold = max(h.shape) * torch.finfo(torch.float64).eps * top
    numerical_rank = int((sigma > threshold).sum()) if top else 0
    q = min(rank_cap, numerical_rank)
    energy = float(sigma.square().sum())
    result = {'U': u[:, :q].cpu(), 'V': vh[:q].T.cpu(),
              'alpha': (sigma[:q] / top).square().cpu() if q else sigma[:0].cpu()}
    result['info'] = {'q': q, 'numerical_rank': numerical_rank,
        'rank_threshold': threshold, 'retained_energy': float(sigma[:q].square().sum())/energy if energy else 0.,
        'boundary_gap': float(sigma[q-1]-sigma[q]) if 0 < q < len(sigma) else None,
        'U_orthogonal_error': float((u[:, :q].T@u[:, :q]-torch.eye(q, device=h.device)).abs().max()) if q else 0.,
        'V_orthogonal_error': float((vh[:q]@vh[:q].T-torch.eye(q, device=h.device)).abs().max()) if q else 0.}
    return result


def geometry(A, V, alpha, C_cur, metric='response', eps=.01):
    a = A.double()
    v, w, c = (x.to(device=a.device, dtype=torch.float64) for x in (V, alpha, C_cur))
    if (a.ndim != 2 or v.shape != (a.shape[1], len(w)) or c.shape != (a.shape[1], a.shape[1])
            or metric not in ('response', 'euclidean') or not 0 < eps <= 1):
        raise ValueError('invalid response geometry')
    for name, x in (('A', a), ('V', v), ('alpha', w), ('C_cur', c)):
        finite(name, x)
    if bool((w < 0).any()):
        raise ValueError('negative spectral weight')
    av = a @ v
    j = symmetric((av * w[None]) @ av.T)
    k = symmetric(a @ c @ a.T)
    zero = bool(torch.count_nonzero(k) == 0)
    eye = torch.eye(len(a), dtype=a.dtype, device=a.device)
    # PSD validation is at the task boundary for fixed C; projected K is checked
    # for each candidate A. Tolerance is fixed, never adapted after a failure.
    if not zero:
        eig = torch.linalg.eigvalsh(k.detach())
        scale = float(eig.abs().max())
        if float(eig[0]) < -PSD_RTOL * max(scale, 1e-300) or float(k.trace()) <= 0:
            raise ArithmeticError('invalid projected response Gram')
    r = eye if metric == 'euclidean' or zero else (1-eps)*len(a)*k/k.trace() + eps*eye
    finite('R', r)
    torch.linalg.cholesky(r)
    return j, r, k, zero


def response_bilateral_prox(B_hat, A, U, V, alpha, C_cur, gamma=.1, metric='response', eps=.01):
    if not math.isfinite(gamma) or gamma < 0:
        raise ValueError('invalid gamma')
    if B_hat.ndim != 2 or A.ndim != 2 or B_hat.shape[1] != A.shape[0]:
        raise ValueError('invalid B/A shape')
    finite('B_hat', B_hat)
    if gamma == 0 or alpha.numel() == 0:
        return B_hat
    dtype, device = B_hat.dtype, B_hat.device
    b = B_hat.double()
    u, w = (x.to(device=device, dtype=torch.float64) for x in (U, alpha))
    if u.shape != (len(b), len(w)):
        raise ValueError('invalid output history')
    finite('U', u)
    j, r, _, _ = geometry(A.to(device), V, w, C_cur, metric, eps)
    if torch.count_nonzero(j) == 0:
        return B_hat
    z = u.T @ b
    matrices = r[None] + gamma*w[:, None, None]*j[None]
    y = torch.linalg.solve(matrices, (z@r).unsqueeze(-1)).squeeze(-1)
    out = b + u@(y-z)
    finite('proximal endpoint', out)
    # No zero-B shortcut: the linear B Jacobian must remain the proximal map.
    return out.to(dtype)


def effective_norm(b, a):
    b, a = b.double(), a.double()
    return (b @ (a@a.T) * b).sum().clamp_min(0).sqrt()


def server_map(b, a, memory, current, arm, gamma=.1, eps=.01):
    if arm == 'E0':
        return response_bilateral_prox(b, a, memory['U'], memory['V'], memory['alpha'], current, gamma=0), 1.
    if arm not in ('E1', 'E2', 'E3', 'E1E2'):
        raise ValueError('unknown experimental arm')
    metric = 'euclidean' if arm in ('E2', 'E1E2') else 'response'
    shadow = response_bilateral_prox(b, a, memory['U'], memory['V'], memory['alpha'], current, gamma, metric, eps)
    if arm not in ('E1', 'E1E2'):
        return shadow, 1.
    den, num = effective_norm(b, a), effective_norm(shadow, a)
    if float(den) == 0:
        if float(num) != 0:
            raise ArithmeticError('zero scalar-control denominator with nonzero shadow')
        return b, 1.
    if float(den) < torch.finfo(torch.float64).tiny**.5 and float(num) >= torch.finfo(torch.float64).tiny**.5:
        raise ArithmeticError('near-zero denominator in scalar control')
    c = num/den
    finite('scalar-control coefficient', c)
    return (b.double()*c).to(b.dtype), c


def diagnostics(b, out, previous, a, mem, current, arm, gamma=.1, eps=.01):
    b, out, previous, a = (x.detach().double() for x in (b, out, previous, a))
    u, v, w = (mem[k].to(b.device).double() for k in ('U', 'V', 'alpha'))
    metric = 'euclidean' if arm in ('E2', 'E1E2') else 'response'
    j, r, k, zero = geometry(a, v, w, current, metric, eps)
    def penalty(x):
        return float(((w.sqrt()[:, None]*(u.T@x@a@v))*w.sqrt()[None]).square().sum())
    delta = out-b
    response = float((delta@k*delta).sum())
    distance = float((delta@r*delta).sum())
    pb, po = penalty(b), penalty(out)
    active_gamma = 0. if arm == 'E0' else gamma
    residual = delta@r + active_gamma*(u*w[None])@(u.T@out)@j
    scale = float((b@r).norm())
    chol = torch.linalg.cholesky(r)
    tmp = torch.linalg.solve_triangular(chol, j, upper=False)
    white = torch.linalg.solve_triangular(chol, tmp.T, upper=False).T
    eigen = torch.linalg.eigvalsh(symmetric(white))
    rho = (active_gamma*w[:, None]*eigen[None]).flatten()
    quant = torch.tensor([0., .25, .5, .75, 1.], device=b.device, dtype=b.dtype)
    def qu(x): return torch.quantile(x, quant).tolist() if x.numel() else []
    def ends(x):
        vals = torch.linalg.eigvalsh(x)
        return [float(vals[0]), float(vals[-1])]
    before = float(effective_norm(b, a))
    correction = float(effective_norm(delta, a))
    return {'A_orthogonal_error': float((a@a.T-torch.eye(len(a), device=a.device)).abs().max()),
        **mem['info'], 'AV_fro': float((a@v).norm()), 'J_spectrum': ends(j), 'R_spectrum': ends(r),
        'R_condition': float(torch.linalg.cond(r)), 'rho_quantiles': qu(rho),
        'shrink_quantiles': qu(1/(1+rho)), 'penalty_before': pb, 'penalty_after': po,
        'objective_before': active_gamma*.5*pb, 'objective_after': .5*distance+active_gamma*.5*po,
        'KKT_relative_residual': float(residual.norm())/max(scale, 1e-300),
        'norm_before': before, 'norm_after': float(effective_norm(out, a)),
        'relative_correction': correction/before if before else 0.,
        'actual_round_step': float(effective_norm(out-previous, a)),
        'reference_response_change': response, 'empty_history': not len(w),
        'zero_overlap': bool(torch.count_nonzero(j) == 0), 'zero_response_metric': zero,
        'optimality_required': arm in ('E0', 'E2', 'E3')}
