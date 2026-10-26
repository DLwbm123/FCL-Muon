"""Training-only Stiefel search in an expanded RRS pool; no test data API."""
import itertools
import torch
from fcl.math import rrs


def ssb_batch(g):
    """Same FP32 operator as mulora_ssb, independently for every client and Q/V."""
    x = g.float().transpose(-1, -2)
    x = x / (torch.linalg.vector_norm(x, dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(5):
        gram = x @ x.transpose(-1, -2)
        x = 3.4445*x + (-4.7750*gram + 2.0315*(gram @ gram)) @ x
    return x.transpose(-1, -2)


def candidate_pool(current, history, a0, width):
    """Preserve the exact original FP32 A0; append subsequent spectral directions."""
    rank, dim = a0.shape
    if not rank < width <= dim:
        raise ValueError('Candidate width must exceed formal rank and not dimension')
    expanded, info = rrs(current, history, width)
    old = a0.T.double()
    extra = expanded[:width-rank].T.double()
    for _ in range(2):
        extra = extra - old @ torch.linalg.solve(old.T @ old, old.T @ extra)
    q, r = torch.linalg.qr(extra, mode='reduced')
    if float(r.diag().abs().min()) < 1e-8:
        raise ArithmeticError('Additional spectral directions lost rank')
    pool = torch.cat((a0.T, q.float()), dim=1)
    z0 = torch.eye(width, device=a0.device, dtype=torch.float32)[:, :rank]
    assert torch.equal((pool @ z0).T, a0)
    return pool, z0, info


def quantities(z, c, weights, h, mode):
    # c: client x {Q,V} x output_dim x candidate_dim.
    projected = c @ z
    mean = torch.einsum('k,kvdr->vdr', weights, projected)
    local = ssb_batch(.1 * projected)
    federated = torch.einsum('k,kvdr->vdr', weights, local)
    central = ssb_batch(.1 * mean)
    direction = federated if mode == 'federated' else central
    utility = (mean * direction).sum()
    hz = z.double().T @ h @ z.double()
    gamma = hz.trace()
    # The actual formal trainer always averages LOCAL SSB updates, including B.
    risk = torch.einsum('vdr,rs,vds->', federated.double(), hz, federated.double())
    norm = torch.linalg.vector_norm(federated)
    return {'utility': utility, 'gamma': gamma, 'risk': risk,
            'direction': direction, 'federated_direction': federated,
            'federated_utility': (mean * federated).sum(), 'update_norm': norm}


def tangent(g, z):
    return g - z @ ((z.T @ g + g.T @ z) * .5)


def project_halfspaces(direction, normals, bounds):
    """Exact small Euclidean projection onto up to two linear inequalities."""
    if not normals:
        return direction
    flat = direction.reshape(-1).double()
    a = torch.stack([n.reshape(-1).double() for n in normals])
    b = torch.as_tensor(bounds, dtype=torch.float64, device=flat.device)
    best, distance = None, float('inf')
    for size in range(len(normals)+1):
        for active in itertools.combinations(range(len(normals)), size):
            candidate = flat
            if active:
                aa = a[list(active)]
                gram = aa @ aa.T
                if float(torch.linalg.eigvalsh(gram).min()) < 1e-12:
                    continue
                multiplier = torch.linalg.solve(gram, aa @ flat - b[list(active)])
                if bool((multiplier < -1e-8).any()):
                    continue
                candidate = flat - aa.T @ multiplier
            if bool((a @ candidate <= b + 1e-7).all()):
                value = float((candidate-flat).square().sum())
                if value < distance:
                    best, distance = candidate, value
    return torch.zeros_like(direction) if best is None else best.reshape_as(direction).to(direction)


def scalar_record(q):
    return {key: float(q[key].detach()) for key in
            ['utility', 'gamma', 'risk', 'federated_utility', 'update_norm']}


def fit_search(c, weights, h, z0, mode, settings, budget=lambda: None):
    """Monotone feasible search; only fit gradients enter this function."""
    baseline = quantities(z0, c, weights, h, mode)
    base = scalar_record(baseline)
    scale_u = max(abs(base['utility']), 1e-12)
    scale_g = max(abs(base['gamma']), 1e-12)
    scale_r = max(abs(base['risk']), 1e-12)
    z = z0.detach().clone()
    trace = []
    tolerance = settings['constraint_relative_tolerance']
    for iteration in range(settings['search_steps']):
        budget()
        z.requires_grad_(True)
        q = quantities(z, c, weights, h, mode)
        u, g, risk = q['utility']/scale_u, q['gamma']/scale_g, q['risk']/scale_r
        gu = tangent(torch.autograd.grad(u, z, retain_graph=True)[0], z)
        gg = tangent(torch.autograd.grad(g, z, retain_graph=True)[0], z)
        gr = tangent(torch.autograd.grad(risk, z)[0], z)
        if not all(torch.isfinite(v).all() for v in (gu, gg, gr)):
            raise ArithmeticError('Nonfinite selection gradient')
        norm = float(gu.norm())
        if norm < 1e-9:
            trace.append({'iteration': iteration, 'stop': 'stationary_objective'})
            break
        direction = gu.detach()/norm
        normals, bounds = [], []
        for grad, value in [(gg, float(g.detach())), (gr, float(risk.detach()))]:
            n = float(grad.norm())
            if n > 1e-9:
                normals.append(grad.detach()/n)
                # A small inward component at the boundary limits retraction curvature.
                bounds.append(-settings['inward_derivative'] if value >= 1-1e-5
                              else (1-value)/(2*settings['initial_step']*n))
        direction = project_halfspaces(direction, normals, bounds)
        z = z.detach()
        if float((gu.detach()*direction).sum()) <= 1e-10:
            trace.append({'iteration': iteration, 'stop': 'no_feasible_ascent'})
            break
        found = False
        step = settings['initial_step']
        with torch.no_grad():
            for _ in range(settings['backtracks']):
                trial, _ = torch.linalg.qr(z+step*direction, mode='reduced')
                trial_q = quantities(trial, c, weights, h, mode)
                valid = (float(trial_q['gamma']) <= base['gamma']+tolerance*scale_g
                         and float(trial_q['risk']) <= base['risk']+tolerance*scale_r
                         and float(trial_q['utility']) > float(q['utility'].detach())+1e-7*scale_u)
                if valid:
                    z = trial
                    trace.append({'iteration': iteration, 'step': step, **scalar_record(trial_q)})
                    found = True
                    break
                step *= .5
        if not found:
            trace.append({'iteration': iteration, 'stop': 'line_search_no_improvement'})
            break
    return z.detach(), {'baseline': base, 'fit_candidate': scalar_record(quantities(z, c, weights, h, mode)),
                        'trace': trace, 'steps_accepted': sum('step' in item for item in trace)}


def certify(pool, z, z0, c_fit, fit_weights, c_check, check_weights, h, mode, settings):
    """Check the single fit-selected FP32 candidate once; never feed back into search."""
    candidate_a = (pool @ z).T.contiguous().float()
    baseline_a = (pool @ z0).T.contiguous().float()
    # Recover coordinates of the ACTUAL deployed matrix, not pre-rounding Z.
    gram = pool.double().T @ pool.double()
    deployed_z = torch.linalg.solve(gram, pool.double().T @ candidate_a.double().T).float()
    reference_z = torch.linalg.solve(gram, pool.double().T @ baseline_a.double().T).float()
    q0 = quantities(reference_z, c_fit, fit_weights, h, mode)
    q1 = quantities(deployed_z, c_fit, fit_weights, h, mode)
    b0, b1 = scalar_record(q0), scalar_record(q1)
    mean_check = torch.einsum('k,kvdp->vdp', check_weights, c_check)
    check0 = float(((mean_check @ reference_z)*q0['direction']).sum())
    check1 = float(((mean_check @ deployed_z)*q1['direction']).sum())
    # Amplitude-matched diagnostics are logged, not used for candidate selection.
    matched = check1 * float(q0['direction'].norm()) / max(float(q1['direction'].norm()), 1e-12)
    actual_check0 = float(((mean_check @ reference_z)*q0['federated_direction']).sum())
    actual_check1 = float(((mean_check @ deployed_z)*q1['federated_direction']).sum())
    matched_actual = actual_check1*b0['update_norm']/max(b1['update_norm'], 1e-12)
    orth = float((candidate_a.double() @ candidate_a.double().T - torch.eye(z.shape[1],device=z.device)).norm())
    pool_residual = float((pool.double() @ deployed_z.double()-candidate_a.double().T).norm())
    tol = settings['constraint_relative_tolerance']
    rel = settings['minimum_relative_gain']
    checks = {'orthogonal_fp32': orth <= settings['orthogonal_tolerance'],
              'pool_reconstruction': pool_residual <= settings['orthogonal_tolerance'],
              'fit_gain': b1['utility'] > b0['utility']+rel*max(abs(b0['utility']),1e-12),
              'historical_exposure': b1['gamma'] <= b0['gamma']+tol*max(abs(b0['gamma']),1e-12),
              'actual_federated_history_response': b1['risk'] <= b0['risk']+tol*max(abs(b0['risk']),1e-12),
              'independent_check': check1 > max(0.,check0+rel*max(abs(check0),1e-12)),
              'finite': all(torch.isfinite(v).all().item() for v in [candidate_a,q1['utility'],q1['gamma'],q1['risk']])}
    accepted = all(checks.values())
    report = {'accepted': accepted, 'checks': checks, 'baseline': b0, 'candidate': b1,
              'check_baseline': check0, 'check_candidate': check1, 'check_candidate_matched_update_norm': matched,
              'actual_federated_check_baseline': actual_check0, 'actual_federated_check_candidate': actual_check1,
              'actual_federated_check_matched_update_norm': matched_actual,
              'orthogonal_error': orth, 'pool_reconstruction_error': pool_residual,
              'projector_distance': float((candidate_a.double().T@candidate_a.double()-baseline_a.double().T@baseline_a.double()).norm()),
              'single_check_no_research': True}
    return (candidate_a if accepted else baseline_a), report
