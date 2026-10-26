"""Full-moment first stage of the boundary-query proposal.

No labels, task oracle, or fitting are used by compose_scores. Statistics are
FP64 centered scatter, not averaged client covariances. This module deliberately
does not implement the later conditional active-query experiment.
"""
from __future__ import annotations

import math
import torch


def moments(x):
    x = x.double()
    if x.ndim != 2 or not len(x) or not torch.isfinite(x).all():
        raise ValueError('Nonempty finite feature matrix required')
    mean = x.mean(0)
    centered = x - mean
    return {'n': len(x), 'mean': mean, 'scatter': centered.T @ centered}


def merge_moments(left, right):
    if left is None:
        return {k: v.clone() if torch.is_tensor(v) else v for k, v in right.items()}
    if left['mean'].shape != right['mean'].shape or min(left['n'], right['n']) <= 0:
        raise ValueError('Incompatible class statistics')
    n = left['n'] + right['n']
    delta = right['mean'] - left['mean']
    return {
        'n': n,
        'mean': left['mean'] + delta * (right['n'] / n),
        'scatter': left['scatter'] + right['scatter']
        + torch.outer(delta, delta) * (left['n'] * right['n'] / n),
    }


def pair_weights(classes, cpt, mode, device):
    if classes % cpt or mode not in ('bidirectional', 'no_old_to_new'):
        raise ValueError('Invalid class grouping or objective mode')
    task = torch.arange(classes, device=device) // cpt
    mask = task[:, None] != task[None, :]
    if mode == 'no_old_to_new':
        current = task.max()
        mask &= ~((task[:, None] < current) & (task[None, :] == current))
    # Preserve the original denominators in the ablation: only remove terms.
    denominator = (task[:, None] != task[None, :]).sum(1).clamp_min(1)
    return mask.double() / denominator[:, None] / classes


def objective_gradient(theta, means, covariances, weights, cfg):
    """Exact gradient of the epsilon-smoothed convex full-moment objective."""
    classes, dim = means.shape
    a, b = theta[:, :dim], theta[:, dim]
    difference = a[:, None, :] - a[None, :, :]
    sigma_d = torch.bmm(difference, covariances)
    variance = (sigma_d * difference).sum(-1)
    if float(variance.min()) < -1e-9:
        raise ArithmeticError('Non-PSD covariance or numerical failure')
    deviation = (variance.clamp_min(0) + cfg['variance_epsilon']).sqrt()
    mean_margin = (difference * means[:, None, :]).sum(-1) + b[:, None] - b[None, :]
    slack = (cfg['margin'] - mean_margin + cfg['kappa'] * deviation).clamp_min(0)
    reg = cfg['regularization'] / classes
    value = (weights * slack.square()).sum() + reg * theta.square().sum()
    coeff = 2 * weights * slack
    direction_gradient = -means[:, None, :] + cfg['kappa'] * sigma_d / deviation[:, :, None]
    local_gradient = coeff[:, :, None] * direction_gradient
    ga = local_gradient.sum(1) - local_gradient.sum(0) + 2 * reg * a
    gb = -coeff.sum(1) + coeff.sum(0) + 2 * reg * b
    gradient = torch.cat((ga, gb[:, None]), 1)
    return value, gradient


def project(theta, radius):
    out = theta.clone()
    norms = out[:, :-1].norm(dim=1, keepdim=True)
    out[:, :-1] *= (radius / norms.clamp_min(radius))
    return out


def solve_evidence(means, covariances, cfg, mode='bidirectional', callback=None):
    """Deterministic accelerated projected gradient with majorization/restart.

    Starting at zero and a bounded iteration count are part of the frozen
    protocol. Report stationarity rather than calling a finite iterate exact.
    """
    classes, dim = means.shape
    weights = pair_weights(classes, cfg['cpt'], mode, means.device)
    theta = torch.zeros((classes, dim + 1), dtype=torch.float64, device=means.device)
    extrapolated = theta.clone()
    momentum = 1.
    step = float(cfg['initial_step'])
    first = None
    trace = []
    converged = False
    iterations = 0
    for iteration in range(cfg['max_iterations']):
        if callback is not None:
            callback(iteration, theta)
        current_value, _ = objective_gradient(theta, means, covariances, weights, cfg)
        if first is None:
            first = float(current_value)
        step = min(step * 1.25, cfg['maximum_step'])
        for restart in range(2):
            value, grad = objective_gradient(extrapolated, means, covariances, weights, cfg)
            for backtrack in range(60):
                candidate = project(extrapolated - step * grad, cfg['radius'])
                delta = candidate - extrapolated
                next_value, _ = objective_gradient(candidate, means, covariances, weights, cfg)
                upper = value + (grad * delta).sum() + delta.square().sum() / (2 * step)
                if float(next_value) <= float(upper) + 1e-12 * max(1., abs(float(value))):
                    break
                step *= .5
            else:
                raise ArithmeticError('Projected solver line search failed')
            if float(next_value) <= float(current_value) + 1e-12:
                break
            # Restart from the feasible iterate when momentum raises the loss.
            extrapolated = theta.clone()
            momentum = 1.
        else:
            raise ArithmeticError('Convex objective unexpectedly increased')
        if not torch.isfinite(candidate).all() or not torch.isfinite(next_value):
            raise ArithmeticError('Nonfinite evidence iterate')
        next_momentum = (1 + math.sqrt(1 + 4 * momentum * momentum)) / 2
        extrapolated = candidate + (momentum - 1) / next_momentum * (candidate - theta)
        momentum = next_momentum
        theta = candidate
        iterations = iteration + 1
        # Unit-step gradient mapping is comparable across line-search steps.
        _, final_grad = objective_gradient(theta, means, covariances, weights, cfg)
        residual = float((theta - project(theta - final_grad, cfg['radius'])).norm())
        if iteration % 100 == 0 or residual <= cfg['stationarity_tolerance']:
            trace.append({'iteration': iterations, 'objective': float(next_value),
                          'gradient_mapping_norm': residual, 'step': step})
        if residual <= cfg['stationarity_tolerance']:
            converged = True
            break
    value, grad = objective_gradient(theta, means, covariances, weights, cfg)
    residual = float((theta - project(theta - grad, cfg['radius'])).norm())
    diagnostic = {
        'mode': mode, 'iterations': iterations, 'converged': converged,
        'initial_objective': first, 'objective': float(value),
        'gradient_mapping_norm': residual, 'max_row_norm': float(theta[:, :-1].norm(dim=1).max()),
        'radius': cfg['radius'], 'pair_count': int((weights > 0).sum()), 'trace': trace,
        'smoothed_variance_epsilon': cfg['variance_epsilon'],
        'exact_optimizer_claim': False,
    }
    if diagnostic['max_row_norm'] > cfg['radius'] + 1e-8:
        raise ArithmeticError('Evidence norm constraint failed')
    return theta, diagnostic


def evidence_scores(phi, theta):
    return phi.double() @ theta[:, :-1].T + theta[:, -1]


def compose_scores(logits, evidence=None, cpt=10):
    """log q(c|task,x) + logsumexp of reference evidence for every task."""
    logits = logits.double()
    if logits.ndim != 2 or logits.shape[1] % cpt:
        raise ValueError('Invalid task blocks')
    blocks = logits.reshape(len(logits), -1, cpt)
    within = torch.log_softmax(blocks, dim=2)
    if evidence is None:
        support = torch.zeros_like(within[:, :, :1])
    else:
        if evidence.shape != logits.shape:
            raise ValueError('Evidence must cover all seen classes')
        support = torch.logsumexp(evidence.reshape_as(blocks), dim=2, keepdim=True)
    result = within + support
    if not torch.equal(result.argmax(2), blocks.argmax(2)):
        raise ArithmeticError('Within-task ranking changed')
    return result.reshape_as(logits)


def lda_classifier(means, covariances, counts, shrinkage):
    """Standard all-class shared-covariance LDA; fixed shrinkage, equal priors.

    This diagnostic trains full class discrimination from the same statistics.
    It is not the paused historical ridge route or a tuned comparison method.
    """
    pooled = (covariances * counts[:, None, None]).sum(0) / counts.sum()
    dim = means.shape[1]
    regularized = (1 - shrinkage) * pooled + shrinkage * pooled.trace() / dim * torch.eye(
        dim, dtype=means.dtype, device=means.device)
    a = torch.linalg.solve(regularized, means.T).T
    b = -.5 * (means * a).sum(1)
    if not torch.isfinite(a).all():
        raise ArithmeticError('Reference LDA failed')
    return torch.cat((a, b[:, None]), 1)
