"""Bounded fixed-chart response selection; protocol values supplied by caller."""
import math
import torch
from research.adamw_combined.implicit_head_draft import UnsupportedHeadDerivative
from research.route_compare_20261102.frame import graph_basis, projector_distance, repair_candidate
from research.response_drift_20261103.energy import within_reference


def canonical_references(frames):
    """Exact same ordered FP32 reference for each paired optimizer/control."""
    references = {}
    for name, (e, z0, n) in frames.items():
        zero = torch.zeros((n.shape[1], z0.shape[1]), device=e.device, dtype=e.dtype)
        references[name] = graph_basis(e, z0, n, zero).float()
    return references


def select_basis(frames, histories, evaluate, *, steps, lambdas, trust_radius,
                 orthogonal_tolerance, loss_tolerance, space_tolerance, outer_iterations, frozen_layers=(),
                 energy_gate=False):
    """Evaluate complete aggregated models; caller supplies deterministic response.

    evaluate(bases, track_response) returns real forward values including a
    differentiable loss when track_response=True. No model state is deployed.
    """
    if set(frames) != set(histories) or not frames or not 1 <= outer_iterations <= 2:
        raise ValueError('invalid layers or search budget')
    if not steps or len(steps) > 3 or any(not math.isfinite(v) or v <= 0 for v in steps):
        raise ValueError('invalid fixed step schedule')
    if any(not math.isfinite(v) or v < 0 for v in (orthogonal_tolerance, loss_tolerance, space_tolerance)):
        raise ValueError('invalid tolerances')
    frozen_layers = set(frozen_layers)
    if not frozen_layers <= set(frames):
        raise ValueError('unknown frozen layer')
    reference = canonical_references(frames)
    current_x = {name: torch.zeros((n.shape[1], z.shape[1]), dtype=e.dtype, device=e.device)
                 for name, (e, z, n) in frames.items()}
    def actual(xs):
        return {name: graph_basis(*frames[name], x).float() for name, x in xs.items()}
    def measure(xs, track):
        result = evaluate(actual(xs), track)
        loss, current = result['loss'], result['current']
        if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(current)):
            raise ArithmeticError('nonfinite actual response')
        return result
    first = measure(current_x, False)
    ref_loss, ref_current = float(first['loss'].detach()), float(first['current'].detach())
    reference_energy = first.get('energy')
    if reference_energy is not None and set(reference_energy) != set(frames):
        raise ValueError('energy layer mismatch')
    if energy_gate and reference_energy is None:
        raise ValueError('energy gate requires response energies')
    best_loss = ref_loss
    history = []
    for iteration in range(outer_iterations):
        xs = {name: value.detach().clone().requires_grad_(True) for name, value in current_x.items()}
        try:
            result = measure(xs, True)
            grads = torch.autograd.grad(result['loss'], list(xs.values()))
        except UnsupportedHeadDerivative as error:
            history.append({'iteration': iteration, 'reason': 'unsupported_head_derivative', 'detail': str(error)})
            break
        norm = torch.sqrt(sum(g.double().square().sum() for g in grads))
        if not bool(torch.isfinite(norm)) or float(norm) == 0:
            history.append({'iteration': iteration, 'reason': 'zero_or_nonfinite_gradient'})
            break
        accepted = False
        for step in steps:
            proposed = {name: value.detach() - step * grad.detach() / norm for (name, value), grad in zip(xs.items(), grads)}
            repaired, layers = {}, {}
            for name, x in proposed.items():
                if name in frozen_layers:
                    repaired[name] = current_x[name]
                    layers[name] = {'accepted': False, 'reason': 'candidate_pool_failure'}
                    continue
                e, z, n = frames[name]
                trial, diag = repair_candidate(e, z, n, x, histories[name], reference[name],
                                                lambdas, trust_radius, orthogonal_tolerance)
                repaired[name] = current_x[name] if trial is None else trial
                layers[name] = diag
            bases = actual(repaired)
            distances = {name: float(projector_distance(bases[name], reference[name])) for name in bases}
            if all(float(projector_distance(bases[name], actual(current_x)[name])) <= space_tolerance for name in bases):
                history.append({'iteration': iteration, 'step': step, 'reason': 'no_space_change',
                                'layers': layers, 'reference_distances': distances})
                continue
            try:
                out = measure(repaired, False)
            except UnsupportedHeadDerivative as error:
                history.append({'iteration': iteration, 'step': step, 'reason': 'unsupported_head_derivative',
                                'detail': str(error), 'layers': layers, 'reference_distances': distances})
                continue
            loss, cur = float(out['loss'].detach()), float(out['current'].detach())
            energies = out.get('energy')
            if reference_energy is not None and (energies is None or set(energies) != set(frames)):
                raise ValueError('candidate energy layer mismatch')
            failed_energy = [] if energies is None else [name for name in frames
                if not within_reference(energies[name], reference_energy[name])]
            loss_valid = loss < best_loss - loss_tolerance and cur <= ref_current + loss_tolerance
            valid = loss_valid and (not energy_gate or not failed_energy)
            reason = 'accepted' if valid else ('response_gate' if not loss_valid else 'energy_gate')
            history.append({'iteration': iteration, 'step': step, 'reason': reason,
                            'loss': loss, 'current': cur, 'layers': layers, 'reference_distances': distances,
                            'energy': energies, 'energy_failed_layers': failed_energy})
            if valid:
                current_x, best_loss, accepted = {k: v.detach() for k, v in repaired.items()}, loss, True
                break
        if not accepted:
            break
    selected = actual(current_x)
    distances = {name: float(projector_distance(selected[name], reference[name])) for name in selected}
    return selected, {'reference_loss': ref_loss, 'reference_current': ref_current, 'selected_loss': best_loss,
                      'reference_energy': reference_energy, 'energy_gate': energy_gate,
                      'reference_distances': distances,
                      'changed_layers': sum(v > space_tolerance for v in distances.values()), 'history': history}
