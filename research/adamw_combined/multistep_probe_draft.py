"""Synthetic-stage transactional runner for paired candidate simulations.

The callback owns fresh optimizers and must not change the model architecture.
This module does not implement candidate generation, data access, or selection.
"""
import copy
import torch
from fcl.checkpoint import rng_state, restore_rng


def paired_probes(model, candidates, run_candidate):
    """Run each candidate from one model/RNG state and restore it on all exits."""
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    values = {k: p.detach().clone() for k, p in params.items()}
    buffer_values = {k: b.detach().clone() for k, b in buffers.items()}
    grads = {k: None if p.grad is None else p.grad.detach().clone() for k, p in params.items()}
    flags = {k: p.requires_grad for k, p in params.items()}
    modes = {m: m.training for m in model.modules()}
    state = rng_state()

    def reset():
        current_params, current_buffers = dict(model.named_parameters()), dict(model.named_buffers())
        if (current_params.keys() != params.keys() or current_buffers.keys() != buffers.keys() or
                any(current_params[k] is not v for k, v in params.items()) or
                any(current_buffers[k] is not v for k, v in buffers.items())):
            raise RuntimeError('candidate changed model architecture')
        with torch.no_grad():
            for k, p in params.items():
                p.copy_(values[k]); p.requires_grad_(flags[k])
                p.grad = None if grads[k] is None else grads[k].clone()
            for k, b in buffers.items():
                b.copy_(buffer_values[k])
        for module, mode in modes.items():
            module.training = mode
        restore_rng(state)

    try:
        results = []
        for candidate in candidates:
            reset()
            results.append(copy.deepcopy(run_candidate(candidate)))
        return results
    finally:
        reset()


def select_from_client_losses(counts, baseline, candidate, history_baseline, history_candidate):
    """Choose candidate only for strict pooled improvement and no client regression.

    All inputs are current-training-side measurements; this is not a safety guarantee.
    Counts describe the evaluated examples, not total client dataset sizes.
    """
    counts = torch.as_tensor(counts, dtype=torch.float64)
    baseline = torch.as_tensor(baseline, dtype=torch.float64)
    candidate = torch.as_tensor(candidate, dtype=torch.float64)
    risks = torch.as_tensor([history_baseline, history_candidate], dtype=torch.float64)
    if (counts.ndim != 1 or not counts.numel() or baseline.shape != counts.shape or
            candidate.shape != counts.shape or not torch.isfinite(counts).all() or
            not torch.isfinite(baseline).all() or not torch.isfinite(candidate).all() or
            not torch.isfinite(risks).all() or (counts < 0).any() or
            (counts != counts.floor()).any() or counts.sum() <= 0 or (risks < 0).any()):
        raise ValueError('invalid candidate evaluation')
    used = counts > 0
    delta = candidate[used] - baseline[used]
    weighted = float((delta * counts[used]).sum() / counts[used].sum())
    accepted = weighted < 0 and bool((delta <= 0).all()) and history_candidate <= history_baseline
    return {'accepted': accepted, 'weighted_loss_change': weighted,
            'worst_client_loss_change': float(delta.max()),
            'history_baseline': float(history_baseline), 'history_candidate': float(history_candidate)}


def simulate_client(model, batches, epochs, task, classes_per_task, adapter_lr, rounds, formal_batches, schedule_epochs=None):
    """Run one candidate/client on supplied batches; caller supplies isolation and filtering.

    This prototype does not handle transforms, client shuffling, Gram fusion, or I/O.
    """
    import math
    from torch.nn import functional as F
    if epochs < 1 or rounds < 1 or formal_batches < 1 or not batches:
        raise ValueError('empty simulation budget')
    bs = model.b_parameters()
    if any(torch.count_nonzero(p) for p in bs):
        raise ValueError('simulation requires task-start zero B')
    offset = task * classes_per_task
    for x, y in batches:
        if len(x) != len(y) or not len(y) or ((y < offset) | (y >= offset+classes_per_task)).any():
            raise ValueError('non-current or invalid samples')
    def measure():
        model.eval(); total = 0.; count = 0
        with torch.no_grad():
            for x, y in batches:
                logits = model(x)[:, offset:offset+classes_per_task]
                total += float(F.cross_entropy(logits, y-offset, reduction='sum'))
                count += len(y)
        return total/count, count
    before, n = measure()
    adapter = torch.optim.AdamW(bs, lr=adapter_lr, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    head = torch.optim.AdamW([model.rows[-1]], lr=.001, weight_decay=0.)
    # The trial consumes a prefix of round zero's formal learning-rate schedule.
    formal_steps = (epochs if schedule_epochs is None else schedule_epochs)*formal_batches
    seen = steps = 0
    # Formal pilot keeps the model in evaluation mode during local optimization.
    model.eval()
    for _ in range(epochs):
        for x, y in batches:
            if steps >= formal_steps:
                raise ValueError('trial exceeds formal learning-rate schedule')
            adapter.param_groups[0]['lr'] = adapter_lr*.5*(1+math.cos(math.pi*steps/max(rounds*formal_steps-1, 1)))
            model.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x)[:, offset:offset+classes_per_task], y-offset)
            if not torch.isfinite(loss): raise ArithmeticError('nonfinite trial loss')
            loss.backward(); adapter.step(); head.step()
            seen += len(y); steps += 1
    after, observed = measure()
    if observed != n or not math.isfinite(before) or not math.isfinite(after):
        raise ArithmeticError('invalid trial measurements')
    return {'before': before, 'after': after, 'samples': n, 'train_visits': seen,
            'steps': steps, 'B': [p.detach().clone() for p in bs],
            'head': model.rows[-1].detach().clone()}


def simulate_candidates(model, candidates, clients, formal_batches, epochs, task, classes_per_task,
                        adapter_lr, rounds, trial_seed, schedule_epochs=None):
    """Paired candidate/client probes; batches and client records stay private.

    The caller must provide fixed, disjoint training-side batches and a legal basis.
    Per-client seeded RNG is paired across candidates, not the formal training stream.
    """
    import random
    import numpy as np
    if not clients or len(clients) != len(formal_batches) or not candidates:
        raise ValueError('invalid trial groups')
    if any(not batch for batch in clients):
        raise ValueError('empty clients need explicit coverage handling')
    names = set(model.adapters)
    if any(set(bases) != names for bases in candidates):
        raise ValueError('candidate adapter mismatch')

    def run_candidate(bases):
        with torch.no_grad():
            for name, basis in bases.items():
                model.adapters[name].A.copy_(basis)
        def run_client(index):
            seed = trial_seed + index
            if not 0 <= seed < 2**32: raise ValueError('invalid trial seed')
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
            return simulate_client(model, clients[index], epochs, task, classes_per_task,
                                   adapter_lr, rounds, formal_batches[index], schedule_epochs)
        return paired_probes(model, list(range(len(clients))), run_client)

    return paired_probes(model, candidates, run_candidate)
