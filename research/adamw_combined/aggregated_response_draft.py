"""Transactional paired candidate evaluation; no data loading or final selection."""
import torch
from torch.nn import functional as F
from research.adamw_combined.multistep_probe_draft import paired_probes, simulate_candidates
from research.head_regmean.solver import aggregate_heads


def endpoint_gram(model, batches):
    """Use the local endpoint encoder and the same normalized features as Head-RegMean."""
    gram = None
    with torch.no_grad():
        for x, _ in batches:
            z = F.normalize(model.backbone.forward_features(x)[:, 0], dim=1).double()
            value = z.T @ z
            gram = value if gram is None else gram + value
    if gram is None or not torch.isfinite(gram).all():
        raise ValueError('empty or nonfinite endpoint Gram')
    return gram.cpu()


def paired_aggregates(model, candidates, clients, formal_batches, formal_sizes, epochs,
                      task, classes_per_task, lr, rounds, trial_seed, measure,
                      schedule_epochs=None):
    """Measure actual trial FedAvg B plus Head-RegMean from the same task anchor.

    B uses formal client sizes; endpoint Gram and head use the trial sample counts.
    Candidate callbacks must not retain private tensors in a public report.
    """
    if (not candidates or not clients or len(clients) != len(formal_sizes) or
            len(clients) != len(formal_batches) or any(n <= 0 for n in formal_sizes)):
        raise ValueError('invalid client coverage')
    if any(torch.count_nonzero(p) for p in model.b_parameters()):
        raise ValueError('task anchor must have zero B')
    names = set(model.adapters)
    if any(set(candidate) != names for candidate in candidates):
        raise ValueError('candidate adapter mismatch')

    def evaluate(bases):
        with torch.no_grad():
            for name, basis in bases.items():
                model.adapters[name].A.copy_(basis)
        trials = simulate_candidates(model, [bases], clients, formal_batches, epochs, task,
                    classes_per_task, lr, rounds, trial_seed, schedule_epochs,
                    endpoint_callback=endpoint_gram)[0]
        sizes = [item['samples'] for item in trials]
        head, _, cert = aggregate_heads([x['head'].cpu() for x in trials],
                    [x['endpoint_extra'] for x in trials], sizes, True, alpha=.5)
        if not cert['deployed_fp32']['passed']:
            raise ArithmeticError('uncertified trial head')
        with torch.no_grad():
            for j, parameter in enumerate(model.b_parameters()):
                value = torch.zeros_like(parameter)
                for trial, n in zip(trials, formal_sizes):
                    value.add_(trial['B'][j].to(value.device), alpha=n/sum(formal_sizes))
                parameter.copy_(value)
            model.rows[-1].copy_(head.to(model.rows[-1].device))
        result = measure(model)
        if not isinstance(result, dict) or not all(torch.isfinite(torch.as_tensor(v)).all() for v in result.values()):
            raise ValueError('invalid aggregate evaluation')
        return {'measurement': result, 'train_visits': sum(x['train_visits'] for x in trials),
                'gram_visits': sum(sizes), 'trial_evaluation_visits': 2*sum(sizes),
                'B': [p.detach().clone() for p in model.b_parameters()],
                'head': model.rows[-1].detach().clone()}

    return paired_probes(model, candidates, evaluate)
