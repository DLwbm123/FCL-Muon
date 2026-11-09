"""Unregistered functional trial -> endpoint Gram -> aggregate response prototype."""
import torch
from torch.func import functional_call
from torch.nn import functional as F
from research.adamw_combined.differentiable_trial_draft import differentiable_trial
from research.adamw_combined.implicit_head_draft import certified_head_surrogate


class _Features(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.backbone.forward_features(x)[:, 0]


def differentiable_response(model, bases, clients, formal_sizes, formal_batches,
                            evaluate, epochs, task, cpt, lr, rounds, schedule_epochs, progress=None,
                            track_response=True):
    """Caller owns RNG, deterministic transforms, split isolation and budgets.

    No retained individual endpoints may be disclosed. FP32 head backward is an
    ideal implicit surrogate, not the exact derivative of the quantized forward.
    """
    if not clients or len(clients) != len(formal_sizes) or len(clients) != len(formal_batches):
        raise ValueError('client lengths differ')
    if any(n <= 0 for n in formal_sizes):
        raise ValueError('missing formal client')
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    pname = {id(p): n for n, p in params.items()}
    bname = {id(b): n for n, b in buffers.items()}
    names = [pname[id(p)] for p in model.b_parameters()]
    head_name = pname[id(model.rows[-1])]
    adjusted = dict(buffers)
    for name, a in model.adapters.items():
        adjusted[bname[id(a.A)]] = bases[name]
    results = [differentiable_trial(model, bases, data, epochs, task, cpt, lr,
                                    rounds, n, schedule_epochs, progress, track_response)
               for data, n in zip(clients, formal_batches)]
    with torch.set_grad_enabled(track_response):
        grams = []
        extractor = _Features(model)
        for result, batches in zip(results, clients):
            local = dict(params)
            local.update(zip(names, result['B']))
            local[head_name] = result['head']
            st = {'model.'+k: v for k, v in local.items()}
            sb = {'model.'+k: v for k, v in adjusted.items()}
            features = []
            for x, _ in batches:
                features.append(F.normalize(functional_call(extractor, (st, sb), (x,)), dim=1).double())
                if progress is not None:
                    progress('gram_visits', len(x))
            grams.append(sum(z.T @ z for z in features))
        trial_sizes = [sum(len(y) for _, y in batches) for batches in clients]
        merged_head, certificate = certified_head_surrogate([r['head'] for r in results], grams, trial_sizes)
        merged_b = [sum(r['B'][j] * (n/sum(formal_sizes)) for r, n in zip(results, formal_sizes))
                    for j in range(len(names))]
        merged = dict(params)
        merged.update(zip(names, merged_b))
        merged[head_name] = merged_head
        if len(evaluate) != len(clients):
            raise ValueError('evaluation client lengths differ')
        losses, current_losses, counts = [], [], []
        for batches, size in zip(evaluate, formal_sizes):
            n = sum(len(y) for _, y in batches)
            if not n:
                raise ValueError('empty evaluation client')
            local_loss = local_current = 0
            for x, y in batches:
                offset = task*cpt
                if bool(((y < offset) | (y >= offset+cpt)).any()):
                    raise ValueError('evaluation includes non-current labels')
                logits = functional_call(model, (merged, adjusted), (x,))
                if progress is not None:
                    progress('evaluation_visits', len(y))
                local_loss = local_loss + F.cross_entropy(logits, y, reduction='sum')
                local_current = local_current + F.cross_entropy(logits[:, offset:offset+cpt], y-offset, reduction='sum')
            losses.append(local_loss * (size / sum(formal_sizes)) / n)
            current_losses.append(local_current * (size / sum(formal_sizes)) / n)
            counts.append(n)
        return {'loss': sum(losses), 'current': sum(current_losses), 'head_certificate': certificate,
                'trial_visits': sum(r['train_visits'] for r in results),
                'gram_visits': sum(trial_sizes), 'evaluation_visits': sum(counts)}
