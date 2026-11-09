"""Functional multi-round federated response with production head forward.

The implicit head backward is a surrogate through FP32 quantization. Batch
selection, paired RNG, resource supervision and selection are caller-owned.
"""
import math
import torch
from torch.nn import functional as F
from torch.func import functional_call
from torch.utils.checkpoint import checkpoint
from research.adamw_combined.differentiable_response_draft import _Features
from research.adamw_combined.implicit_head_draft import certified_head_surrogate
from research.head_regmean.solver import aggregate_heads
from research.route_compare_20261102.functional_optim import muon_step, adamw_step


def federated_response(model, bases, support, gram_batches, query, formal_sizes, formal_batches,
                       optimizer, lr, total_rounds, formal_epochs, task, classes_per_task,
                       *, use_checkpoint=True, track_response=True, server_operator=None):
    """Two or more genuine aggregated short rounds; no source model mutation.

    support is [round][client][step] of pre-paired (x,y) batches. Every B
    starts at the global value each round. Client Muon moments persist across
    short rounds; B/head AdamW state resets per client per round as in trainer.
    Raw sampled Gram sums are scaled once by formal_N/sample_N before the
    production solver, whose raw sums must not be multiplied by N again.
    """
    if model.training or optimizer not in ('adamw', 'muon') or not support:
        raise ValueError('invalid response configuration')
    original_clients = len(formal_sizes)
    if (original_clients == 0 or len(formal_batches) != original_clients or len(gram_batches) != original_clients
            or len(query) != original_clients or any(n < 0 for n in formal_sizes)
            or any(n < 0 for n in formal_batches) or set(bases) != set(model.adapters)):
        raise ValueError('invalid client structure')
    if any(len(r) != original_clients for r in support):
        raise ValueError('missing client in short round')
    for k, (size, batches) in enumerate(zip(formal_sizes, formal_batches)):
        if size == 0:
            if batches or gram_batches[k] or query[k] or any(round_[k] for round_ in support):
                raise ValueError('empty client received data')
        elif batches == 0:
            raise ValueError('nonempty client missing formal schedule')
    active_clients = [k for k, size in enumerate(formal_sizes) if size]
    if not active_clients:
        raise ValueError('no participating clients')
    support = [[round_[k] for k in active_clients] for round_ in support]
    gram_batches = [gram_batches[k] for k in active_clients]
    query = [query[k] for k in active_clients]
    formal_sizes = [formal_sizes[k] for k in active_clients]
    formal_batches = [formal_batches[k] for k in active_clients]
    clients = len(active_clients)
    params, buffers = dict(model.named_parameters()), dict(model.named_buffers())
    pnames = {id(p): name for name, p in params.items()}
    bnames = {id(b): name for name, b in buffers.items()}
    bnames_param = [pnames[id(p)] for p in model.b_parameters()]
    head_name = pnames[id(model.rows[-1])]
    bnames_basis = [bnames[id(model.adapters[k].A)] for k in bases]
    base_values = list(bases.values())
    if any(torch.count_nonzero(p) for p in model.b_parameters()):
        raise ValueError('short response requires zero-B task anchor')
    if any(a.shape != model.adapters[k].A.shape for k, a in bases.items()):
        raise ValueError('invalid basis shape')
    extractor = _Features(model)
    n_b = len(bnames_param)

    def call_model(x, variables, selected_bases, *, features=False):
        def evaluate(*items):
            values, chosen = items[:n_b+1], items[n_b+1:]
            p = dict(params)
            p.update(zip(bnames_param, values[:n_b]))
            p[head_name] = values[-1]
            b = dict(buffers)
            b.update(zip(bnames_basis, chosen))
            if features:
                return functional_call(extractor, ({'model.'+k: v for k, v in p.items()},
                                                  {'model.'+k: v for k, v in b.items()}), (x,))
            return functional_call(model, (p, b), (x,))
        inputs = tuple(variables) + tuple(selected_bases)
        if use_checkpoint and torch.is_grad_enabled():
            return checkpoint(evaluate, *inputs, use_reentrant=False, preserve_rng_state=True)
        return evaluate(*inputs)

    global_b = [p.detach().clone().requires_grad_(True) for p in model.b_parameters()]
    global_head = model.rows[-1].detach().clone().requires_grad_(True)
    muon_states = [[torch.zeros_like(p) for p in global_b] for _ in range(clients)]
    counts = {'trial_visits': 0, 'trial_steps': 0, 'gram_visits': 0, 'query_visits': 0,
              'head_fusions': 0}
    certificates = []
    total_size = sum(formal_sizes)
    for round_no, round_clients in enumerate(support):
        endpoints, heads, grams = [], [], []
        for k, batches in enumerate(round_clients):
            if not batches:
                raise ValueError('nonempty formal client has no trial batch')
            values = [b.clone() for b in global_b] + [global_head.clone()]
            adam_state, head_state = None, None
            formal_steps = formal_epochs * formal_batches[k]
            for step, (x, y) in enumerate(batches):
                if len(x) != len(y) or not len(y) or step >= formal_steps:
                    raise ValueError('invalid short-round schedule')
                lo = task * classes_per_task
                if bool(((y < lo) | (y >= lo+classes_per_task)).any()):
                    raise ValueError('trial batch has non-current label')
                logits = call_model(x, values, base_values)
                loss = F.cross_entropy(logits[:, lo:lo+classes_per_task], y-lo)
                grads = torch.autograd.grad(loss, values, create_graph=track_response)
                rate = lr * .5 * (1 + math.cos(math.pi*(round_no*formal_steps+step) /
                                               max(total_rounds*formal_steps-1, 1)))
                if optimizer == 'muon':
                    new_b, muon_states[k] = muon_step(values[:-1], muon_states[k], grads[:-1], rate)
                    new_h, head_state = adamw_step([values[-1]], head_state, [grads[-1]], [.001])
                    values = new_b + new_h
                else:
                    values, adam_state = adamw_step(values, adam_state, grads,
                                                    [rate]*n_b + [.001])
                if not track_response:
                    values = [v.detach().requires_grad_(True) for v in values]
                    muon_states[k] = [m.detach() for m in muon_states[k]]
                counts['trial_visits'] += len(y)
                counts['trial_steps'] += 1
            endpoints.append(values[:-1]); heads.append(values[-1])
            features = []
            for x, y in gram_batches[k]:
                if len(x) != len(y) or not len(y):
                    raise ValueError('invalid sampled Gram batch')
                with torch.set_grad_enabled(track_response):
                    features.append(F.normalize(call_model(x, values, base_values, features=True), dim=1).double())
                counts['gram_visits'] += len(y)
            if not features:
                raise ValueError('empty sampled Gram')
            samples = sum(len(z) for z in features)
            grams.append(sum(z.T @ z for z in features) * (formal_sizes[k] / samples))
        with torch.set_grad_enabled(track_response):
            global_b = [sum(ends[j] * (size/total_size) for ends, size in zip(endpoints, formal_sizes))
                        for j in range(n_b)]
            if track_response:
                global_head, certificate = certified_head_surrogate(heads, grams, formal_sizes)
            else:
                global_head, _, certificate = aggregate_heads(heads, grams, formal_sizes, True)
            if server_operator is not None:
                global_b = server_operator(global_b, bases)
        counts['head_fusions'] += 1
        certificates.append(certificate)
        if not track_response:
            global_b = [b.detach().requires_grad_(True) for b in global_b]
            global_head = global_head.detach().requires_grad_(True)
    active = [k for k, batches in enumerate(query) if batches]
    denom = sum(formal_sizes[k] for k in active)
    if not denom:
        raise ValueError('no query coverage')
    covered_classes = sorted({int(label) for k in active for _, y in query[k]
                              for label in y.detach().cpu().tolist()})
    all_loss = current_loss = 0
    with torch.set_grad_enabled(track_response):
        for k in active:
            count = sum(len(y) for _, y in query[k])
            if count <= 0:
                raise ValueError('empty query')
            for x, y in query[k]:
                lo = task * classes_per_task
                if len(x) != len(y) or bool(((y < lo) | (y >= lo+classes_per_task)).any()):
                    raise ValueError('query includes non-current label')
                logits = call_model(x, global_b + [global_head], base_values)
                weight = formal_sizes[k] / (denom * count)
                all_loss = all_loss + weight * F.cross_entropy(logits, y, reduction='sum')
                current_loss = current_loss + weight * F.cross_entropy(
                    logits[:, lo:lo+classes_per_task], y-lo, reduction='sum')
                counts['query_visits'] += len(y)
    return {'loss': all_loss, 'current': current_loss, 'counts': counts,
            'query_client_coverage': len(active), 'query_weight_coverage': denom/total_size,
            'query_classes': covered_classes, 'active_client_indices': active_clients,
            'head_certificates': certificates, 'B': global_b, 'head': global_head,
            'muon_states': muon_states if optimizer == 'muon' else None}
