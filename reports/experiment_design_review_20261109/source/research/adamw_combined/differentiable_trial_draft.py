"""Unregistered functional multi-step AdamW client trial; no I/O or mutation.

The caller supplies already prepared current-task batches and paired RNG state.
The resulting graph can be expensive; this has not been validated on real ViT.
"""
import math
import torch
from torch.nn import functional as F
from torch.func import functional_call


def differentiable_trial(model, bases, batches, epochs, task, cpt, lr, rounds,
                         formal_batches, schedule_epochs, progress=None, track_response=True):
    if model.training or not batches or epochs < 1 or formal_batches < 1 or rounds < 1:
        raise ValueError('invalid trial state')
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    pnames = {id(p): n for n, p in params.items()}
    bnames = {id(b): n for n, b in buffers.items()}
    if set(bases) != set(model.adapters):
        raise ValueError('basis mismatch')
    virtual_buffers = dict(buffers)
    for name, adapter in model.adapters.items():
        if bases[name].shape != adapter.A.shape:
            raise ValueError('basis shape mismatch')
        virtual_buffers[bnames[id(adapter.A)]] = bases[name]
    bp = model.b_parameters()
    if any(torch.count_nonzero(p) for p in bp):
        raise ValueError('task anchor must have zero B')
    names = [pnames[id(p)] for p in bp] + [pnames[id(model.rows[-1])]]
    virtual = dict(params)
    variables = [params[n].detach().clone().requires_grad_(True) for n in names]
    moment = [torch.zeros_like(v) for v in variables]
    variance = [torch.zeros_like(v) for v in variables]
    formal_steps = schedule_epochs * formal_batches
    seen = steps = 0
    for _ in range(epochs):
        for x, y in batches:
            if steps >= formal_steps or len(y) == 0 or len(x) != len(y):
                raise ValueError('invalid batch schedule')
            offset = task * cpt
            if bool(((y < offset) | (y >= offset+cpt)).any()):
                raise ValueError('non-current task label')
            virtual.update(zip(names, variables))
            logits = functional_call(model, (virtual, virtual_buffers), (x,))
            loss = F.cross_entropy(logits[:, offset:offset+cpt], y-offset)
            grads = torch.autograd.grad(loss, variables, create_graph=track_response)
            if progress is not None:
                progress('trial_visits', len(y))
            steps += 1
            adapter_lr = lr * .5 * (1 + math.cos(math.pi*(steps-1)/max(rounds*formal_steps-1, 1)))
            new_vars = []
            for j, (v, g) in enumerate(zip(variables, grads)):
                moment[j] = .9*moment[j] + .1*g
                variance[j] = .999*variance[j] + .001*g.square()
                mhat = moment[j] / (1-.9**steps)
                vhat = variance[j] / (1-.999**steps)
                rate = .001 if j == len(variables)-1 else adapter_lr
                # sqrt has an undefined derivative at exactly zero. Preserve its
                # forward value there and define a zero derivative at that point.
                root = torch.where(vhat > 0, vhat.clamp_min(torch.finfo(vhat.dtype).tiny).sqrt(), torch.zeros_like(vhat))
                updated = v - rate*mhat/(root+1e-8)
                new_vars.append(updated if track_response else updated.detach().requires_grad_(True))
            variables = new_vars
            seen += len(y)
    return {'B': variables[:-1], 'head': variables[-1], 'steps': steps, 'train_visits': seen}
