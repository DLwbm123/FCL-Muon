"""Current-task-only gradient probe that restores model observation state."""
import torch
from torch.nn import functional as F
from fcl.checkpoint import rng_state, restore_rng


def probe(pilot, client, count=32):
    if (pilot.s['task'], pilot.s['round'], pilot.s['phase']) != (2, 0, 'round_start'):
        raise ValueError('probe requires the shared T3 start')
    if count <= 0:
        raise ValueError('probe count must be positive')
    ids = sorted(pilot.ids('train', 2, client))[:count]
    if not ids:
        raise ValueError('empty current-task training probe')
    model = pilot.model
    parameters = list(model.parameters())
    gradients_before = [None if p.grad is None else p.grad.detach().clone() for p in parameters]
    modes = {m: m.training for m in model.modules()}
    buffers = {name: b.detach().clone() for name, b in model.named_buffers()}
    state_rng = rng_state()
    b_parameters = model.b_parameters()
    values = [torch.zeros_like(b) for b in b_parameters]
    samples = 0
    try:
        model.eval()
        for x, y in pilot.loader(pilot.raw[0], ids, pilot.cfg['batch']):
            pilot.budget(); pilot.check_labels(y)
            y = y.to(pilot.device)
            logits = model(pilot.test_tf(x.to(pilot.device)))
            c = pilot.cfg['cpt']
            loss = F.cross_entropy(logits[:, 2 * c:3 * c], y - 2 * c, reduction='sum')
            if not torch.isfinite(loss):
                raise ValueError('nonfinite current-task probe loss')
            grad = torch.autograd.grad(loss, b_parameters)
            for total, delta in zip(values, grad):
                total.add_(delta.detach())
            samples += len(y)
        if samples != len(ids):
            raise ValueError('probe loader count mismatch')
        values = [(v / samples).cpu() for v in values]
        if any(not torch.isfinite(v).all() for v in values):
            raise ValueError('nonfinite current-task probe gradient')
        return values, {'samples': samples, 'student_forward_images': samples,
                        'student_backward_images': samples, 'teacher_forward_images': 0}
    finally:
        with torch.no_grad():
            for name, buffer in model.named_buffers():
                buffer.copy_(buffers[name])
        for parameter, saved in zip(parameters, gradients_before):
            parameter.grad = saved
        for module, mode in modes.items():
            module.training = mode
        restore_rng(state_rng)
