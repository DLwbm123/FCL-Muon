"""Two weighted current-task objectives at one server anchor, without an optimizer step."""
import torch
from torch.nn import functional as F
from fcl.checkpoint import restore_rng, rng_state


def gradients(pilot, client_splits, part):
    if part not in (0, 1):raise ValueError('invalid split')
    groups = pilot.manifest['partitions']['train'][2]
    if len(client_splits) != len(groups):raise ValueError('client count mismatch')
    weights = [len(g) for g in groups]
    denominator = sum(w for w, pair in zip(weights, client_splits) if pair[0] and pair[1])
    if not denominator:raise ValueError('no eligible clients')
    model = pilot.model;params = model.b_parameters();all_params = list(model.parameters())
    old_grads = [None if v.grad is None else v.grad.detach().clone() for v in all_params]
    modes = {m: m.training for m in model.modules()}
    buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
    random_state = rng_state()
    acc_all = [torch.zeros_like(v, dtype=torch.float64) for v in params]
    acc_cur = [torch.zeros_like(v, dtype=torch.float64) for v in params]
    samples = 0
    try:
        model.eval()
        for k, pair in enumerate(client_splits):
            ids = pair[part];count = 0
            for x, y in pilot.loader(pilot.raw[0], ids, pilot.cfg['batch']):
                pilot.budget();pilot.check_labels(y);y = y.to(pilot.device)
                logits = model(pilot.test_tf(x.to(pilot.device))).double();c = pilot.cfg['cpt']
                loss_all = F.cross_entropy(logits, y, reduction='sum')
                loss_cur = F.cross_entropy(logits[:, 2*c:3*c], y-2*c, reduction='sum')
                if not torch.isfinite(loss_all) or not torch.isfinite(loss_cur):
                    raise ValueError('nonfinite probe objective')
                grad_all = torch.autograd.grad(loss_all, params, retain_graph=True)
                grad_cur = torch.autograd.grad(loss_cur, params)
                factor = weights[k] / (denominator * len(ids))
                with torch.no_grad():
                    for total, g in zip(acc_all, grad_all):total.add_(g.detach().double(), alpha=factor)
                    for total, g in zip(acc_cur, grad_cur):total.add_(g.detach().double(), alpha=factor)
                count += len(y);samples += len(y)
            if count != len(ids):raise ValueError('probe sample count mismatch')
        if any(not torch.isfinite(v).all() for v in (*acc_all, *acc_cur)):
            raise ValueError('nonfinite probe gradients')
        return [v.cpu() for v in acc_all], [v.cpu() for v in acc_cur], {
            'samples': samples, 'weight_coverage': denominator / sum(weights),
            'student_forward_images': samples, 'student_backward_images': 2*samples}
    finally:
        with torch.no_grad():
            for name, value in model.named_buffers():value.copy_(buffers[name])
        for parameter, value in zip(all_params, old_grads):parameter.grad = value
        for module, mode in modes.items():module.training = mode
        restore_rng(random_state)
