"""Frozen synthetic real-ViT/CUDA preflight; zero dataset images."""
import copy
import gc
import json
import time
import torch
from fcl.checkpoint import rng_state, restore_rng
from research.response_bilateral.runtime import setup, roots, validate, build, import_t1, write
from research.route_compare_20261102.response import federated_response


def main():
    setup()
    frozen, job = validate()
    root, source, prior = roots()
    clock = {k: job[k] for k in ('started', 'deadline')}
    g = torch.Generator().manual_seed(426793)
    # Synthetic inputs only; a real source T1 is restored separately.
    raw = [(torch.randn(3, 224, 224, generator=g), torch.tensor(10+i % 10)) for i in range(20)]
    manifest = {'partitions': {'train': [[], [list(range(10)), list(range(10, 20))]],
                              'test': [[], [list(range(10)), list(range(10, 20))]]},
                'labels': {'train': [int(y) for x, y in raw], 'test': [int(y) for x, y in raw]}}
    def loader(raw_, ids, batch, shuffle=False):
        order = torch.randperm(len(ids)).tolist() if shuffle else list(range(len(ids)))
        ids = [ids[i] for i in order]
        return [tuple(torch.stack(v) for v in zip(*(raw_[j] for j in ids[i:i+batch])))
                for i in range(0, len(ids), batch)]
    p = build(0, 'E3', 3, clock, synthetic=([raw, raw], manifest, loader))
    prefix = torch.load(prior/'runs/smoke_adamw/latest.pt', map_location='cpu', weights_only=False)
    import_t1(p, prefix, {'synthetic_source_fork': True})
    # Restore the real T1 first, then change only this synthetic exercise's size.
    p.cfg.update(clients=2, epochs=2, batch=4, bilateral_probe=False, retain_round_checkpoints=True)
    p.cfg['route']['mode'] = 'rrs'
    p.out = root/'runs/synthetic'; p.out.mkdir(exist_ok=False)
    p.initialize_task()
    p.save()
    bases = {k: a.A.detach().clone().requires_grad_(True) for k, a in p.model.adapters.items()}
    x = torch.randn(4, 3, 224, 224, generator=g, device='cpu').cuda()
    y = torch.tensor([10, 11, 12, 13], device='cuda')
    support = [[[(x, y)]*4 for _ in range(10)] for _ in range(2)]
    grams = [[(x, y)] for _ in range(10)]
    snapshot = {k: v.detach().cpu().clone() for k, v in p.model.state_dict().items()}
    rng = rng_state()
    records = []
    for arm in ('E0', 'E1', 'E2', 'E3'):
        p.cfg['bilateral']['arm'] = arm
        restore_rng(rng)
        start = time.monotonic()
        args = (p.model, bases, support, grams, grams, [32]*10, [2]*10,
                'adamw', .0002, 5, 5, 1, 10)
        # Full production short-response shape, including the solve in both
        # short rounds and its outer derivatives through candidate A.
        with torch.autograd.graph.save_on_cpu(pin_memory=False):
            tracked = federated_response(*args, use_checkpoint=True, track_response=True,
                                          server_operator=p.response_server_operator)
            gradients = torch.autograd.grad(tracked['loss'], list(bases.values()))
        if any(not torch.isfinite(v).all() for v in gradients): raise ArithmeticError('nonfinite outer gradient')
        restore_rng(rng)
        direct = federated_response(*args, use_checkpoint=False, track_response=False,
                                    server_operator=p.response_server_operator)
        error = max(float((a.detach()-b.detach()).abs().max()) for a, b in zip(tracked['B'], direct['B']))
        if error > 2e-6 or abs(float(tracked['loss'].detach()-direct['loss'])) > 2e-6:
            raise ArithmeticError('tracked/direct endpoint mismatch')
        if arm == 'E0':
            restore_rng(rng)
            original = federated_response(*args, use_checkpoint=False, track_response=False)
            if (any(not torch.equal(x.detach(), y.detach()) for x, y in zip(original['B'], direct['B']))
                    or not torch.equal(original['head'], direct['head'])
                    or not torch.equal(original['loss'], direct['loss'])):
                raise ArithmeticError('gamma=0 real ViT baseline regression')
            del original
        if any(not torch.equal(v, p.model.state_dict()[k].cpu()) for k, v in snapshot.items()):
            raise ValueError('source model changed during response')
        torch.cuda.synchronize()
        records.append({'arm': arm, 'counts': direct['counts'], 'B_max_abs_track_direct': error,
                        'seconds': time.monotonic()-start})
        del tracked, direct, gradients
        gc.collect(); torch.cuda.empty_cache()
    # Exercise the actual formal server placement, round checkpoint, and full
    # restore on ViT; use E3 and the unmodified production trainer.
    p.cfg['bilateral']['arm'] = 'E3'
    restore_rng(rng)
    while p.s['round'] < 1: p.step()
    state = copy.deepcopy(p.model.state_dict())
    endpoint = torch.load(p.out/'round_t1_r1.pt', map_location='cpu', weights_only=False)
    for b, v in zip(p.model.b_parameters(), endpoint['B']):
        if not torch.equal(b.cpu(), v): raise ValueError('round checkpoint endpoint mismatch')
    p.restore()
    if any(not torch.equal(v, p.model.state_dict()[k]) for k, v in state.items()):
        raise ValueError('real ViT recovery mismatch')
    if p.s['bilateral']['applied'] != [0]: raise ValueError('wrong formal correction count')
    write(root/'reports/preflight_result.json', {'status': 'passed', 'real_dataset_images': 0,
        'source_T1_cuda_rng_restored': True, 'functional_arms': records,
        'ViT_formal_round_and_complete_checkpoint_restore': True,
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
        'head_derivative_scope': 'existing implicit surrogate, not a certified exact full-selector gradient'})
    print('real ViT synthetic preflight passed; dataset images=0', flush=True)


if __name__ == '__main__': main()
