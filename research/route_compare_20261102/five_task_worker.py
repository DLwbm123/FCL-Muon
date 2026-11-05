"""Unregistered five-task A0/A1 continuation; requires a separate supervisor."""
import copy
import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    root = Path(os.environ['PROJECT_ROOT']).resolve()
    source = Path(os.environ['SOURCE_ROOT']).resolve()
    prior = Path(os.environ['PRIOR_ROOT']).resolve()
    reports = root / 'reports'
    frozen = json.loads((reports / 'FROZEN_FIVE_TASK.json').read_text())
    seed = int(os.environ['STUDY_SEED'])
    if (frozen.get('status') != 'frozen_before_new_test' or frozen.get('tasks') != 5 or
            frozen.get('seeds') != [0, 1, 2] or seed not in frozen['seeds'] or
            frozen.get('arms') != ['rrs', 'selected']):
        raise ValueError('missing five-task registration')
    job = json.loads((reports / 'job.json').read_text())
    clock = {k: job[k] for k in ('started', 'deadline')}
    if (clock != {k: frozen[k] for k in clock} or
            any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in clock.values()) or
            not 0 < clock['deadline'] - clock['started'] <= 86400 or time.time() >= clock['deadline']):
        raise ValueError('invalid or expired original clock')
    if digest(root / 'research/route_compare_20261102/FIVE_TASK_PROTOCOL_ZH.md') != frozen['protocol_sha256']:
        raise ValueError('protocol changed')
    oldfrozen = json.loads((prior / 'reports/FROZEN.json').read_text())
    core = ('research/route_compare_20261102/pilot.py', 'research/route_compare_20261102/selection.py',
            'research/route_compare_20261102/response.py', 'research/route_compare_20261102/frame.py')
    for name in core:
        if frozen['source_sha256'].get(name) != oldfrozen['source_sha256'][name]:
            raise ValueError('not the original A1 source')
    for name, expected in frozen['source_sha256'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or digest(path) != expected:
            raise ValueError('frozen source mismatch')
    if not {'research/route_compare_20261102/five_task_worker.py',
            'research/route_compare_20261102/long_sequence_prefix.py'}.issubset(frozen['source_sha256']):
        raise ValueError('worker not in source manifest')
    if frozen['route'] != oldfrozen['route']:
        raise ValueError('original route parameters changed')
    manifest_path = source / 'manifests/seq-cifar100_n10_a0.05_s0.json' if seed == 0 else prior / f'manifests/seed{seed}.json'
    if digest(manifest_path) != frozen['manifest_sha256'][str(seed)]:
        raise ValueError('manifest mismatch')
    manifest = json.loads(manifest_path.read_text())
    weights = json.loads((source / 'weights.json').read_text())
    if weights['sha256'] != frozen['weight_sha256'] or weights['sha256'] != oldfrozen['weight_sha256']:
        raise ValueError('pretrained identity mismatch')
    oldjob = json.loads((prior / 'reports/job.json').read_text())
    oldclock = {k: oldjob[k] for k in ('started', 'deadline')}
    path = prior / ('runs/smoke_adamw/latest.pt' if seed == 0 else f'runs/seed{seed}_adamw_prefix/latest.pt')
    if digest(path) != frozen['prefix_sha256'][str(seed)]:
        raise ValueError('source checkpoint mismatch')

    import torch
    import timm
    import kornia
    from safetensors.torch import load_file
    from torchvision.datasets import CIFAR100
    from fcl.data import loader, transforms
    from fcl.model import MuLoRAFL
    from research.aggregation_utility.lifecycle import import_prefix
    from research.route_compare_20261102.long_sequence_prefix import derive_prefix
    from research.route_compare_20261102.pilot import RoutePilot
    from research.route_compare_20261102.smoke import write

    environment = {'torch': str(torch.__version__), 'timm': str(timm.__version__),
                   'kornia': str(kornia.__version__), 'cuda': str(torch.version.cuda)}
    if environment != frozen['environment']:
        raise ValueError('environment mismatch')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    cfg0, identity = checkpoint['config'], checkpoint['identity']
    if (identity['seed'] != seed or identity['weights_sha256'] != weights['sha256'] or
            manifest['identity'] != {k: cfg0[k] for k in ('dataset', 'seed', 'alpha', 'clients')} or
            {k: v for k, v in cfg0['route'].items() if k not in ('mode', 'optimizer')} != frozen['route']):
        raise ValueError('prefix identity mismatch')
    raw = [CIFAR100(source / 'data', train=t, download=False) for t in (True, False)]
    if [list(x.targets) for x in raw] != [manifest['labels']['train'], manifest['labels']['test']]:
        raise ValueError('dataset labels mismatch')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.cuda.set_per_process_memory_fraction(.45, 0)
    train_tf, test_tf = transforms()
    train_tf, test_tf = train_tf.cuda(), test_tf.cuda()
    results = {}
    for mode in ('rrs', 'selected'):
        cfg = copy.deepcopy(cfg0)
        cfg.update(tasks=5, selection_mode=mode, output=f'runs/seed{seed}_{mode}')
        cfg['route']['mode'] = mode
        backbone = timm.create_model(cfg['backbone'], pretrained=False, num_classes=0)
        backbone.load_state_dict({k: v for k, v in load_file(weights['path']).items() if not k.startswith('head.')}, strict=True)
        model = MuLoRAFL(backbone, 100, cfg['rank'], cfg['cosine_scale']).cuda()
        pilot = RoutePilot(model, raw, manifest, loader, train_tf, test_tf, cfg, root / cfg['output'], identity, clock)
        fork = derive_prefix(checkpoint, cfg, identity, oldclock, clock)
        import_prefix(pilot, fork)
        if len(checkpoint['rng']['cuda']) != 1 or not torch.equal(torch.cuda.get_rng_state(), checkpoint['rng']['cuda'][0]):
            raise ValueError('prefix CUDA RNG mismatch')
        del fork
        while pilot.step():
            write(reports / f'progress_{seed}.json', {'time': time.time(), 'seed': seed, 'arm': mode,
                  'task': pilot.s['task'], 'round': pilot.s['round'], 'phase': pilot.s['phase'],
                  'client_rounds': len(pilot.s['training_records'])})
        if len(pilot.s['training_records']) != 250 or pilot.s['history'].committed != set(range(5)):
            raise ValueError('incomplete five-task lifecycle')
        stages = copy.deepcopy(pilot.s['stages'])
        results[mode] = {'stages': stages, 'route_records': copy.deepcopy(pilot.s.get('route_records', [])),
                         'communication': copy.deepcopy(pilot.s['communication']), 'client_rounds': 250}
        pilot.restore()
        pilot.s['task'] = 4
        if pilot.evaluate()['confusion'] != stages[-1]['confusion']:
            raise ValueError('final reload mismatch')
        results[mode]['reload_confusion_equal'] = True
        write(reports / f'result_seed{seed}_private.json', results)
        del pilot, model, backbone
        gc.collect()
        torch.cuda.empty_cache()
    write(reports / f'seed{seed}_done.json', {'status': 'completed', 'arms': list(results)})


if __name__ == '__main__':
    main()
