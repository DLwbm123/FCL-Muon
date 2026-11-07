"""Shared identity validation and factory. Never downloads assets."""
import copy
import hashlib
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
from fcl.checkpoint import restore_rng


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def write(path, data):
    from fcl.checkpoint import atomic_save
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(data, f, allow_nan=False, indent=2); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def setup():
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available(): torch.cuda.set_per_process_memory_fraction(.65, 0)


def roots():
    return [Path(os.environ[k]).resolve() for k in ('PROJECT_ROOT', 'SOURCE_ROOT', 'PRIOR_ROOT')]


def validate():
    import timm, kornia
    root, source, prior = roots()
    f = json.loads((root/'reports/FROZEN_PROTOCOL.json').read_text())
    job = json.loads((root/'reports/job.json').read_text())
    if any(f[k] != job[k] for k in ('started', 'deadline')):
        raise ValueError('original clock identity mismatch')
    for name, sha in f['source_sha256'].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or digest(path) != sha:
            raise ValueError('frozen source mismatch: '+name)
    if {'torch': str(torch.__version__), 'timm': str(timm.__version__), 'kornia': str(kornia.__version__),
            'cuda': torch.version.cuda} != f['environment']:
        raise ValueError('frozen environment changed')
    for name, sha in f['bootstrap_sha256'].items():
        if digest(root.parent/name) != sha: raise ValueError('bootstrap mismatch')
    for seed, sha in f['manifest_sha256'].items():
        if digest(root/f'manifests/seed{seed}.json') != sha: raise ValueError('manifest mismatch')
    # The immutable weights and prefix were read/hashed once at registration.
    # Verify their low-cost stat identity at each worker; no repeated giant hashes.
    for name, stat in f['asset_stat'].items():
        s = Path(name).stat()
        if [s.st_size, s.st_mtime_ns, s.st_ino] != stat: raise ValueError('source asset changed')
    return f, job


def build(seed, arm, horizon, clock, *, synthetic=None):
    import timm
    from safetensors.torch import load_file
    from fcl.model import MuLoRAFL
    from fcl.data import loader, transforms
    from .pilot import BilateralPilot
    root, source, prior = roots()
    frozen = json.loads((root/'reports/FROZEN_PROTOCOL.json').read_text())
    cfg = copy.deepcopy(frozen['base_config'])
    cfg.update(seed=seed, tasks=10, total_tasks=10, stop_after_task=horizon,
               manifest=f'manifests/seed{seed}.json', output=f'runs/s{seed}_{arm}_T{horizon}',
               selection_mode='selected', retain_round_checkpoints=False,
               retain_task_checkpoints=False, retain_first_round_endpoints=False,
               common_intervention=seed == 0 and arm == 'E0',
               bilateral_storage_stop=18500000000,
               bilateral={'arm': arm, 'gamma': .1, 'epsilon': .01, 'rank_cap': 64, 'spectral_power': 2})
    cfg['route'] = dict(frozen['route'], mode='selected', optimizer='adamw')
    if synthetic is not None:
        cfg['output'] = 'runs/synthetic_initial'
    weights = frozen['weights']
    manifest = json.loads((root/cfg['manifest']).read_text())
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    backbone = timm.create_model(cfg['backbone'], pretrained=False, num_classes=0)
    backbone.load_state_dict({k: v for k, v in load_file(weights['path']).items() if not k.startswith('head.')}, strict=True)
    model = MuLoRAFL(backbone, 100, cfg['rank'], cfg['cosine_scale']).cuda().eval()
    train_tf, test_tf = transforms()
    if synthetic is None:
        from torchvision.datasets import CIFAR100
        raw = [CIFAR100(source/'data', train=t, download=False) for t in (True, False)]
        if any(list(r.targets) != manifest['labels'][s] for r, s in zip(raw, ('train', 'test'))):
            raise ValueError('CIFAR labels differ from frozen manifest')
        accesses = {'train_loaded': 0, 'test_loaded': 0}
        def counted(raw_data, ids, batch, shuffle=False):
            role = 'train_loaded' if raw_data is raw[0] else 'test_loaded'
            def batches():
                for x, y in loader(raw_data, ids, batch, shuffle):
                    accesses[role] += len(y)
                    yield x, y
            # Trainer needs len(dl) for the original local learning-rate schedule.
            class Counted:
                def __len__(self): return (len(ids)+batch-1)//batch
                def __iter__(self): return batches()
            return Counted()
        used_loader = counted
    else:
        raw, manifest, used_loader = synthetic
        accesses = {'dataset_images': 0}
    identity = {'seed': seed, 'optimizer': 'adamw', 'weights_sha256': weights['sha256'],
                'manifest_sha256': frozen['manifest_sha256'][str(seed)], 'source_sha256': frozen['code_identity']}
    pilot = BilateralPilot(model, raw, manifest, used_loader, train_tf.cuda(), test_tf.cuda(), cfg,
                           root/cfg['output'], identity, clock)
    pilot.accesses = accesses
    return pilot


def import_t1(pilot, prefix, provenance):
    # Explicit new-protocol fork; preserve every tensor, state, RNG and the
    # original source record while changing only registered config/clock fields.
    expected = {'clients': 10, 'rounds': 5, 'epochs': 5, 'batch': 16, 'rank': 32,
                'lr': .0002, 'cpt': 10, 'backbone': 'vit_base_patch16_224.augreg_in21k'}
    if any(prefix['config'].get(k) != v for k, v in expected.items()):
        raise ValueError('incompatible source T1 config')
    s = prefix['state']
    if ((s['task'], s['round'], s['phase']) != (1, 0, 'task_start') or s['merges'] != [0]
            or s['history'].committed != {0} or prefix['head_count'] != 1
            or len(s['training_records']) != 50 or sum(v['visited'] for v in s['training_records']) != 125000
            or prefix['config']['route']['optimizer'] != 'adamw'
            or prefix['config']['route']['mode'] != 'prefix'):
        raise ValueError('invalid source T1 boundary')
    if prefix['schema'] != 'cosine-head-regmean-pilot-1': raise ValueError('prefix schema')
    # tasks=10 preserves the manifest's ten tasks. The inherited learning-rate
    # schedule uses rounds/local steps only; no task-horizon input occurs.
    target = {k: v for k, v in pilot.cfg['route'].items() if k not in ('mode', 'optimizer')}
    source_route = {k: v for k, v in prefix['config']['route'].items() if k not in ('mode', 'optimizer')}
    if source_route != target: raise ValueError('prefix selector settings changed')
    pilot.model.add_head(pilot.cfg['cpt'])
    pilot.model.load_state_dict(prefix['model'], strict=True)
    pilot.s = copy.deepcopy(s)
    for name, a in pilot.model.adapters.items():
        a.merged = prefix['merged_flags'][name]
        if not a.merged or torch.count_nonzero(a.Bq) or torch.count_nonzero(a.Bv):
            raise ValueError('prefix not merged/zeroed')
    pilot.s['prefix_provenance'] = provenance
    # A generated E0 T1 is exactly identity; method state must not carry E0 to E3.
    pilot.s.pop('bilateral', None)
    restore_rng(prefix['rng'])
    if len(prefix['rng']['cuda']) != 1 or not torch.equal(torch.cuda.get_rng_state(), prefix['rng']['cuda'][0]):
        raise ValueError('source CUDA RNG restore mismatch')
    pilot.save()
