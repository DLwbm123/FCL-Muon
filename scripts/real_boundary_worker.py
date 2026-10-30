"""Draft independent Muon boundary diagnostic; not a registered or launched run."""
import copy
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import timm
import torch
from fcl.checkpoint import rng_state
from fcl.data import loader, transforms
from fcl.math import mulora_ssb
from fcl.model import MuLoRAFL
from research.head_regmean.pilot import Pilot
from research.continual_muon.current_cifar import CurrentCIFAR
from research.continual_muon.current_task_guard import CurrentTaskGuard
from research.continual_muon.import_boundary import import_t1
from research.continual_muon.real_boundary_geometry import measure, last_momentum_proxy


def write(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def same_rng(a, b):
    return (a['python'] == b['python'] and a['numpy'][0] == b['numpy'][0] and
            np.array_equal(a['numpy'][1], b['numpy'][1]) and a['numpy'][2:] == b['numpy'][2:] and
            torch.equal(a['torch'], b['torch']) and len(a['cuda']) == len(b['cuda']) and
            all(torch.equal(x, y) for x, y in zip(a['cuda'], b['cuda'])))


def main():
    root = Path(os.environ['PROJECT_ROOT'])
    source = Path(os.environ['SOURCE_ROOT'])
    clock = json.loads((root / 'reports/job.json').read_text())
    clock = {k: clock[k] for k in ('started', 'deadline')}
    if not clock['started'] <= time.time() < clock['deadline']:
        raise TimeoutError('independent diagnostic clock exhausted')
    with (root / 'reports/WORKER_CLAIM.json').open('x') as claim:
        json.dump({'started': time.time(), 'clock': clock}, claim)
    cfg = json.loads((source / 'configs/aggregation_utility_three_task_seed0.json').read_text())
    cfg['output'] = 'runs/diagnostic'
    for key, value in {'seed': 0, 'dataset': 'seq-cifar100', 'alpha': .05, 'clients': 10,
                       'rounds': 5, 'epochs': 5, 'batch': 16, 'rank': 32,
                       'lr': .0002, 'aggregation': 'regmean', 'selection_mode': 'prefix'}.items():
        if cfg.get(key) != value:
            raise ValueError('unexpected source config: ' + key)
    manifest = json.loads((source / cfg['manifest']).read_text())
    if manifest['identity'] != {k: cfg[k] for k in ('dataset', 'seed', 'alpha', 'clients')}:
        raise ValueError('manifest identity mismatch')
    groups = manifest['partitions']['train']
    indices = {t: [i for group in groups[t] for i in group] for t in (1, 2)}
    if any(len(indices[t]) != 5000 or len(set(indices[t])) != 5000 for t in (1, 2)):
        raise ValueError('unexpected training partition size or duplicates')
    if any(not t * cfg['cpt'] <= manifest['labels']['train'][i] < (t + 1) * cfg['cpt']
           for t in (1, 2) for i in indices[t]):
        raise ValueError('unexpected task label')
    weights = json.loads((source / 'weights.json').read_text())
    identity = {'manifest': cfg['manifest'], 'weight_identity': weights['sha256'],
                'source_revision': cfg['source_revision'], 'purpose': 'three-task aggregation-utility selection'}
    # Exclude test metadata from all trainer paths, in addition to never opening the test file.
    manifest = {'identity': manifest['identity'], 'partitions': {'train': groups},
                'labels': {'train': manifest['labels']['train']}}
    raw = CurrentCIFAR(source / 'data/cifar-100-python/train', manifest['labels']['train'], indices)
    guard = CurrentTaskGuard(loader, raw, indices)
    try:
        torch.set_num_threads(4); torch.set_float32_matmul_precision('highest')
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        model = MuLoRAFL(timm.create_model(cfg['backbone'], pretrained=False, num_classes=0),
                         100, cfg['rank'], cfg['cosine_scale']).cuda()
        train_tf, stat_tf = transforms()
        pilot = Pilot(model, [raw, None], manifest, guard, train_tf.cuda(), stat_tf.cuda(),
                      cfg, root / cfg['output'], identity, clock)
        checkpoint = torch.load(source / 'runs/prefix/latest.pt', map_location='cpu', weights_only=False)
        receipt = import_t1(pilot, checkpoint, identity)
        if len(checkpoint['rng']['cuda']) != 1 or not torch.equal(torch.cuda.get_rng_state(), checkpoint['rng']['cuda'][0]):
            raise ValueError('source CUDA RNG not restored')
        del checkpoint
        preflight_rng = rng_state()
        i = torch.arange(768, dtype=torch.float32)[:, None] + 1
        j = torch.arange(32, dtype=torch.float32)[None, :] + 1
        probe = torch.linalg.qr(torch.sin(i * j * .013), mode='reduced').Q
        cpu_probe = mulora_ssb(probe)
        gpu_probe = mulora_ssb(probe.cuda()).cpu()
        oracle_error = float(torch.linalg.vector_norm(cpu_probe - gpu_probe))
        if (not torch.isfinite(gpu_probe).all() or oracle_error > 1e-4 or
                not same_rng(preflight_rng, rng_state())):
            raise ValueError('GPU SSB preflight or RNG isolation failed')
        write(root / 'reports/REAL_PREFLIGHT.json', {'passed': True, 'cpu_gpu_ssb_l2': oracle_error,
              'source_cuda_rng_restored': True, 'rng_unchanged': True, 'student_forwards': 0,
              'student_backwards': 0, 'image_accesses': 0, 'test_accesses': 0})
        old_visits = sum(r['visited'] for r in pilot.s['training_records'])
        old_stats = len(pilot.s['stats_records'])
        old_communication = copy.deepcopy(pilot.s['communication'])
        while pilot.s['phase'] != 'ready_evaluation':
            if pilot.s['task'] != 1 or pilot.s['round'] > 5:
                raise ValueError('unexpected T2 lifecycle')
            pilot.step()
            write(root / 'reports/PROGRESS.json', {'time': time.time(), 'task': 2,
                  'round': pilot.s['round'], 'phase': pilot.s['phase'], 'reads': raw.reads})
        if pilot.s['round'] != 5 or sum(r['visited'] for r in pilot.s['training_records']) - old_visits != 125000:
            raise ValueError('unexpected training visits')
        before = rng_state()
        momentum = {k: [v.detach().cpu().clone() for v in pilot.s['client_states'][k]['m']]
                    for k in range(cfg['clients'])}
        old_bases = [a.A.detach().cpu().clone() for a in model.adapters.values()]
        if not same_rng(before, rng_state()):
            raise ValueError('observation altered RNG')
        if (any(not torch.equal(a.A.detach().cpu(), b) for a, b in zip(model.adapters.values(), old_bases)) or
                any(not torch.equal(a, b) for k in momentum
                    for a, b in zip(pilot.s['client_states'][k]['m'], momentum[k]))):
            raise ValueError('observation altered basis or momentum')
        if any(len(v) != 2 * len(old_bases) for v in momentum.values()):
            raise ValueError('momentum/basis ordering mismatch')
        # Original evaluate is intentionally never called. Merge and commit only in this new run.
        pilot.s['phase'] = 'evaluated'
        for expected in ('merged', 'committed', 'task_start'):
            pilot.step()
            if pilot.s['phase'] != expected:
                raise ValueError('unexpected diagnostic-only transition')
        if pilot.s['task'] != 2 or pilot.s['client_states']:
            raise ValueError('boundary state not cleared')
        raw.enter_t3(); guard.enter_t3()
        pilot.step()
        if pilot.s['task'] != 2 or pilot.s['phase'] != 'round_start':
            raise ValueError('new task basis not ready')
        new_bases = [a.A.detach().cpu().clone() for a in model.adapters.values()]
        ssb = lambda a: mulora_ssb(torch.from_numpy(a)).numpy()
        sizes = np.asarray([len(g) for g in groups[1]], dtype=np.float64)
        if (sizes <= 0).any():
            raise ValueError('empty client; frozen diagnostic requires 10 nonempty clients')
        weights_client = sizes / sizes.sum()
        rows = []
        for k in range(cfg['clients']):
            pilot.budget()
            row = [measure(momentum[k][j].numpy(), old_bases[j // 2].numpy(),
                           new_bases[j // 2].numpy(), ssb) for j in range(2 * len(old_bases))]
            source2 = sum(v['source_norm_squared'] for v in row)
            projected2 = sum(v['projected_norm_squared'] for v in row)
            rows.append({'matrices': row, 'energy_retained': projected2 / source2 if source2 else None})
        proxies = [last_momentum_proxy([momentum[k][j].numpy() for k in range(cfg['clients'])],
                                       weights_client, ssb) for j in range(2 * len(old_bases))]
        client_retention = [r['energy_retained'] for r in rows]
        valid = [k for k, v in enumerate(client_retention) if v is not None]
        valid_weight = float(sum(weights_client[k] for k in valid))
        weighted = (float(sum(weights_client[k] * client_retention[k] for k in valid) / valid_weight)
                    if valid_weight else None)
        sorted_retention = sorted(client_retention[k] for k in valid)
        max_pyth = max(m['pythagorean_relative_error'] for r in rows for m in r['matrices'])
        max_orth = max(m['residual_orthogonality_error'] for r in rows for m in r['matrices'])
        stats_visits = sum(r['samples'] for r in pilot.s['stats_records'][old_stats:])
        if raw.reads != guard.visits or raw.reads != {1: 155000, 2: 5000} or stats_visits != 35000:
            raise ValueError('actual image access/accounting mismatch')
        engineering = max_pyth <= 1e-5 and max_orth <= 1e-5
        informative = (engineering and len(valid) >= 8 and weighted is not None and weighted >= .05 and
                       float(np.median(sorted_retention)) >= .01)
        summary = {'status': 'completed', 'seed': 0, 'new_training_visits': 125000,
                   'source_t1_training_visits': receipt['shared_training_visits'],
                   'raw_image_reads': raw.reads, 'loader_image_visits': guard.visits,
                   'new_stats_forward_images': stats_visits,
                   'test_images_used': 0, 'old_images_used': 0, 'new_task_training_visits': 0,
                   'weighted_client_retention': weighted,
                   'valid_client_weight': valid_weight, 'zero_energy_clients': cfg['clients'] - len(valid),
                   'retention_sorted': sorted_retention,
                   'maximum_pythagorean_error': max_pyth, 'maximum_residual_orthogonality_error': max_orth,
                   'engineering_passed': engineering, 'diagnostic_signal_informative': informative,
                   'new_logical_communication_bytes': {k: v - old_communication.get(k, 0)
                                                       for k, v in pilot.s['communication'].items()},
                   'proxy_difference_norms': [p['difference_norm'] for p in proxies],
                   'proxy_relative_differences': [p['relative_difference'] for p in proxies],
                   'elapsed_seconds': time.time() - clock['started'], 'trajectory_note': 'T2 evaluation omitted; not a full run',
                   'limitations': ['training-only diagnostics; no evidence of old-class safety or T3 benefit',
                                   'last local momenta proxy is not the full training trajectory bias']}
        write(root / 'reports/PRIVATE_CLIENT_METRICS.json', {'rows': rows, 'proxies': proxies})
        write(root / 'reports/RESULT.json', summary)
    finally:
        raw.close()


if __name__ == '__main__':
    main()
