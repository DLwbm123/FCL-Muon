"""Unregistered draft: paired T3 state-transport experiment."""
import copy
import json
import os
import shutil
import time
from pathlib import Path
import numpy as np
import timm
import torch
from torchvision.datasets import CIFAR100
from fcl.checkpoint import atomic_save, rng_state
from fcl.data import loader, transforms
from fcl.math import mulora_ssb
from fcl.model import MuLoRAFL
from research.head_regmean.pilot import Pilot, observe
from research.continual_muon.current_cifar import CurrentCIFAR
from research.continual_muon.current_task_guard import CurrentTaskGuard
from research.continual_muon.current_gradient_probe import probe
from research.continual_muon.import_boundary import import_t1
from research.continual_muon.paired_boundary import capture, install
from research.continual_muon.paired_metrics import compare, conditions
from scripts.real_boundary_worker import same_rng, write


def model_for(cfg):
    return MuLoRAFL(timm.create_model(cfg['backbone'], pretrained=False, num_classes=0),
                    100, cfg['rank'], cfg['cosine_scale']).cuda()


def predictions(pilot, task):
    y_all, predictions = [], []
    with observe(pilot.model):
        for t in range(task + 1):
            for x, y in loader(pilot.raw[1], pilot.ids('test', t), pilot.cfg['batch']):
                pilot.budget()
                output = pilot.model(pilot.test_tf(x.to(pilot.device)))
                predictions.extend(output.argmax(1).cpu().tolist())
                y_all.extend(y.tolist())
    y, pred = np.asarray(y_all), np.asarray(predictions)
    if len(y) != 1000 * (task + 1) or len(pred) != len(y):
        raise ValueError('unexpected evaluation size')
    c = pilot.cfg['cpt']
    per_task = [float(np.mean(pred[(y >= i*c) & (y < (i+1)*c)] ==
                             y[(y >= i*c) & (y < (i+1)*c)])) for i in range(task + 1)]
    return y, pred, per_task


def main():
    root, source = Path(os.environ['PROJECT_ROOT']), Path(os.environ['SOURCE_ROOT'])
    report = root / 'reports'
    job = json.loads((report / 'job.json').read_text())
    clock = {k: job[k] for k in ('started', 'deadline')}
    if not clock['started'] <= time.time() < clock['deadline']:
        raise TimeoutError('original study clock exhausted')
    with (report / 'WORKER_CLAIM.json').open('x') as f:
        json.dump({'started': time.time(), 'clock': clock}, f)
    cfg = json.loads((source / 'configs/aggregation_utility_three_task_seed0.json').read_text())
    cfg['output'] = 'runs/shared'
    expected = {'seed': 0, 'dataset': 'seq-cifar100', 'alpha': .05, 'clients': 10,
                'rounds': 5, 'epochs': 5, 'batch': 16, 'rank': 32, 'lr': .0002,
                'aggregation': 'regmean', 'selection_mode': 'prefix'}
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError('source config mismatch')
    full_manifest = json.loads((source / cfg['manifest']).read_text())
    if full_manifest['identity'] != {k: cfg[k] for k in ('dataset', 'seed', 'alpha', 'clients')}:
        raise ValueError('manifest identity mismatch')
    groups = full_manifest['partitions']['train']
    indices = {t: [i for g in groups[t] for i in g] for t in (1, 2)}
    if any(len(indices[t]) != 5000 or len(set(indices[t])) != 5000 for t in (1, 2)):
        raise ValueError('invalid current train partition')
    if any(not t * cfg['cpt'] <= full_manifest['labels']['train'][i] < (t+1)*cfg['cpt']
           for t in (1, 2) for i in indices[t]):
        raise ValueError('invalid training labels')
    train_manifest = {'identity': full_manifest['identity'], 'partitions': {'train': groups},
                      'labels': {'train': full_manifest['labels']['train']}}
    weight_info = json.loads((source / 'weights.json').read_text())
    identity = {'manifest': cfg['manifest'], 'weight_identity': weight_info['sha256'],
                'source_revision': cfg['source_revision'], 'purpose': 'three-task aggregation-utility selection'}
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.benchmark = False;torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False;torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    train_tf, test_tf = transforms();train_tf = train_tf.cuda();test_tf = test_tf.cuda()
    raw = CurrentCIFAR(source / 'data/cifar-100-python/train', train_manifest['labels']['train'], indices)
    guard = CurrentTaskGuard(loader, raw, indices)
    try:
        pilot = Pilot(model_for(cfg), [raw, None], train_manifest, guard, train_tf, test_tf,
                      cfg, root / cfg['output'], identity, clock)
        prefix = torch.load(source / 'runs/prefix/latest.pt', map_location='cpu', weights_only=False)
        receipt = import_t1(pilot, prefix, identity)
        if len(prefix['rng']['cuda']) != 1 or not torch.equal(torch.cuda.get_rng_state(), prefix['rng']['cuda'][0]):
            raise ValueError('source GPU RNG restore mismatch')
        t1_matrix = copy.deepcopy(prefix['state']['matrix'])
        old_visits = sum(v['visited'] for v in prefix['state']['training_records'])
        source_stats_count = len(prefix['state']['stats_records'])
        prefix_communication = copy.deepcopy(prefix['state']['communication'])
        del prefix
        preflight_rng = rng_state()
        i = torch.arange(768, dtype=torch.float32)[:, None] + 1
        j = torch.arange(32, dtype=torch.float32)[None, :] + 1
        sample = torch.linalg.qr(torch.sin(i * j * .013), mode='reduced').Q
        oracle_error = float(torch.linalg.vector_norm(mulora_ssb(sample) - mulora_ssb(sample.cuda()).cpu()))
        if not np.isfinite(oracle_error) or oracle_error > 1e-4 or not same_rng(preflight_rng, rng_state()):
            raise ValueError('GPU SSB preflight or RNG isolation failed')
        write(report / 'REAL_PREFLIGHT.json', {'passed': True, 'source_cuda_rng_restored': True,
              'cpu_gpu_ssb_l2': oracle_error, 'rng_unchanged': True, 'image_accesses': 0,
              'student_forward_images': 0, 'student_backward_images': 0})
        while pilot.s['phase'] != 'ready_evaluation':
            if pilot.s['task'] != 1:
                raise ValueError('unexpected T2 state')
            pilot.step()
            write(report / 'PROGRESS.json', {'time': time.time(), 'stage': 'shared_t2',
                  'round': pilot.s['round'], 'phase': pilot.s['phase'], 'reads': raw.reads})
        if sum(v['visited'] for v in pilot.s['training_records']) - old_visits != 125000:
            raise ValueError('T2 training budget mismatch')
        before = rng_state(); snapshot = capture(pilot)
        if not same_rng(before, rng_state()):
            raise ValueError('capture altered RNG')
        snapshot_path = root / 'runs/t2_private_state.pt'
        atomic_save(snapshot, snapshot_path)
        reloaded = torch.load(snapshot_path, map_location='cpu', weights_only=False)
        if (reloaded['identity'] != snapshot['identity'] or reloaded['clock'] != snapshot['clock'] or
                reloaded['adapter_names'] != snapshot['adapter_names'] or
                any(not torch.equal(a, b) for a, b in zip(reloaded['old_bases'], snapshot['old_bases'])) or
                any(not torch.equal(a, b) for k in snapshot['momenta']
                    for a, b in zip(reloaded['momenta'][k], snapshot['momenta'][k])) or
                not same_rng(before, rng_state())):
            raise ValueError('T2 private snapshot reload or RNG mismatch')
        snapshot = reloaded
        shutil.copyfile(pilot.out / 'latest.pt', root / 'runs/t2_endpoint_private.pt')
        pilot.s['phase'] = 'evaluated'
        for expected_phase in ('merged', 'committed', 'task_start'):
            pilot.step()
            if pilot.s['phase'] != expected_phase:
                raise ValueError('unexpected T2 merge/commit lifecycle')
        raw.enter_t3();guard.enter_t3();pilot.step()
        if (pilot.s['task'], pilot.s['round'], pilot.s['phase']) != (2, 0, 'round_start'):
            raise ValueError('T3 initialization failed')
        if sum(v['samples'] for v in pilot.s['stats_records'][source_stats_count:]) != 35000:
            raise ValueError('shared training statistics mismatch')
        pre_probe = rng_state()
        probes, probe_costs = {}, []
        for k in range(cfg['clients']):
            probes[k], cost = probe(pilot, k, count=32)
            probe_costs.append(cost)
        if sum(v['samples'] for v in probe_costs) != 292:
            raise ValueError('frozen training probe count mismatch')
        if not same_rng(pre_probe, rng_state()):
            raise ValueError('probe altered global RNG')
        atomic_save({'probes': probes, 'costs': probe_costs}, root / 'runs/t3_probe_private.pt')
        pilot.save()
        shared_checkpoint = pilot.out / 'latest.pt'
        shared_comm = copy.deepcopy(pilot.s['communication'])
        shared_stats_count = len(pilot.s['stats_records'])
        shared_comm_delta = {k: int(v - prefix_communication.get(k, 0)) for k, v in shared_comm.items()}
        branch_receipts, branch_ckpts, branch_comm = {}, {}, {}
        for mode in ('zero', 'projected', 'guarded'):
            out = root / 'runs' / mode;out.mkdir()
            shutil.copyfile(shared_checkpoint, out / 'latest.pt')
            pilot.out = out;pilot.restore()
            if not same_rng(pre_probe, rng_state()):
                raise ValueError('branch RNG is not matched')
            branch_receipts[mode] = install(pilot, snapshot, mode, probes if mode == 'guarded' else None)
            if not same_rng(pre_probe, rng_state()):
                raise ValueError('state installation altered branch RNG')
            if mode == 'guarded' and any(not info['accepted'] and
                any(torch.count_nonzero(v) for v in pilot.s['client_states'][k]['m'])
                for k, info in branch_receipts[mode].items()):
                raise ValueError('rejected client was not restored to exact zero')
            pilot.save()
            while pilot.s['phase'] != 'ready_evaluation':
                if pilot.s['task'] != 2:
                    raise ValueError('unexpected T3 training state')
                pilot.step()
                write(report / 'PROGRESS.json', {'time': time.time(), 'stage': mode,
                      'round': pilot.s['round'], 'phase': pilot.s['phase'], 'reads': raw.reads})
            visits = sum(v['visited'] for v in pilot.s['training_records'] if v['task'] == 2)
            if visits != 125000:
                raise ValueError('T3 arm training budget mismatch')
            if sum(v['samples'] for v in pilot.s['stats_records'][shared_stats_count:]) != 25000:
                raise ValueError('T3 arm statistics budget mismatch')
            branch_comm[mode] = {k: int(v - shared_comm.get(k, 0)) for k, v in pilot.s['communication'].items()}
            branch_ckpts[mode] = out / 'latest.pt'
        if raw.reads != guard.visits or raw.reads != {1: 155000, 2: 455292}:
            raise ValueError('training-side image accounting mismatch')
        # No test dataset is instantiated before all training arms reach their fixed endpoint.
        test = CIFAR100(source / 'data', train=False, download=False)
        if list(test.targets) != full_manifest['labels']['test']:
            raise ValueError('test labels disagree with frozen manifest')
        arm_predictions, arm_metrics = {}, {}
        t2_out = root / 'runs/t2_eval';t2_out.mkdir()
        shutil.copyfile(root / 'runs/t2_endpoint_private.pt', t2_out / 'latest.pt')
        for mode, path, task in [('t2', t2_out, 1)] + [(m, branch_ckpts[m].parent, 2) for m in ('zero', 'projected', 'guarded')]:
            evaluator = Pilot(model_for(cfg), [None, test], full_manifest, loader, train_tf, test_tf,
                              cfg, path, identity, clock)
            evaluator.restore()
            y, pred, per_task = predictions(evaluator, task)
            arm_predictions[mode] = (y, pred)
            arm_metrics[mode] = per_task
            del evaluator
            torch.cuda.empty_cache()
        t2 = arm_metrics['t2']
        comparisons = {}
        baseline_y, baseline_pred = arm_predictions['zero']
        for mode in ('zero', 'projected', 'guarded'):
            y, pred = arm_predictions[mode]
            if not np.array_equal(y, baseline_y):
                raise ValueError('paired test order mismatch')
            comparisons[mode] = compare(y, pred, baseline_pred, cfg['cpt'], t1_matrix[0][0], t2)
        gate_conditions = conditions(comparisons)
        write(report / 'PRIVATE_GATE_RECORDS.json', branch_receipts)
        write(report / 'RESULT.json', {'status': 'completed', 'seed': 0, 'comparisons': comparisons,
              'source_t1_training_visits': receipt['shared_training_visits'], 'new_training_visits': 500000,
              'training_side_image_reads': raw.reads, 'probe_costs': probe_costs,
              'new_training_statistics_forward_images': 110000,
              'shared_logical_communication_bytes': shared_comm_delta,
              'per_branch_logical_communication_bytes': branch_comm,
              'gate_conditions': gate_conditions, 'exploratory_gate_passed': all(gate_conditions.values()),
              'test_images_used': 11000, 'test_dataset_opened_after_all_training': True,
              'elapsed_seconds': time.time() - clock['started'],
              'limitations': ['single-seed component study, not final method or statistical significance',
                              'current-task probe is not an old-class safety guarantee']})
    finally:
        raw.close()


if __name__ == '__main__':
    main()
