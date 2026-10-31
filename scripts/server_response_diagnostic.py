"""Fixed-endpoint Muon server candidate diagnostic; no training or test access."""
import copy
import json
import os
import time
from pathlib import Path
import numpy as np
import timm
import torch
from torch.nn import functional as F
from fcl.checkpoint import restore_rng, rng_state
from fcl.data import loader, transforms
from fcl.model import MuLoRAFL
from research.head_regmean.pilot import Pilot, observe
from research.continual_muon.current_cifar import CurrentCIFAR
from research.continual_muon.current_task_guard import CurrentTaskGuard
from research.continual_muon.server_candidate import candidate
from research.continual_muon.server_probe_split import split
from scripts.real_boundary_worker import same_rng, write


def evaluate(pilot, client_splits, part):
    groups = pilot.manifest['partitions']['train'][2]
    weights = [len(g) for g in groups]
    denom = sum(w for w, pair in zip(weights, client_splits) if pair[0] and pair[1])
    if not denom:
        raise ValueError('no client has both disjoint probes')
    totals = {'all': 0., 'current': 0., 'accuracy': 0., 'samples': 0, 'correct': 0}
    predictions = []
    with observe(pilot.model):
        for k, pair in enumerate(client_splits):
            ids = pair[part]
            local_all = local_current = local_correct = count = 0
            local_predictions = []
            for x, y in pilot.loader(pilot.raw[0], ids, pilot.cfg['batch']):
                pilot.budget();pilot.check_labels(y)
                y = y.to(pilot.device)
                logits = pilot.model(pilot.test_tf(x.to(pilot.device))).double()
                c = pilot.cfg['cpt']
                local_all += float(F.cross_entropy(logits, y, reduction='sum'))
                local_current += float(F.cross_entropy(logits[:, 2*c:3*c], y-2*c, reduction='sum'))
                pred = logits.argmax(1)
                local_predictions.extend(zip(y.cpu().tolist(), pred.cpu().tolist()))
                local_correct += int((pred == y).sum());count += len(y)
            if count != len(ids):
                raise ValueError('probe count mismatch')
            if count:
                factor = weights[k] / denom
                totals['all'] += factor * local_all / count
                totals['current'] += factor * local_current / count
                totals['accuracy'] += factor * local_correct / count
                totals['samples'] += count;totals['correct'] += local_correct
            predictions.extend(local_predictions)
    totals['weight_coverage'] = denom / sum(weights)
    return totals, predictions


def main():
    root = Path(os.environ['PROJECT_ROOT']);source = Path(os.environ['SOURCE_ROOT'])
    endpoints = Path(os.environ['ENDPOINT_ROOT']);report = root / 'reports'
    job = json.loads((report / 'job.json').read_text());clock = {k: job[k] for k in ('started', 'deadline')}
    if not clock['started'] <= time.time() < clock['deadline']:
        raise TimeoutError('diagnostic original deadline exhausted')
    with (report / 'WORKER_CLAIM.json').open('x') as f:json.dump({'clock': clock, 'started': time.time()}, f)
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.benchmark = False;torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False;torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    origin = torch.load(endpoints / 'runs/shared/latest.pt', map_location='cpu', weights_only=False)
    final = torch.load(endpoints / 'runs/zero/latest.pt', map_location='cpu', weights_only=False)
    if (origin['clock'] != final['clock'] or origin['identity'] != final['identity'] or
            origin['config'] != final['config'] or origin['head_count'] != 3 or final['head_count'] != 3):
        raise ValueError('endpoints do not share identity and source clock')
    if ((origin['state']['task'], origin['state']['round'], origin['state']['phase']) != (2, 0, 'round_start') or
            (final['state']['task'], final['state']['round'], final['state']['phase']) != (2, 5, 'ready_evaluation') or
            final['state']['history'].committed != {0, 1}):
        raise ValueError('unexpected endpoint lifecycle')
    cfg = copy.deepcopy(final['config'])
    manifest = json.loads((source / cfg['manifest']).read_text())
    weight_info = json.loads((source / 'weights.json').read_text())
    if (cfg['seed'] != 0 or manifest['identity'] != {k: cfg[k] for k in ('dataset', 'seed', 'alpha', 'clients')} or
            final['identity']['manifest'] != cfg['manifest'] or
            final['identity']['weight_identity'] != weight_info['sha256'] or
            final['identity']['source_revision'] != cfg['source_revision']):
        raise ValueError('source manifest mismatch')
    groups = manifest['partitions']['train'];indices = {t: [i for group in groups[t] for i in group] for t in (1, 2)}
    if len(indices[2]) != 5000 or any(not group for group in groups[2]):
        raise ValueError('unexpected current training clients')
    train_manifest = {'identity': manifest['identity'], 'partitions': {'train': groups},
                      'labels': {'train': manifest['labels']['train']}}
    raw = CurrentCIFAR(source / 'data/cifar-100-python/train', train_manifest['labels']['train'], indices)
    guard = CurrentTaskGuard(loader, raw, indices);raw.enter_t3();guard.enter_t3()
    try:
        backbone = timm.create_model(cfg['backbone'], pretrained=False, num_classes=0)
        model = MuLoRAFL(backbone, 100, cfg['rank'], cfg['cosine_scale']).cuda()
        for _ in range(3):model.add_head(cfg['cpt'])
        model.load_state_dict(final['model'], strict=True)
        for name, adapter in model.adapters.items():adapter.merged = final['merged_flags'][name]
        train_tf, stat_tf = transforms()
        pilot = Pilot(model, [raw, None], train_manifest, guard, train_tf.cuda(), stat_tf.cuda(),
                      cfg, root / 'runs/diagnostic', final['identity'], clock)
        pilot.s = final['state'];restore_rng(final['rng'])
        if len(final['rng']['cuda']) != 1 or not torch.equal(torch.cuda.get_rng_state(), final['rng']['cuda'][0]):
            raise ValueError('source GPU RNG restore mismatch')
        from fcl.math import mulora_ssb
        state_rng = rng_state()
        i = torch.arange(768, dtype=torch.float32)[:, None] + 1
        j = torch.arange(32, dtype=torch.float32)[None, :] + 1
        q = torch.linalg.qr(torch.sin(i*j*.013), mode='reduced').Q
        error = float(torch.linalg.vector_norm(mulora_ssb(q) - mulora_ssb(q.cuda()).cpu()))
        if not np.isfinite(error) or error > 1e-4 or not same_rng(state_rng, rng_state()):
            raise ValueError('GPU deterministic SSB preflight failed')
        write(report / 'REAL_PREFLIGHT.json', {'passed': True, 'cpu_gpu_ssb_l2': error,
              'source_gpu_rng_restored': True, 'rng_unchanged': True, 'image_accesses': 0,
              'student_forwards': 0, 'student_backwards': 0})
        bases, matrices = [], []
        for name, adapter in model.adapters.items():
            key = next(k for k in final['model'] if k.endswith('blocks.'+name+'.attn.qkv.A'))
            if not torch.equal(origin['model'][key], final['model'][key]):
                raise ValueError('shared and endpoint bases differ')
            bases.append(adapter.A.detach().cpu().clone())
            matrices.extend((adapter.Bq, adapter.Bv))
        baseline = [v.detach().cpu().clone() for v in matrices]
        # Shared T3 B is identically zero, hence the endpoint B is the five-round cumulative update.
        origin_b = [v for k, v in origin['model'].items() if k.endswith(('.Bq', '.Bv'))]
        if len(origin_b) != 24 or len(matrices) != 24 or any(torch.count_nonzero(v) for v in origin_b):
            raise ValueError('shared initial B not zero')
        clients = pilot.s['client_states']
        if set(clients) != set(range(cfg['clients'])):
            raise ValueError('client momentum set mismatch')
        weights = [len(group) / len(indices[2]) for group in groups[2]]
        started = time.time();deployed, layer_receipts = [], []
        for j, u in enumerate(baseline):
            pilot.budget()
            name = list(model.adapters)[j // 2]
            a = bases[j // 2].double();history = pilot.s['history'].raw[name].double()
            projected_history = a @ history @ a.T
            values = [clients[k]['m'][j] for k in range(cfg['clients'])]
            result, info = candidate(u, values, weights, projected_history, angle=.1, damping=1., tolerance=1e-10)
            deployed.append(result);layer_receipts.append(info)
        candidate_seconds = time.time() - started
        ssb_calls = sum(0 if v['reason'] in ('empty_history', 'zero_update_or_angle') else 11 for v in layer_receipts)
        if not same_rng(state_rng, rng_state()):
            raise ValueError('candidate construction changed RNG')
        heads = [x.detach().clone() for x in model.rows]
        buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
        gradients = [None if v.grad is None else v.grad.detach().clone() for v in model.parameters()]
        client_splits = split(groups[2]);samples = sum(len(a)+len(b) for a,b in client_splits)
        if samples != 292:
            raise ValueError('frozen probe count mismatch')
        before = [evaluate(pilot, client_splits, part) for part in (0, 1)]
        with torch.no_grad():
            for target, value in zip(matrices, deployed):target.copy_(value)
        after = [evaluate(pilot, client_splits, part) for part in (0, 1)]
        with torch.no_grad():
            for target, value in zip(matrices, baseline):target.copy_(value)
        if (any(not torch.equal(a, b) for a, b in zip(heads, model.rows)) or
                any(not torch.equal(buffers[k], v) for k, v in model.named_buffers()) or
                any(not torch.equal(a.detach().cpu(), b) for a, b in zip(matrices, baseline)) or
                not same_rng(state_rng, rng_state())):
            raise ValueError('head, buffer, baseline or RNG was not restored')
        for parameter, previous in zip(model.parameters(), gradients):
            if (previous is None) != (parameter.grad is None) or (previous is not None and not torch.equal(previous, parameter.grad)):
                raise ValueError('existing gradients changed')
        if raw.reads != guard.visits or raw.reads != {1: 0, 2: 584}:
            raise ValueError('image cost mismatch')
        metrics = {}
        for part, name in enumerate(('selection', 'confirmation')):
            base, bp = before[part];new, npred = after[part]
            if len(bp) != len(npred) or any(a[0] != b[0] for a, b in zip(bp, npred)):
                raise ValueError('paired predictions do not align')
            counts = {'correct_to_wrong': 0, 'wrong_to_correct': 0, 'wrong_to_different_wrong': 0}
            for (y, a), (_, b) in zip(bp, npred):
                if a == y and b != y:counts['correct_to_wrong'] += 1
                elif a != y and b == y:counts['wrong_to_correct'] += 1
                elif a != y and b != y and a != b:counts['wrong_to_different_wrong'] += 1
            metrics[name] = {'baseline': base, 'candidate': new, 'paired_flips': counts,
                             'all_loss_difference': new['all'] - base['all'],
                             'current_loss_difference': new['current'] - base['current']}
        confirmed = metrics['confirmation'];accepted = sum(v['accepted'] for v in layer_receipts)
        checks = {'accepted_matrix': bool(accepted > 0), 'coverage': bool(confirmed['baseline']['weight_coverage'] >= .95),
                  'all_loss_improved': bool(confirmed['all_loss_difference'] < -1e-6),
                  'current_loss_not_worse': bool(confirmed['current_loss_difference'] <= 1e-6)}
        reasons = {r: sum(v['reason'] == r for v in layer_receipts) for r in sorted({v['reason'] for v in layer_receipts})}
        output = {'status': 'completed', 'checks': checks, 'local_conditions_passed': all(checks.values()),
                  'seed': cfg['seed'], 'accepted_matrices': accepted, 'matrix_reasons': reasons,
                  'metrics': metrics, 'student_forward_images': 584, 'student_backward_images': 0,
                  'teacher_forward_images': 0, 'new_training_visits': 0, 'test_images_used': 0,
                  'old_train_images_used': 0, 'ssb_calls': ssb_calls, 'candidate_seconds': candidate_seconds,
                  'extra_network_bytes_measured': None, 'raw_image_reads': raw.reads,
                  'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'elapsed_seconds': time.time() - clock['started'],
                  'limitations': ['B is the five-round cumulative task update, not the last round',
                                  'last client momenta are not common-anchor gradients or full trajectory bias',
                                  'both disjoint probes were previously used in training, basis statistics and a separate gated branch probe',
                                  'no performance, generalization, old-class safety or measured-network claim']}
        write(report / 'RESULT.json', output)
    finally:
        raw.close()


if __name__ == '__main__':
    main()
