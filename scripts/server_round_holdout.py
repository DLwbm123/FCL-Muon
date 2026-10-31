"""Draft one-round Muon server candidate diagnostic with training-out confirmation."""
import copy
import json
import os
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
from research.fed_spectra.heldout_round import GuardedLoader, split_clients
from research.continual_muon.current_cifar import CurrentCIFAR
from research.continual_muon.current_task_guard import CurrentTaskGuard
from research.continual_muon.import_t2_round import import_boundary
from research.continual_muon.server_candidate import candidate
from research.continual_muon.server_gradient_probe import directional
from scripts.server_response_diagnostic import evaluate
from scripts.real_boundary_worker import same_rng, write


def main():
    root = Path(os.environ['PROJECT_ROOT']);source = Path(os.environ['SOURCE_ROOT'])
    endpoints = Path(os.environ['ENDPOINT_ROOT']);report = root / 'reports'
    job = json.loads((report / 'job.json').read_text());clock = {k: job[k] for k in ('started', 'deadline')}
    if not clock['started'] <= time.time() < clock['deadline']:
        raise TimeoutError('original diagnostic deadline exhausted')
    with (report / 'WORKER_CLAIM.json').open('x') as f:json.dump({'clock': clock, 'started': time.time()}, f)
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.benchmark = False;torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False;torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    source_state = torch.load(endpoints / 'runs/t2_endpoint_private.pt', map_location='cpu', weights_only=False)
    snapshot = torch.load(endpoints / 'runs/t2_private_state.pt', map_location='cpu', weights_only=False)
    cfg = copy.deepcopy(source_state['config']);cfg['output'] = 'runs/diagnostic'
    expected = {'seed': 0, 'dataset': 'seq-cifar100', 'alpha': .05, 'clients': 10,
                'rounds': 5, 'epochs': 5, 'batch': 16, 'rank': 32, 'lr': .0002,
                'aggregation': 'regmean', 'selection_mode': 'prefix'}
    if any(cfg.get(k) != v for k, v in expected.items()):raise ValueError('source config mismatch')
    manifest = json.loads((source / cfg['manifest']).read_text())
    weights_info = json.loads((source / 'weights.json').read_text())
    expected_identity = {'manifest': cfg['manifest'], 'weight_identity': weights_info['sha256'],
                         'source_revision': cfg['source_revision'], 'purpose': 'three-task aggregation-utility selection'}
    if manifest['identity'] != {k: cfg[k] for k in ('dataset', 'seed', 'alpha', 'clients')}:
        raise ValueError('manifest identity mismatch')
    original_groups = manifest['partitions']['train'];fit, heldout = split_clients(original_groups[2], 0)
    if sum(map(len, fit)) != 4005 or sum(map(len, heldout)) != 995:
        raise ValueError('fixed training/confirmation split mismatch')
    original_ids = {t: [i for group in original_groups[t] for i in group] for t in (1, 2)}
    groups = copy.deepcopy(original_groups);groups[2] = fit
    train_manifest = {'identity': manifest['identity'], 'partitions': {'train': groups},
                      'labels': {'train': manifest['labels']['train']}}
    raw = CurrentCIFAR(source / 'data/cifar-100-python/train', train_manifest['labels']['train'], original_ids)
    taskguard = CurrentTaskGuard(loader, raw, original_ids)
    fit_ids = [i for group in fit for i in group];held_ids = [i for group in heldout for i in group]
    guard = GuardedLoader(taskguard, raw, fit_ids, held_ids)
    try:
        backbone = timm.create_model(cfg['backbone'], pretrained=False, num_classes=0)
        model = MuLoRAFL(backbone, 100, cfg['rank'], cfg['cosine_scale']).cuda()
        train_tf, stat_tf = transforms()
        pilot = Pilot(model, [raw, None], train_manifest, guard, train_tf.cuda(), stat_tf.cuda(),
                      cfg, root / 'runs/diagnostic', expected_identity, clock)
        receipt = import_boundary(pilot, source_state, snapshot, expected_identity)
        if len(source_state['rng']['cuda']) != 1 or not torch.equal(torch.cuda.get_rng_state(), source_state['rng']['cuda'][0]):
            raise ValueError('source GPU RNG restore mismatch')
        state_rng = rng_state();i = torch.arange(768, dtype=torch.float32)[:, None] + 1
        j = torch.arange(32, dtype=torch.float32)[None, :] + 1
        q = torch.linalg.qr(torch.sin(i*j*.013), mode='reduced').Q
        error = float(torch.linalg.vector_norm(mulora_ssb(q) - mulora_ssb(q.cuda()).cpu()))
        if not np.isfinite(error) or error > 1e-4 or not same_rng(state_rng, rng_state()):
            raise ValueError('GPU deterministic SSB preflight failed')
        write(report / 'REAL_PREFLIGHT.json', {'passed': True, 'cpu_gpu_ssb_l2': error,
              'source_gpu_rng_restored': True, 'rng_unchanged': True, 'image_accesses': 0,
              'student_forwards': 0, 'student_backwards': 0})
        before_visits = sum(v['visited'] for v in pilot.s['training_records'])
        before_stats = len(pilot.s['stats_records']);before_comm = copy.deepcopy(pilot.s['communication'])
        pilot.s['phase'] = 'evaluated'  # Skip source T2 evaluation; no test images are opened.
        for expected_phase in ('merged', 'committed', 'task_start'):
            pilot.step()
            if pilot.s['phase'] != expected_phase:raise ValueError('T2 boundary transition failed')
        raw.enter_t3();taskguard.enter_t3();pilot.step()
        if (pilot.s['task'], pilot.s['round'], pilot.s['phase']) != (2, 0, 'round_start'):
            raise ValueError('T3 initialization failed')
        start_b = [v.detach().cpu().clone() for v in model.b_parameters()]
        if len(start_b) != 24 or any(torch.count_nonzero(v) for v in start_b):
            raise ValueError('expected zero T3 starting B matrices')
        while pilot.s['phase'] != 'local_done':
            if pilot.s['task'] != 2 or pilot.s['round'] != 0:raise ValueError('unexpected T3 round state')
            pilot.step()
            write(report / 'PROGRESS.json', {'time': time.time(), 'round': pilot.s['round'],
                  'phase': pilot.s['phase'], 'new_client_rounds': len(pilot.s['training_records']) - 100,
                  'fit_reads': guard.training_access_images, 'heldout_reads': guard.validation_access_images})
        if (sum(v['visited'] for v in pilot.s['training_records']) - before_visits != 20025 or
                sum(v['samples'] for v in pilot.s['stats_records'][before_stats:]) != 8010 or
                guard.validation_access_images or guard.training_validation_accesses):
            raise ValueError('training or validation isolation mismatch')
        # The true one-round aggregate B and client terminal momenta are present at local_done.
        if len(pilot.s['agg_B']) != 24 or len(pilot.s['client_states']) != 10:
            raise ValueError('incomplete real-round endpoint')
        pilot.fuse();baseline = [v.detach().cpu().clone() for v in model.b_parameters()]
        if pilot.s['round'] != 1 or pilot.s['phase'] != 'round_start':raise ValueError('round fusion failed')
        weights = [len(g) / len(fit_ids) for g in fit]
        candidate_start = time.time();deployed, layer_receipts = [], []
        for j, (origin, base) in enumerate(zip(start_b, baseline)):
            pilot.budget();name = list(model.adapters)[j // 2]
            a = model.adapters[name].A.detach().cpu().double();history = pilot.s['history'].raw[name].double()
            local_momenta = [pilot.s['client_states'][k]['m'][j] for k in range(cfg['clients'])]
            update = base - origin
            adjusted, info = candidate(update, local_momenta, weights, a @ history @ a.T,
                                       angle=.1, damping=1., tolerance=1e-10)
            deployed.append(origin + adjusted);layer_receipts.append(info)
        candidate_seconds = time.time() - candidate_start
        ssb_calls = sum(0 if v['reason'] in ('empty_history', 'zero_update_or_angle') else 11 for v in layer_receipts)
        client_splits = [(sorted(a)[:32], sorted(b)[:32]) for a, b in zip(fit, heldout)]
        if sum(len(a) for a, _ in client_splits) != 281 or sum(len(b) for _, b in client_splits) != 236:
            raise ValueError('fixed fit/confirmation probe counts mismatch')
        if guard.validation_access_images or guard.training_validation_accesses:
            raise ValueError('heldout data were seen before candidate frozen')
        write(report / 'CANDIDATE_FROZEN.json', {'accepted_matrices': sum(v['accepted'] for v in layer_receipts),
              'heldout_reads_at_freeze': guard.validation_access_images, 'time': time.time()})
        matrices = model.b_parameters();heads = [v.detach().clone() for v in model.rows]
        buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
        grads = [None if v.grad is None else v.grad.detach().clone() for v in model.parameters()]
        anchor_rng = rng_state();correction = [new.double()-old.double() for new, old in zip(deployed, baseline)]
        metrics, derivatives = {}, {}
        for part, name in enumerate(('fit', 'confirmation')):
            if part:guard.allow_validation = True
            before, bp = evaluate(pilot, client_splits, part)
            derivatives[name] = directional(pilot, client_splits, correction, part)
            with torch.no_grad():
                for target, value in zip(matrices, deployed):target.copy_(value)
            after, ap = evaluate(pilot, client_splits, part)
            with torch.no_grad():
                for target, value in zip(matrices, baseline):target.copy_(value)
            if len(bp) != len(ap) or any(a[0] != b[0] for a,b in zip(bp, ap)):
                raise ValueError('paired probe order mismatch')
            flips = {'correct_to_wrong': 0, 'wrong_to_correct': 0, 'wrong_to_different_wrong': 0}
            for (y,a),(_,b) in zip(bp,ap):
                if a == y and b != y:flips['correct_to_wrong'] += 1
                elif a != y and b == y:flips['wrong_to_correct'] += 1
                elif a != y and b != y and a != b:flips['wrong_to_different_wrong'] += 1
            metrics[name] = {'baseline': before, 'candidate': after, 'paired_flips': flips,
                             'all_loss_difference': after['all']-before['all'],
                             'current_loss_difference': after['current']-before['current'],
                             'accuracy_difference': after['accuracy']-before['accuracy']}
        if (guard.training_validation_accesses or guard.training_access_images != 28878 or
                guard.validation_access_images != 708 or raw.reads != taskguard.visits or raw.reads != {1: 0, 2: 29586}):
            raise ValueError('raw or guarded data accounting mismatch')
        if (not same_rng(anchor_rng, rng_state()) or any(not torch.equal(a,b) for a,b in zip(heads, model.rows)) or
                any(not torch.equal(buffers[k], v) for k,v in model.named_buffers()) or
                any(not torch.equal(a.detach().cpu(), b) for a,b in zip(matrices, baseline))):
            raise ValueError('model or RNG not restored after diagnostics')
        for parameter, previous in zip(model.parameters(), grads):
            if (previous is None) != (parameter.grad is None) or (previous is not None and not torch.equal(previous, parameter.grad)):
                raise ValueError('existing gradient not restored')
        accepted = sum(v['accepted'] for v in layer_receipts)
        checks = {'accepted_matrix': bool(accepted > 0)}
        for name in ('fit','confirmation'):
            m = metrics[name]
            checks[name+'_coverage'] = bool(m['baseline']['weight_coverage'] >= .95)
            checks[name+'_all_loss_improved'] = bool(m['all_loss_difference'] < -1e-6)
            checks[name+'_current_loss_not_worse'] = bool(m['current_loss_difference'] <= 1e-6)
        checks['confirmation_accuracy_not_lower'] = bool(metrics['confirmation']['accuracy_difference'] >= -1e-12)
        reasons = {r: sum(v['reason'] == r for v in layer_receipts) for r in sorted({v['reason'] for v in layer_receipts})}
        communication = {k: int(v-before_comm.get(k, 0)) for k,v in pilot.s['communication'].items()}
        output = {'status': 'completed', 'local_conditions_passed': all(checks.values()), 'checks': checks,
                  'seed': 0, 'accepted_matrices': accepted, 'matrix_reasons': reasons, 'metrics': metrics,
                  'directional': derivatives, 'new_training_visits': 20025, 'statistics_forward_images': 8010,
                  'diagnostic_student_forward_images': 1551, 'diagnostic_student_backward_images': 517,
                  'teacher_forward_images': 0, 'test_images_used': 0, 'old_train_images_used': 0,
                  'fit_reads': guard.training_access_images, 'heldout_reads': guard.validation_access_images,
                  'heldout_reads_at_candidate_freeze': 0, 'ssb_calls': ssb_calls, 'candidate_seconds': candidate_seconds,
                  'source_t1_t2_training_visits': receipt['source_training_visits'],
                  'new_logical_communication_bytes': communication,
                  'theoretical_extra_client_momentum_upload_bytes': 23592960,
                  'theoretical_two_probe_gradient_upload_bytes': 47185920,
                  'theoretical_one_candidate_broadcast_bytes': 23592960,
                  'extra_network_bytes_measured': None, 'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'elapsed_seconds': time.time()-clock['started'],
                  'limitations': ['one seed and one round, not a full FCL result or old-class safety evidence',
                                  'heldout excluded from this run but previously used in other research; pretraining overlap unknown',
                                  'last local momenta are not common-anchor gradients or full trajectory bias']}
        write(report / 'RESULT.json', output)
    finally:
        raw.close()


if __name__ == '__main__':
    main()
