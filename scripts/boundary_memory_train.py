"""Three-task frozen-reference full-moment pilot; all paths via environment."""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import timm
import torch
from safetensors.torch import load_file
from torchvision.datasets import CIFAR100

from fcl.data import loader, transforms
from research.basic_repair.fcl_basic_repair_tools import center_task_logits, summarize
from research.boundary_memory.core import (
    compose_scores, evidence_scores, lda_classifier, merge_moments, moments, solve_evidence,
)


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def save_tensor(path, value):
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def size(root):
    result = 0
    for directory, _, files in os.walk(root, followlinks=False):
        for name in files:
            path = Path(directory) / name
            try:
                if not path.is_symlink():
                    result += path.stat().st_blocks * 512
            except FileNotFoundError:
                pass
    return result


def summarize_matrix(stages, arm):
    matrix = [[p['all_seen_accuracy'] for p in s['arms'][arm]['per_task']] for s in stages]
    last = matrix[-1]
    return {
        'matrix': matrix, 'final_overall': float(np.mean(last)),
        'final_old': float(np.mean(last[:-1])) if len(last) > 1 else None,
        'final_new': last[-1], 'new_each_stage': [row[-1] for row in matrix],
        'mean_new_T2_T3': float(np.mean([row[-1] for row in matrix[1:]])) if len(matrix) > 1 else None,
        'average_incremental_accuracy': float(np.mean([np.mean(row) for row in matrix])),
        'forgetting': float(np.mean([max(matrix[j][i] for j in range(i, len(matrix))) - last[i]
                                    for i in range(len(last) - 1)])) if len(last) > 1 else None,
    }


def decision(stages, summary, converged):
    base = summary['centered']
    new = summary['boundary_fused']
    steals = lambda arm: sum(p['restricted_correct_but_other_task_wins'] for p in stages[-1]['arms'][arm]['per_task'])
    conditions = {
        'final_overall_improved': new['final_overall'] > base['final_overall'],
        'AIA_improved': new['average_incremental_accuracy'] > base['average_incremental_accuracy'],
        'final_old_not_worse': new['final_old'] >= base['final_old'],
        'mean_new_not_worse': new['mean_new_T2_T3'] >= base['mean_new_T2_T3'],
        'cross_task_stolen_errors_lower': steals('boundary_fused') < steals('centered'),
        'server_objectives_converged': converged,
    }
    return {
        'conditions': conditions, 'gate_passed': all(conditions.values()),
        'fused_minus_reference_lda_final_pp': 100 * (new['final_overall'] - summary['reference_lda']['final_overall']),
        'fused_minus_evidence_alone_final_pp': 100 * (new['final_overall'] - summary['evidence_alone']['final_overall']),
        'fused_minus_no_old_to_new_final_pp': 100 * (new['final_overall'] - summary['no_old_to_new_fused']['final_overall']),
        'next': 'Report all outcomes; no automatic compression or ten-task expansion under this pilot.',
        'not_significance_or_heldout_validation': True,
    }


def main():
    root = Path(os.environ['PROJECT_ROOT'])
    source = Path(os.environ['SOURCE_ROOT'])
    cfg = json.loads((root / 'configs/boundary_full_three_task_seed0.json').read_text())
    clock = json.loads((root / 'reports/job.json').read_text())
    clock = {k: clock[k] for k in ('started', 'deadline')}
    out = root / 'runs/pilot'
    out.mkdir(parents=True, exist_ok=False)
    progress = {'status': 'initializing', 'clock': clock, 'train_reference_images': 0,
                'test_reference_images': 0, 'logical_statistics_upload_bytes': 0,
                'logical_evidence_broadcast_bytes': 0, 'messages': [], 'stages': []}

    def budget():
        if time.time() >= clock['deadline']:
            raise TimeoutError('Original experiment deadline')

    def event(**fields):
        progress.update(fields, updated=time.time())
        write_json(out / 'PROGRESS.json', progress)
        print(json.dumps({k: progress[k] for k in ('status', 'task', 'updated') if k in progress}), flush=True)

    torch.set_num_threads(4)
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    random.seed(cfg['seed'])
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.cuda.reset_peak_memory_stats()
    manifest = json.loads((source / cfg['manifest']).read_text())
    assert manifest['identity'] == {'dataset': 'seq-cifar100', 'seed': 0, 'alpha': .05, 'clients': 10}
    train = CIFAR100(source / 'data', train=True, download=False)
    test = CIFAR100(source / 'data', train=False, download=False)
    assert list(train.targets) == manifest['labels']['train']
    assert list(test.targets) == manifest['labels']['test']
    wm = json.loads((source / 'weights.json').read_text())
    reference = timm.create_model(cfg['backbone'], pretrained=False, num_classes=0)
    reference.load_state_dict({k: v for k, v in load_file(wm['path']).items() if not k.startswith('head.')}, strict=True)
    reference = reference.cuda().eval().requires_grad_(False)
    _, transform = transforms()
    transform = transform.cuda().eval()
    # Parameters never enter an optimizer. State is checked once after all work.
    parameter_versions = [p._version for p in reference.parameters()]
    src_run = source / cfg['source_run']
    source_metrics = json.loads((src_run / 'metrics.json').read_text())
    archive, message_ids, visited_ids, stages, source_identities = [], set(), set(), [], []
    test_phi, test_indices, test_labels = [], [], []
    all_solver_converged = True
    write_json(out / 'IDENTITY.json', {
        'manifest': cfg['manifest'], 'weight_identity': wm['sha256'], 'clock': clock,
        'source_run': str(src_run), 'source_commit': '01d7bd9',
        'proposal': 'https://chatgpt.com/share/6adefa60-1a7c-83e8-aa39-f2ebdc1f76cd',
        'adapted_training_steps_added': 0, 'no_future_checkpoint_loaded': True,
    })

    def extract(data, ids):
        rows, labels = [], []
        with torch.no_grad():
            for x, y in loader(data, ids, cfg['batch'], shuffle=False):
                budget()
                z = reference.forward_features(transform(x.cuda()))[:, 0]
                phi = torch.nn.functional.normalize(z, dim=1)
                if not torch.isfinite(phi).all():
                    raise ArithmeticError('Invalid reference features')
                rows.append(phi.cpu())
                labels.append(y)
        return torch.cat(rows), torch.cat(labels)

    for task in range(cfg['tasks']):
        classes = (task + 1) * cfg['cpt']
        event(status='current_client_statistics', task=task)
        current = [None] * cfg['cpt']
        task_train_ids = []
        for client, ids in enumerate(manifest['partitions']['train'][task]):
            budget()
            assert len(set(ids)) == len(ids) and not (set(ids) & visited_ids)
            assert all(task * cfg['cpt'] <= train.targets[i] < classes for i in ids)
            task_train_ids.extend(ids)
            visited_ids.update(ids)
            if not ids:
                continue
            phi, labels = extract(train, ids)
            progress['train_reference_images'] += len(ids)
            # The simulated client returns moments only; no feature replay buffer.
            for label in labels.unique().tolist():
                message_id = f'task{task}:client{client}:class{label}:reference_v1:statistics_v1'
                assert message_id not in message_ids
                message_ids.add(message_id)
                local = moments(phi[labels == label].cuda())
                local = {k: v.cpu() if torch.is_tensor(v) else v for k, v in local.items()}
                current[label - task * cfg['cpt']] = merge_moments(current[label - task * cfg['cpt']], local)
                dim = phi.shape[1]
                # int64 n, FP64 mean/scatter; no compression claimed.
                payload_bytes = 8 + 8 * dim + 8 * dim * dim
                progress['logical_statistics_upload_bytes'] += payload_bytes
                progress['messages'].append({'id': message_id, 'n': local['n'], 'bytes': payload_bytes})
            del phi, labels
            event(client=client)
        assert len(task_train_ids) == len(set(task_train_ids)) == 5000
        assert all(s is not None and s['n'] == 500 for s in current)
        archive.extend(current)
        assert len(visited_ids) == 5000 * (task + 1)
        counts = torch.tensor([s['n'] for s in archive], dtype=torch.float64, device='cuda')
        means = torch.stack([s['mean'] for s in archive]).cuda()
        covariance = torch.stack([s['scatter'] / s['n'] for s in archive]).cuda()
        covariance = (covariance + covariance.transpose(1, 2)) * .5
        assert len(archive) == classes and means.shape[1] == 768
        save_tensor(out / 'latest.pt', {'phase': 'statistics_saved', 'task': task,
                                      'archive': archive, 'clock': clock, 'config': cfg})
        fits, diagnostics = {}, {}
        for mode in ('bidirectional', 'no_old_to_new'):
            event(status='server_fit', objective=mode, solver_iteration=0)
            start = time.time()

            def callback(iteration, theta):
                budget()
                if iteration % 100 == 0:
                    event(solver_iteration=iteration)

            theta, report = solve_evidence(means, covariance, cfg['solver'], mode, callback)
            report['wall_seconds'] = time.time() - start
            fits[mode] = theta.cpu()
            diagnostics[mode] = report
            all_solver_converged &= report['converged']
        fits['reference_lda'] = lda_classifier(means, covariance, counts, cfg['lda_shrinkage']).cpu()
        # Save fitted state BEFORE any test features/logits for this stage are opened.
        checkpoint = {'phase': 'fit_saved', 'task': task, 'archive': archive, 'fits': fits,
                      'solver': diagnostics, 'clock': clock, 'config': cfg,
                      'train_reference_images': progress['train_reference_images']}
        save_tensor(out / 'latest.pt', checkpoint)
        save_tensor(out / f'task{task + 1}_evidence.pt', {'fits': fits, 'task': task, 'clock': clock})
        progress['logical_evidence_broadcast_bytes'] += fits['bidirectional'].numel() * 8 * cfg['clients']
        event(status='evaluation_after_fit_saved')
        ids = [i for client_ids in manifest['partitions']['test'][task] for i in client_ids]
        assert len(ids) == len(set(ids)) == 1000
        phi, labels = extract(test, ids)
        test_phi.append(phi)
        test_labels.append(labels)
        test_indices.extend(ids)
        progress['test_reference_images'] += len(ids)
        features = torch.cat(test_phi).double()
        y = torch.cat(test_labels).numpy()
        path = src_run / f'task{task + 1}_private_logits.npz'
        identity = {'task': task, 'file': path.name, 'size': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
        with np.load(path) as z:
            assert np.array_equal(z['indices'], np.array(test_indices))
            assert np.array_equal(z['labels'], y)
            logits = torch.from_numpy(z['logits'].copy()).double()
        assert logits.shape == (len(y), classes)
        raw = summarize(logits.numpy(), y)
        confusion = np.bincount(y * classes + logits.argmax(1).numpy(), minlength=classes * classes).reshape(classes, classes)
        assert confusion.tolist() == source_metrics['stages'][task]['confusion']
        centered = center_task_logits(logits.numpy())
        assert summarize(centered, y) == source_metrics['stages'][task]['centered']
        source_identities.append(identity)
        evidence = evidence_scores(features, fits['bidirectional'])
        single = evidence_scores(features, fits['no_old_to_new'])
        scores = {
            'raw': logits, 'centered': torch.from_numpy(centered),
            'equal_task_support': compose_scores(logits),
            'reference_lda': evidence_scores(features, fits['reference_lda']),
            'evidence_alone': evidence,
            'boundary_fused': compose_scores(logits, evidence),
            'no_old_to_new_fused': compose_scores(logits, single),
        }
        arms = {key: summarize(value.numpy(), y) for key, value in scores.items()}
        for key in ('equal_task_support', 'boundary_fused', 'no_old_to_new_fused'):
            assert [p['restricted_accuracy'] for p in arms[key]['per_task']] == [p['restricted_accuracy'] for p in raw['per_task']]
        base_pred = scores['centered'].argmax(1).numpy()
        new_pred = scores['boundary_fused'].argmax(1).numpy()
        stage = {
            'task': task, 'classes': classes, 'arms': arms, 'solver': diagnostics,
            'class_counts': counts.cpu().tolist(), 'source_identity': identity,
            'paired_to_centered': {
                'corrected': int(((base_pred != y) & (new_pred == y)).sum()),
                'broken': int(((base_pred == y) & (new_pred != y)).sum()),
            },
            'evidence_scale': {'min': float(evidence.min()), 'max': float(evidence.max()),
                               'std': float(evidence.std())},
            'within_task_rankings_unchanged': True,
            'test_fit_or_true_task_inference': False,
        }
        stages.append(stage)
        checkpoint['phase'] = 'stage_complete'
        checkpoint['stages'] = stages
        save_tensor(out / 'latest.pt', checkpoint)
        # Private reproducibility evidence; never synced to public reports.
        np.savez(out / f'task{task + 1}_private_evaluation.npz', indices=np.array(test_indices), labels=y,
                 reference_features=features.numpy(), source_logits=logits.numpy(),
                 **{f'pred_{key}': value.argmax(1).numpy() for key, value in scores.items()})
        write_json(out / f'TASK{task + 1}.json', stage)
        progress['stages'] = [{'task': s['task'], 'fused': s['arms']['boundary_fused']['all_seen_accuracy'],
                               'centered': s['arms']['centered']['all_seen_accuracy']} for s in stages]
        event(status='stage_complete')
        del covariance, means, counts

    event(status='qualifying')
    assert [p._version for p in reference.parameters()] == parameter_versions
    assert all(not p.requires_grad and p.grad is None for p in reference.parameters())
    assert progress['train_reference_images'] == cfg['training_feature_visits'] == len(visited_ids)
    assert progress['test_reference_images'] == cfg['reference_test_feature_visits']
    assert len(message_ids) == len(progress['messages'])
    assert sum(m['n'] for m in progress['messages']) == 15000
    assert [s['n'] for s in archive] == [500] * 30
    for identity in source_identities:
        p = src_run / identity['file']
        assert (p.stat().st_size, p.stat().st_mtime_ns) == (identity['size'], identity['mtime_ns'])
    saved = torch.load(out / 'latest.pt', map_location='cpu', weights_only=False)
    assert saved['clock'] == clock and saved['task'] == 2 and saved['phase'] == 'stage_complete'
    with np.load(out / 'task3_private_evaluation.npz') as final:
        phi = torch.from_numpy(final['reference_features'])
        z = torch.from_numpy(final['source_logits'])
        restored_evidence = evidence_scores(phi, saved['fits']['bidirectional'])
        restored_single = evidence_scores(phi, saved['fits']['no_old_to_new'])
        restored = {
            'raw': z, 'centered': torch.from_numpy(center_task_logits(z.numpy())),
            'equal_task_support': compose_scores(z),
            'reference_lda': evidence_scores(phi, saved['fits']['reference_lda']),
            'evidence_alone': restored_evidence, 'boundary_fused': compose_scores(z, restored_evidence),
            'no_old_to_new_fused': compose_scores(z, restored_single),
        }
        for key, value in restored.items():
            assert np.array_equal(value.argmax(1).numpy(), final[f'pred_{key}'])
            assert summarize(value.numpy(), final['labels']) == stages[-1]['arms'][key]
    summary = {key: summarize_matrix(stages, key) for key in stages[-1]['arms']}
    qualification = {
        'qualified': True, 'tasks': 3, 'seed': 0, 'train_reference_unique_images': len(visited_ids),
        'test_reference_images': progress['test_reference_images'], 'stage_evaluation_images': 6000,
        'final_saved_evidence_reload_images': 3000, 'all_seven_arms_reload_predictions_equal': True,
        'no_adapted_model_retraining': True, 'frozen_reference_unchanged': True,
        'no_future_checkpoint': True, 'source_logits_unchanged_size_mtime': True,
        'source_raw_and_centered_metrics_reproduced': True, 'within_task_rankings_unchanged': True,
        'class_counts': [s['n'] for s in archive], 'unique_statistic_messages': len(message_ids),
        'solver_all_converged': all_solver_converged, 'clock': clock,
    }
    cost = {
        'reference_train_forwards': progress['train_reference_images'],
        'reference_test_forwards': progress['test_reference_images'],
        'adapted_model_forward_or_training_added': 0,
        'source_training_is_reused_sunk_cost': True,
        'logical_statistics_upload_bytes': progress['logical_statistics_upload_bytes'],
        'logical_primary_evidence_broadcast_bytes': progress['logical_evidence_broadcast_bytes'],
        'message_count': len(message_ids), 'fp64_statistics': True,
        'worker_wall_seconds': time.time() - clock['started'],
        'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'allocated_root_bytes': size(root),
        'deployment': 'Requires frozen and adapted forward passes; offline logits reuse does not eliminate deployment cost.',
        'reference_LDA_and_ablation': 'Diagnostics computed server-side from same uploaded moments; no extra client extraction.',
    }
    result = {'protocol': cfg['protocol'], 'stages': stages, 'summary': summary,
              'qualification': qualification, 'cost': cost,
              'decision': decision(stages, summary, all_solver_converged)}
    write_json(out / 'RESULTS.json', result)
    write_json(out / 'QUALIFICATION.json', qualification)
    write_json(out / 'COMPLETE.json', {'completed': time.time(), 'qualified': True, 'solver_all_converged': all_solver_converged})
    event(status='completed')


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        root = Path(os.environ['PROJECT_ROOT'])
        write_json(root / 'reports/WORKER_FAILURE.json', {'time': time.time(), 'error': repr(exc), 'no_automatic_retry': True})
        raise
