"""Read-only artifact audit and sanitized summary for the completed paired study."""
import json
import os
import statistics
import time
from pathlib import Path
import torch
from research.continual_muon.paired_metrics import conditions


def main():
    started = time.time()
    root = Path(os.environ['PROJECT_ROOT']);reports = root / 'reports'
    job = json.loads((reports / 'job.json').read_text())
    launch = json.loads((reports / 'LAUNCH.json').read_text())
    if job['status'] != 'completed' or job['worker_exit_code'] != 0:
        raise ValueError('worker did not complete successfully')
    if job['ended'] > job['deadline'] or job['allocated_bytes'] >= 6_000_000_000:
        raise ValueError('original budget was exceeded')
    for pid in (launch['timeout_pid'], job['supervisor_pid'], job['worker_pid']):
        path = Path('/proc') / str(pid)
        if path.exists() and b'PROJECT_ROOT=' + str(root).encode() in (path / 'environ').read_bytes().split(b'\0'):
            raise ValueError('recorded study process still active')
    first = json.loads((reports / 'CPU_TESTS.json').read_text())
    revised = json.loads((reports / 'CPU_AMENDMENT_TESTS.json').read_text())
    gpu = json.loads((reports / 'REAL_PREFLIGHT.json').read_text())
    if first['passed'] or first['failures'] != 1 or not revised['passed'] or revised['tests'] != 22 or not gpu['passed']:
        raise ValueError('preflight history mismatch')
    result = json.loads((reports / 'RESULT.json').read_text())
    if result['gate_conditions'] != conditions(result['comparisons']) or result['exploratory_gate_passed'] != all(result['gate_conditions'].values()):
        raise ValueError('science conditions mismatch')
    if (result['new_training_visits'] != 500000 or result['new_training_statistics_forward_images'] != 110000 or
            result['training_side_image_reads'] != {'1': 155000, '2': 455292} or
            result['test_images_used'] != 11000 or not result['test_dataset_opened_after_all_training']):
        raise ValueError('reported visit counts do not match protocol')
    probe = result['probe_costs']
    if sum(v['student_forward_images'] for v in probe) != 292 or sum(v['student_backward_images'] for v in probe) != 292:
        raise ValueError('probe accounting mismatch')
    receipt = json.loads((reports / 'PRIVATE_GATE_RECORDS.json').read_text())
    records = receipt['guarded']
    if len(records) != 10:
        raise ValueError('missing guarded client records')
    accepted = sum(bool(v['accepted']) for v in records.values())
    retention = [float(v['retained_energy']) for v in records.values() if v['retained_energy'] is not None]
    cosines = [float(v['probe_cosine']) for v in records.values() if v['probe_cosine'] is not None]
    snapshot = torch.load(root / 'runs/t2_private_state.pt', map_location='cpu', weights_only=False)
    if (snapshot['schema'] != 'paired-private-t2-boundary-v1' or len(snapshot['momenta']) != 10 or
            len(snapshot['old_bases']) != 12 or snapshot['clock'] != {k: job[k] for k in ('started', 'deadline')}):
        raise ValueError('invalid saved T2 boundary')
    nonzero = sum(any(bool(torch.count_nonzero(m)) for m in values) for values in snapshot['momenta'].values())
    checkpoint_summary = {};baseline = None
    for mode in ('zero', 'projected', 'guarded'):
        x = torch.load(root / 'runs' / mode / 'latest.pt', map_location='cpu', weights_only=False)
        state = x['state']
        records_t3 = [v for v in state['training_records'] if v['task'] == 2]
        if ((state['task'], state['round'], state['phase']) != (2, 5, 'ready_evaluation') or
                x['head_count'] != 3 or len(records_t3) != 50 or sum(v['visited'] for v in records_t3) != 125000 or
                x['clock'] != snapshot['clock'] or x['identity'] != snapshot['identity']):
            raise ValueError('invalid T3 checkpoint')
        matrices = {k: v for k, v in x['model'].items() if k.endswith(('.Bq', '.Bv'))}
        if len(matrices) != 24:
            raise ValueError('unexpected B matrix count')
        checkpoint_summary[mode] = {'t3_training_visits': 125000, 't3_client_rounds': 50}
        if baseline is None:
            baseline = {k: v.clone() for k, v in matrices.items()}
        else:
            if baseline.keys() != matrices.keys():
                raise ValueError('unmatched B matrix keys')
            deltas = [matrices[k].double() - baseline[k].double() for k in baseline]
            checkpoint_summary[mode]['B_l2_difference_vs_zero'] = float(sum(float((v*v).sum()) for v in deltas)**.5)
            checkpoint_summary[mode]['B_matrices_different'] = sum(bool(torch.count_nonzero(v)) for v in deltas)
        del x, matrices
    summary = {'status': 'completed', 'scientific_pass': result['exploratory_gate_passed'],
        'gate_conditions': result['gate_conditions'], 'comparisons': result['comparisons'],
        'guarded_accepted_clients': accepted, 'guarded_total_clients': len(records),
        'retained_energy_client_mean': statistics.mean(retention) if retention else None,
        'probe_cosine_min': min(cosines) if cosines else None, 'probe_cosine_max': max(cosines) if cosines else None,
        'snapshot_nonzero_clients': nonzero, 'checkpoint_audit': checkpoint_summary,
        'new_training_visits': 500000, 'new_training_statistics_forward_images': 110000,
        'probe_student_forward_images': 292, 'probe_student_backward_images': 292,
        'test_forward_images': 11000, 'training_side_image_reads': result['training_side_image_reads'],
        'source_t1_training_visits': result['source_t1_training_visits'],
        'shared_logical_communication_bytes': result['shared_logical_communication_bytes'],
        'per_branch_logical_communication_bytes': result['per_branch_logical_communication_bytes'],
        'preflight_history': {'first_cpu_tests': first['tests'], 'first_cpu_failures': first['failures'],
                              'amended_cpu_tests': revised['tests'], 'amended_cpu_passed': revised['passed'],
                              'gpu_ssb_cpu_l2': gpu['cpu_gpu_ssb_l2'], 'gpu_preflight_image_accesses': 0},
        'original_elapsed_seconds': job['ended'] - job['started'], 'allocated_bytes_at_completion': job['allocated_bytes'],
        'audit_cpu_elapsed_seconds': time.time() - started,
        'limitations': ['single seed, three tasks, component study; no significance or generality claim',
                        'zero corrected/broken counts do not establish identical per-image predictions',
                        'server-side historical Muon aggregation was not implemented in this experiment',
                        'test counts follow original evaluation assertions; no independent image-read counter was saved',
                        'probe and state transfer are simulated locally; no network or privacy guarantee']}
    path = reports / 'SANITIZED_SUMMARY.json'
    if path.exists():
        raise ValueError('summary already exists')
    path.write_text(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
