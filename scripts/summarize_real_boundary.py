"""Recover aggregate observations after the original result serialization failure."""
import json
import math
import os
import statistics
import time
from pathlib import Path
import torch


def main():
    root = Path(os.environ['PROJECT_ROOT'])
    source = Path(os.environ['SOURCE_ROOT'])
    report = root / 'reports'
    job = json.loads((report / 'job.json').read_text())
    if job['status'] != 'worker_failed' or job['worker_exit_code'] != 1:
        raise ValueError('not the recorded failed run')
    if (report / 'RESULT.json').exists() or (report / 'RECOVERED_RESULT.json').exists():
        raise ValueError('unexpected existing result')
    log = (report / 'worker_recovery1.log').read_text()
    if "Object of type bool_ is not JSON serializable" not in log or "reports/RESULT.json" not in log:
        raise ValueError('unexpected failure; cannot reconstruct')
    p = json.loads((report / 'PRIVATE_CLIENT_METRICS.json').read_text())
    rows, proxies = p['rows'], p['proxies']
    if len(rows) != 10 or len(proxies) != 24 or any(len(row['matrices']) != 24 for row in rows):
        raise ValueError('incomplete private numeric observations')
    x = torch.load(root / 'runs/diagnostic/latest.pt', map_location='cpu', weights_only=False)
    old = torch.load(source / 'runs/prefix/latest.pt', map_location='cpu', weights_only=False)
    state = x['state']
    if (state['task'], state['round'], state['phase']) != (2, 0, 'round_start'):
        raise ValueError('unexpected final diagnostic checkpoint')
    if state['merges'] != [0, 1] or state['history'].committed != {0, 1}:
        raise ValueError('unexpected task-boundary history')
    records = state['training_records']
    if records[:len(old['state']['training_records'])] != old['state']['training_records']:
        raise ValueError('source training records changed')
    visits = sum(v['visited'] for v in records if v['task'] == 1)
    stats = state['stats_records'][len(old['state']['stats_records']):]
    if visits != 125000 or len(records) - len(old['state']['training_records']) != 50:
        raise ValueError('unexpected T2 training cost')
    if sum(v['samples'] for v in stats) != 35000:
        raise ValueError('unexpected statistics cost')
    if any(torch.count_nonzero(m) for client in state['client_states'].values() for m in client['m']):
        raise ValueError('T3 original zero-state initialization not preserved')
    cfg = x['config']
    manifest = json.loads((source / cfg['manifest']).read_text())
    sizes = [len(ids) for ids in manifest['partitions']['train'][1]]
    if len(sizes) != 10 or min(sizes) <= 0 or sum(sizes) != 5000:
        raise ValueError('unexpected source client weights')
    valid = []
    for k, row in enumerate(rows):
        original = sum(m['source_norm_squared'] for m in row['matrices'])
        projected = sum(m['projected_norm_squared'] for m in row['matrices'])
        expected = projected / original if original else None
        if expected is None:
            if row['energy_retained'] is not None:
                raise ValueError('invalid zero-state ratio')
        elif not math.isclose(expected, row['energy_retained'], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('invalid recorded client energy')
        else:
            valid.append((k, expected))
    valid_weight = sum(sizes[k] for k, _ in valid)
    weighted = sum(sizes[k] * v for k, v in valid) / valid_weight if valid_weight else None
    ordered = sorted(v for _, v in valid)
    median = statistics.median(ordered) if ordered else None
    max_pyth = max(m['pythagorean_relative_error'] for row in rows for m in row['matrices'])
    max_orth = max(m['residual_orthogonality_error'] for row in rows for m in row['matrices'])
    geometry = bool(max_pyth <= 1e-5 and max_orth <= 1e-5)
    signal = bool(geometry and len(valid) >= 8 and weighted is not None and weighted >= .05 and median >= .01)
    progress = json.loads((report / 'PROGRESS.json').read_text())
    if (progress['task'], progress['round'], progress['phase'], progress['reads']) != (2, 5, 'ready_evaluation', {'1': 155000, '2': 0}):
        raise ValueError('unexpected T2 access receipt')
    previous_comm = old['state']['communication']
    result = {'status': 'recovered_after_worker_failure', 'original_worker_status': job['status'],
              'original_worker_exit_code': job['worker_exit_code'], 'engineering_passed': False,
              'failure': 'NumPy bool_ could not be serialized to the final JSON result',
              'geometric_checks_passed': geometry, 'exploratory_signal_conditions_met': signal,
              'seed': 0, 'new_training_visits': visits, 'new_stats_forward_images': 35000,
              'source_t1_training_visits': sum(v['visited'] for v in old['state']['training_records']),
              't2_loader_read_count_recorded': 155000,
              't3_loader_read_count': 5000,
              't3_read_count_evidence': 'the original 5000-read assertion executed before private metric write; no separate counter receipt was saved',
              'test_images_used': 0, 'old_images_used': 0, 'new_task_training_visits': 0,
              'weighted_client_retention': weighted, 'retention_sorted': ordered,
              'median_client_retention': median, 'valid_client_weight': valid_weight / 5000,
              'zero_energy_clients': 10 - len(valid), 'maximum_pythagorean_error': max_pyth,
              'maximum_residual_orthogonality_error': max_orth,
              'proxy_difference_norms': [v['difference_norm'] for v in proxies],
              'proxy_relative_differences': [v['relative_difference'] for v in proxies],
              'new_logical_communication_bytes': {k: v - previous_comm.get(k, 0) for k, v in state['communication'].items()},
              'original_elapsed_seconds': job['ended'] - job['started'], 'allocated_bytes': job['allocated_bytes'],
              'audit_elapsed_seconds': time.time() - audit_start,
              'limitations': ['original frozen engineering gate failed; recovered observations are exploratory',
                              'no evidence for T3 performance or old-class safety',
                              'last local momenta proxy is not full training trajectory bias']}
    tmp = report / 'RECOVERED_RESULT.tmp'
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False))
    tmp.replace(report / 'RECOVERED_RESULT.json')


if __name__ == '__main__':
    audit_start = time.time()
    main()
