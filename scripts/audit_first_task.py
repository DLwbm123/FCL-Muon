"""Read logs/checkpoint tensors and run synthetic arithmetic tests; no image IO."""
import json
import math
import os
import time
from pathlib import Path

import torch


def spread(values):
    t = torch.tensor(values, dtype=torch.float64)
    return {'min': float(t.min()), 'median': float(t.median()), 'max': float(t.max()), 'mean': float(t.mean())}


def main():
    import pytest
    from fcl import math as deployed_math
    root = Path(os.environ['EXPERIMENT_ROOT'])
    out = Path(os.environ['PROJECT_ROOT']) / 'results'
    out.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    started = time.time()
    result = {'scope': 'Existing pilot logs/checkpoint tensors plus synthetic CPU arithmetic; no real-data evaluation, training or tuning',
              'deployed_math': str(Path(deployed_math.__file__).relative_to(root)), 'methods': {}}
    logmap = {}
    for method in ('mulora_fl', 'mulora_fl_fedmuon', 'lorm_official'):
        folder = root / 'runs' / ('pilot_'+method)
        events = [json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
        logmap[method] = [e for e in events if e['kind']=='client' and e['task']==0]
        checkpoint = folder/'latest.pt'
        before = (checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns)
        s = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert s['task']==3 and s['round']==0 and not s['initialized']
        row = {'checkpoint_files': [p.name for p in folder.glob('*.pt')],
               'checkpoint_stage': s['task'], 'task1_accuracy': s['matrix'][0][0],
               'config': s['config'], 'client_head_snapshots_available': False,
               'task1_full_checkpoint_available': False,
               'saved_round_evaluations_task1': [{'round': e['round']+1, 'accuracy': e['all_seen_micro']}
                    for e in events if e['kind']=='round_evaluation' and e['task']==0]}
        if method != 'lorm_official':
            w = s['model']['rows.0'].double()
            norms = w.norm(dim=1)
            unit = torch.nn.functional.normalize(w, dim=1)
            cos = unit@unit.T
            mask = ~torch.eye(len(w), dtype=torch.bool)
            row['retained_task1_head'] = {'norms': norms.tolist(), 'norm_summary': spread(norms.tolist()),
                'max_min_norm_ratio': float(norms.max()/norms.min()),
                'pairwise_off_diagonal_cosine': spread(cos[mask].tolist()),
                'finite': bool(torch.isfinite(w).all()), 'zero_rows': int((norms==0).sum()),
                'provenance': 'rows.0 freezes at start of task2; only current row block is optimized/aggregated; merge affects backbone only'}
        row['rounds'] = []
        for rnd in range(5):
            clients = [e for e in logmap[method] if e['round']==rnd]
            rr = {'round': rnd+1, 'client_count': len(clients),
                  'weighted_local_process_ce': sum(e['loss']*e['visited'] for e in clients)/sum(e['visited'] for e in clients),
                  'sample_counts': [e['N'] for e in clients], 'steps': [e['steps'] for e in clients]}
            if method != 'lorm_official':
                rr['first_step_all_24_matrices_fro'] = spread([d['fro'] for e in clients for d in e['first_step_delta_B_all_matrices']])
                rr['learning_rate_range'] = [min(min(e['lr']) for e in clients), max(max(e['lr']) for e in clients)]
                agg = next(e for e in events if e['kind']=='round' and e['task']==0 and e['round']==rnd)
                rr['aggregate_B_all_24_matrices_fro'] = [d['fro'] for d in agg['aggregate_B_all_matrices']]
            row['rounds'].append(rr)
        row['checkpoint_size_mtime_unchanged'] = before == (checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns)
        assert row['checkpoint_size_mtime_unchanged']
        result['methods'][method] = row
        del s
    a,b = logmap['mulora_fl'], logmap['mulora_fl_fedmuon']
    a = sorted([e for e in a if e['round']==0], key=lambda e:e['client'])
    b = sorted([e for e in b if e['round']==0], key=lambda e:e['client'])
    ratios = [y['fro']/x['fro'] for ca,cb in zip(a,b) for x,y in zip(ca['first_step_delta_B_all_matrices'],cb['first_step_delta_B_all_matrices'])]
    result['task1_round1_logged_first_step_ratio_fedmuon_over_mulora'] = {'count': len(ratios), 'summary': spread(ratios), 'expected_sqrt768': math.sqrt(768), 'max_abs_error': max(abs(r-math.sqrt(768)) for r in ratios), 'scope': '240 norm pairs only, not full gradient/direction tensors'}
    # Source operator check on actual target shape, not a full-model experiment.
    import sys
    sys.path.insert(0, str(Path(os.environ['PROJECT_ROOT'])/'tests'))
    from test_first_task_audit import official_operator
    torch.manual_seed(72)
    g = torch.randn(768,32)
    actual, expected = deployed_math.muon_operator(g), official_operator()(g)
    result['official_operator_check'] = {'shape': list(g.shape), 'max_abs_error': float((actual-expected).abs().max()),
        'relative_fro_error': float((actual-expected).norm()/expected.norm()),
        'same_gradient_scaled_over_unscaled_norm_ratio': float(actual.norm()/deployed_math.mulora_ssb(g).norm()),
        'base_lr': .0002, 'lr_times_dimension_multiplier': .0002*math.sqrt(768),
        'warning': 'The multiplied scalar is not a measured spectral norm; finite NS has nonunit singular values'}
    code = pytest.main(['-q', 'tests/test_first_task_audit.py', '--junitxml='+str(out/'tests.xml')])
    result['test_exit_code'] = code
    result['elapsed_seconds'] = time.time()-started
    (out/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'test_exit_code': code, 'elapsed_seconds': result['elapsed_seconds'], 'output': str(out)}, indent=2))
    raise SystemExit(code)


if __name__ == '__main__':
    main()
