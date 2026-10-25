"""Summarize existing pilot logs only; no training, model loading, or evaluation."""
import json
import os
from pathlib import Path

root = Path(os.environ.get('PROJECT_ROOT', '.'))
result = {'scope': 'Read-only existing logs; no new experimental measurements', 'methods': {}}
for method in ('mulora_fl', 'mulora_fl_fedmuon', 'lorm_official'):
    events = [json.loads(line) for line in (root / 'runs' / ('pilot_' + method) / 'events.jsonl').read_text().splitlines()]
    tasks = []
    for task in range(3):
        rounds = []
        for rnd in range(5):
            clients = [e for e in events if e['kind'] == 'client' and e['task'] == task and e['round'] == rnd]
            visited = sum(e['visited'] for e in clients)
            row = {'round': rnd + 1, 'sample_weighted_local_train_ce': sum(e['loss'] * e['visited'] for e in clients) / visited,
                   'client_steps_min_max': [min(e['steps'] for e in clients), max(e['steps'] for e in clients)],
                   'total_steps': sum(e['steps'] for e in clients)}
            if method != 'lorm_official':
                row['first_matrix_first_step_fro_client_mean'] = sum(e['first_delta_B']['fro'] for e in clients) / len(clients)
                agg = next(e for e in events if e['kind'] == 'round' and e['task'] == task and e['round'] == rnd)
                row['first_matrix_aggregate_B_fro'] = agg['aggregate_B']['fro']
            rounds.append(row)
        tasks.append({'task': task + 1, 'rounds': rounds})
    rrs = [e for e in events if e['kind'] == 'rrs']
    result['methods'][method] = {'tasks': tasks, 'rrs_records': len(rrs),
                                'rrs_jitter_escalations': sum(e['jitter'] > e['epsilon'] * 1.01 for e in rrs)}
result['norm_scope'] = 'First Q matrix only; not a claim about all layers or every update'
(root / 'reports/PILOT_LOG_AUDIT.json').write_text(json.dumps(result, indent=2) + '\n')
print('Saved existing-log audit for three completed pilots.')
