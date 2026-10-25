"""Offline completion audit; reads saved state only, never evaluates a model."""
import json
import math
import os
from pathlib import Path

import torch

root = Path(os.environ.get('PROJECT_ROOT', '.'))
queue = json.loads((root / 'reports/queue-state.json').read_text())
lorm = json.loads((root / 'reports/lorm-state.json').read_text())
identities = set()
report = {}
for method in ['mulora_fl', 'mulora_fl_fedmuon', 'lorm_official']:
    out = root / 'runs' / ('pilot_' + method)
    receipt = json.loads((out / 'COMPLETE.json').read_text())
    cfg = receipt['config']
    state = lorm if method == 'lorm_official' else queue['configs/pilot_' + method + '.json']
    assert state['status'] == 'completed' and state['exit_code'] == 0
    assert state['ended'] - state['started'] < 86400
    events = [json.loads(line) for line in (out / 'events.jsonl').read_text().splitlines()]
    records = [e for e in events if e['kind'] == 'client']
    assert len(records) == cfg['tasks'] * cfg['rounds'] * cfg['clients'] == 150
    assert len({(e['task'], e['round'], e['client']) for e in records}) == 150
    for e in records:
        assert e['visited'] == cfg['epochs'] * e['N']
        assert e['steps'] == cfg['epochs'] * math.ceil(e['N'] / cfg['batch'])
        assert math.isfinite(e['loss'])
        if method == 'lorm_official':
            assert e['train_matrix'] == ('B' if e['round'] % 2 == 0 else 'A')
    metrics = json.loads((out / 'metrics.json').read_text())
    matrix = metrics['matrix']
    assert len(matrix) == 3
    assert all(len(row) == i + 1 and all(math.isfinite(v) and 0 <= v <= 1 for v in row)
               for i, row in enumerate(matrix))
    forgetting = sum(max(matrix[j][i] for j in range(i, 3)) - matrix[-1][i] for i in range(2)) / 2
    assert abs(forgetting - metrics['average_forgetting']) < 1e-12
    saved = torch.load(out / 'latest.pt', map_location='cpu', weights_only=False)
    assert saved['task'] == 3 and saved['round'] == 0 and not saved['initialized']
    assert saved['matrix'] == matrix and saved['config'] == cfg
    identities.add((saved['manifest_hash'], saved['weight_hash']))
    report[method] = {'qualified': True, 'client_records': len(records), 'exit_code': 0,
                      'checkpoint_task': saved['task'], 'checkpoint_round': saved['round'],
                      'initialized': saved['initialized'], 'matrix_matches_checkpoint': True,
                      'purpose': 'Offline saved-state completion and full-epoch budget audit; no new model evaluation'}
    # The original LoRM supervisor was loaded before the receipt-writing addition.
    # Record this later audit explicitly, without rewriting its completion time.
    if method == 'lorm_official' and not (out / 'QUALIFICATION.json').exists():
        (out / 'QUALIFICATION.json').write_text(json.dumps(report[method], indent=2))
    del saved
assert len(identities) == 1
report['shared_manifest_and_weights'] = True
(root / 'reports/pilot_completion_audit.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
