"""Engineering qualification only. Never selects hyperparameters on test scores."""
import json,math
from pathlib import Path


def qualify(root,cfg):
    out=Path(root)/cfg['output']
    if not (out/'COMPLETE.json').exists():raise ValueError('Missing completion receipt')
    events=[json.loads(line) for line in (out/'events.jsonl').read_text().splitlines()]
    # Failed/incomplete attempts remain in logs; accept last committed identity.
    records={(e['task'],e['round'],e['client']):e for e in events if e['kind']=='client'}
    client=list(records.values())
    expected=cfg['tasks']*cfg['rounds']*cfg['clients']
    assert len(client)==expected,(len(client),expected)
    for e in client:
        assert e['steps']>0 and math.isfinite(e['loss'])
        assert e['local_cumulative_B']['fro']>0 and math.isfinite(e['local_cumulative_B']['fro'])
        assert all(math.isfinite(v) for v in e['lr'])
        if not cfg.get('train_batches'):
            assert e['steps']==cfg['epochs']*math.ceil(e['N']/cfg['batch'])
            assert e['visited']==cfg['epochs']*e['N']
    m=json.loads((out/'metrics.json').read_text())
    assert len(m['matrix'])==cfg['tasks']
    for i,row in enumerate(m['matrix']):
        assert len(row)==i+1 and all(math.isfinite(v) and 0<=v<=1 for v in row)
    result={'qualified':True,'purpose':'numerical/completion qualification, no test-score ranking',
            'client_records':len(client),'final_micro':m['final_micro'],'old_micro':m['old_micro'],
            'new_accuracy':m['new_accuracy'],'average_forgetting':m['average_forgetting']}
    (out/'QUALIFICATION.json').write_text(json.dumps(result,indent=2))
    return result
