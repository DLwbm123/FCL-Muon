"""Freeze split counts before any new local training or candidate evaluation."""
import json,os,time
from pathlib import Path
import torch
from research.fed_spectra.heldout_round import split_clients
from research.fed_spectra.lifecycle import write
root=Path(os.environ['PROJECT_ROOT']);origin=json.loads((root/'reports/ORIGIN.json').read_text())
assert time.time()<origin['started']+6900
assert json.loads((root/'reports/CPU_TESTS.json').read_text())['passed']
for seed,path in enumerate(json.loads((root/'sources_private.json').read_text())):
    source=Path(path);x=torch.load(source/'runs/control/task3.pt',map_location='cpu',weights_only=False)
    assert x['config']['seed']==seed and x['state']['task']==3 and x['state']['phase']=='task_start' and x['head_count']==3
    assert len(x['state']['training_records'])==150
    m=json.loads((source/x['config']['manifest']).read_text());train,validation=split_clients(m['partitions']['train'][3],seed)
    n=sum(map(len,train));row={'seed':seed,'train_counts':list(map(len,train)),'validation_counts':list(map(len,validation)),
        'expected_new_training_visits':5*n,'validation_weight_coverage':sum(len(g) for g,v in zip(train,validation) if v)/n}
    assert all(train) and row['validation_weight_coverage']>=.95
    target=root/f'reports/SPLIT_seed{seed}.json';assert not target.exists();write(target,row)
    print(json.dumps(row),flush=True);del x,m
