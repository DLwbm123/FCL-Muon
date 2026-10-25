"""Single prespecified intervention on saved logits; no model or fitting."""
import json
import os
from pathlib import Path
import numpy as np
from research.basic_repair import fcl_basic_repair_tools as ref

root=Path(os.environ['PROJECT_ROOT'])
parent=Path(os.environ['EXPERIMENT_ROOT'])
out=root/'reports/task_block_centering.json'
if out.exists():raise RuntimeError('Refusing duplicate intervention')
rows={}
expected={'mulora_fl':.6716666666666666,'mulora_fl_fedmuon':.4676666666666667,'lorm_official':.8873333333333333}
for method in expected:
    path=parent/'runs/diagnostic_pilot_checkpoint_20261025'/f'{method}_private_logits.npz'
    digest=ref._hash(path) # explicitly requested by the supplied repair protocol
    with np.load(path,allow_pickle=False) as a:
        scores,y,ids=a['logits'],a['labels'],a['indices']
    assert len(set(ids))==len(ids)==3000
    original=ref.summarize(scores,y)
    assert original['all_seen_accuracy']==expected[method]
    centered=ref.center_task_logits(scores)
    changed=ref.summarize(centered,y)
    assert [v['restricted_accuracy'] for v in original['per_task']]==[v['restricted_accuracy'] for v in changed['per_task']]
    assert ref._hash(path)==digest
    rows[method]={'input_sha256':digest,'original':original,'centered':changed,
        'class6_to7_before':int(((y==6)&(scores.argmax(1)==7)).sum()),
        'class6_to7_after':int(((y==6)&(centered.argmax(1)==7)).sum())}
out.write_text(json.dumps({'self_test':ref.self_test(),'rule':'subtract every fixed task block mean; no sample task identity','no_parameters_fit':True,'methods':rows},indent=2))
print({k:(v['original']['all_seen_accuracy'],v['centered']['all_seen_accuracy']) for k,v in rows.items()},flush=True)
