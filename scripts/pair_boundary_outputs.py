"""Private server-side same-ID comparison; writes aggregate counts only."""
import json,os
from pathlib import Path
import numpy as np
from research.basic_repair.fcl_basic_repair_tools import center_task_logits

root=Path(os.environ['PROJECT_ROOT'])
previous=Path(os.environ['PREVIOUS_OUTPUT'])
new_output=Path(os.environ['CURRENT_OUTPUT'])
result={}
for task in range(1,int(os.environ.get('TASK_COUNT','10'))+1):
    with np.load(new_output/f'task{task}_private_logits.npz') as f:new={k:f[k].copy() for k in f.files}
    with np.load(previous/f'task{task}_private_logits.npz') as f:old={k:f[k].copy() for k in f.files}
    ni=np.argsort(new['indices']);oi=np.argsort(old['indices'])
    assert np.array_equal(new['indices'][ni],old['indices'][oi])
    assert np.array_equal(new['labels'][ni],old['labels'][oi])
    labels=new['labels'][ni];rows={}
    for rule in ('raw','centered'):
        a=new['logits'][ni];b=old['logits'][oi]
        if rule=='centered':a=center_task_logits(a,10);b=center_task_logits(b,10)
        pn=a.argmax(1);po=b.argmax(1);rn=pn==labels;ro=po==labels
        def stats(mask):
            return {'n':int(mask.sum()),'changed_predictions':int(((pn!=po)&mask).sum()),
                    'corrected':int((~ro&rn&mask).sum()),'broken':int((ro&~rn&mask).sum()),
                    'net_correct':int(((rn.astype(int)-ro.astype(int))*mask).sum()),
                    'both_wrong_but_changed':int((~ro&~rn&(pn!=po)&mask).sum())}
        rows[rule]={'all':stats(np.ones(len(labels),dtype=bool)),
                    'old':stats(labels<(task-1)*10),'new':stats(labels>=(task-1)*10)}
    result[str(task)]=rows
(root/os.environ['PAIR_REPORT']).write_text(json.dumps(result,indent=2)+'\n')
