"""Publish only completed pilot results; never rank partial runs as final."""
import json,os
from pathlib import Path
root=Path(os.environ.get('PROJECT_ROOT','.'))
audit_path=root/'reports/pilot_completion_audit.json'
audit=json.loads(audit_path.read_text()) if audit_path.exists() else {}
result={'scope':'CIFAR100 first3 tasks, seed0 alpha.05 n10, shared manifest; standard-method configurations',
        'warning':'Single seed; LoRM differs in rank, module positions, head, precision, optimizer and statistics overhead.',
        'methods':{}}
for method in ['mulora_fl','mulora_fl_fedmuon','lorm_official']:
    p=root/'runs'/('pilot_'+method)
    if not (p/'COMPLETE.json').exists() or not audit.get(method,{}).get('qualified'):
        result['methods'][method]={'status':'incomplete','final_metrics':None};continue
    metrics=json.loads((p/'metrics.json').read_text())
    complete=json.loads((p/'COMPLETE.json').read_text())
    matrix=metrics['matrix'];assert len(matrix)==3
    result['methods'][method]={'status':'completed','final_metrics':{k:metrics[k] for k in ['final_micro','new_accuracy','old_micro','average_forgetting']},
                              'completion_evidence':audit[method],
                              'stage_new_accuracy':[row[i] for i,row in enumerate(matrix)],
                              'matrix':matrix,'elapsed_seconds':complete['elapsed']}
if result['methods']['lorm_official']['status']=='completed':
    ref=result['methods']['lorm_official']['final_metrics']
    for method in ['mulora_fl','mulora_fl_fedmuon']:
        entry=result['methods'][method]
        if entry['status']=='completed':
            entry['difference_vs_lorm_percentage_points']={k:100*(entry['final_metrics'][k]-v) for k,v in ref.items()}
(root/'reports/pilot_comparison.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
