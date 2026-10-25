import json,os
from pathlib import Path
import torch
root=Path(os.environ['PROJECT_ROOT'])
rows=[]
for out in sorted((root/'runs').glob('smoke_*')):
 if not (out/'COMPLETE.json').exists():continue
 s=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
 assert s['task']==2 and not s['initialized'] and s['history'].committed=={0,1}
 assert not s['pending'] and not s['client_states'] and not s['server_c']
 for key,a in s['model'].items():
  if key.endswith('.A'):torch.testing.assert_close(a@a.T,torch.eye(a.shape[0]),atol=3e-6,rtol=1e-5)
  if key.endswith('.Bq') or key.endswith('.Bv'):assert not torch.count_nonzero(a)
 events=[json.loads(x) for x in (out/'events.jsonl').read_text().splitlines()]
 receipt=json.loads((out/'COMPLETE.json').read_text());m=json.loads((out/'metrics.json').read_text())
 rows.append({'run':out.name,'elapsed_seconds':receipt['elapsed'],'peak_gpu_bytes':max(e['peak_gpu_bytes'] for e in events if e['kind']=='round'),
              'checkpoint_bytes':(out/'latest.pt').stat().st_size,'manifest_hash':s['manifest_hash'],'core_task_boundary_audit':'passed',
              'metrics':m,'warning':'tiny real-data smoke, not scientific performance evidence'})
assert len(rows)==2 and rows[0]['manifest_hash']==rows[1]['manifest_hash']
(root/'reports/smoke_summary.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
