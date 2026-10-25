"""Original-lineage clock and storage guard, no automatic retry."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
import torch
root=Path(os.environ['PROJECT_ROOT']);original=Path(os.environ['EXPERIMENT_ROOT']);state=root/'reports/job.json'
if state.exists():raise SystemExit('Existing continuation receipt: never duplicate')
cfg=json.loads((root/'research/head_regmean/PILOT_PROTOCOL.json').read_text())
initial=torch.load(original/cfg['original_R1_boundary'],map_location='cpu',weights_only=False)
head=torch.load(original/cfg['source_checkpoint'],map_location='cpu',weights_only=False)
parent_start=json.loads((original/'basic_repair_20261025/reports/job.json').read_text())['started']
started=min(parent_start,initial['original_start'],head['original_start']);del initial,head
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','-i',os.environ['CUDA_VISIBLE_DEVICES']],text=True).strip())
if free<14000:raise SystemExit('Insufficient GPU free memory')
lineage=[root,original/'basic_repair_20261025',original/'head_regmean_round_20261025']
def size(path):return sum(p.stat().st_blocks*512 for p in path.rglob('*') if p.is_file() and not p.is_symlink())
s={'started':started,'deadline':started+cfg['hard_timeout_from_earliest_source_seconds'],
    'continuation_launched':time.time(),'status':'running','gpu':os.environ['CUDA_VISIBLE_DEVICES'],
    'gpu_free_mib':free,'new_storage_limit':cfg['new_storage_guard_bytes'],
    'lineage_storage_limit':cfg['lineage_storage_guard_bytes'],'lineage_before_bytes':sum(size(p) for p in lineage),
    'previous_head_phase_deadline':json.loads((original/'head_regmean_round_20261025/reports/job.json').read_text())['deadline']}
if time.time()>=s['deadline']:raise SystemExit('Original lineage clock exhausted')
def record():
    tmp=state.with_suffix('.tmp');tmp.write_text(json.dumps(s,indent=2));tmp.replace(state)
record()
env=dict(os.environ,ENTRY_FILE='scripts/head_regmean_pilot.py');env.pop('ENTRY_MODULE',None)
with (root/'reports/console.log').open('x') as log:
    p=subprocess.Popen([sys.executable,'/data/bmw/workers/e17.py'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    s['pid']=p.pid;record()
    while p.poll() is None:
        new=size(root);total=sum(size(v) for v in lineage)
        if time.time()>=s['deadline'] or new>=s['new_storage_limit'] or total>=s['lineage_storage_limit']:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            s['status']='budget_exhausted';break
        time.sleep(10)
s.update(exit_code=p.returncode,ended=time.time(),allocated_bytes=size(root),lineage_allocated_bytes=sum(size(v) for v in lineage))
if s['status']=='running':s['status']='completed' if p.returncode==0 and (root/'runs/pilot/COMPLETE.json').exists() else 'failed'
record()
