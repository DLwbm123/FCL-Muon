"""Single non-retrying neutral-entry job with persistent clock and storage guard."""
import json,os,signal,subprocess,sys,time
from pathlib import Path

root=Path(os.environ['PROJECT_ROOT']);record=root/'reports/job.json'
cfg=json.loads((root/os.environ['JOB_PROTOCOL']).read_text())
if record.exists():raise SystemExit('Existing job receipt; do not duplicate or reset clock')
def size():return sum(p.stat().st_blocks*512 for p in root.rglob('*') if p.is_file() and not p.is_symlink())
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','-i',os.environ['CUDA_VISIBLE_DEVICES']],text=True).strip())
if free<cfg['minimum_gpu_free_mib']:raise SystemExit('Insufficient GPU memory')
started=time.time();s={'started':started,'deadline':started+cfg['hard_timeout_seconds'],'status':'running',
    'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'gpu_free_mib':free,'baseline_allocated_bytes':size(),
    'storage_limit_bytes':cfg['storage_limit_bytes'],'protocol':os.environ['JOB_PROTOCOL']}
def save():
    tmp=record.with_suffix('.tmp');tmp.write_text(json.dumps(s,indent=2));tmp.replace(record)
save();env=dict(os.environ,ENTRY_FILE=cfg['entry']);env.pop('ENTRY_MODULE',None)
with (root/'reports/console.log').open('x') as log:
    child=subprocess.Popen([sys.executable,'/data/bmw/workers/e17.py'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    s['pid']=child.pid;save()
    while child.poll() is None:
        allocated=size()
        if time.time()>=s['deadline'] or allocated>=s['storage_limit_bytes']:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            s['status']='budget_exhausted';break
        time.sleep(10)
s.update(ended=time.time(),exit_code=child.returncode,allocated_bytes=size())
if s['status']=='running':s['status']='completed' if child.returncode==0 and (root/cfg['completion']).exists() else 'failed'
save()
