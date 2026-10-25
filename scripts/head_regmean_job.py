"""Bounded single-run supervisor; no implicit resume or retry."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
root=Path(os.environ['PROJECT_ROOT']);state=root/'reports/job.json'
if state.exists():raise SystemExit('Existing run: do not duplicate or reset clock')
cfg=json.loads((root/'research/head_regmean/PROTOCOL.json').read_text())
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','-i',os.environ['CUDA_VISIBLE_DEVICES']],text=True).strip())
if free<12000:raise SystemExit('Insufficient GPU memory')
started=time.time();s={'started':started,'deadline':started+cfg['hard_timeout_seconds'],'status':'running',
                      'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'gpu_free_mib':free,'storage_limit':cfg['new_storage_guard_bytes']}
def record():
    temp=state.with_suffix('.tmp');temp.write_text(json.dumps(s,indent=2));temp.replace(state)
record()
env=dict(os.environ,ENTRY_FILE='scripts/head_regmean_round.py',EXPERIMENT_STARTED=str(started));env.pop('ENTRY_MODULE',None)
with (root/'reports/console.log').open('x') as log:
    child=subprocess.Popen([sys.executable,'/data/bmw/workers/e17.py'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    s['pid']=child.pid;record()
    while child.poll() is None:
        allocated=sum(p.stat().st_blocks*512 for p in root.rglob('*') if p.is_file() and not p.is_symlink())
        if time.time()>=s['deadline'] or allocated>=s['storage_limit']:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            s['status']='budget_exhausted';break
        time.sleep(10)
s.update(exit_code=child.returncode,ended=time.time(),allocated_bytes=sum(p.stat().st_blocks*512 for p in root.rglob('*') if p.is_file() and not p.is_symlink()))
if s['status']!='budget_exhausted':s['status']='completed' if child.returncode==0 and (root/'runs/COMPLETE.json').exists() else 'failed'
record()
