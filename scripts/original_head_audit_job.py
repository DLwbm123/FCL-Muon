"""One bounded read-only original-head evaluation; never resumes training."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
root=Path(os.environ['PROJECT_ROOT']);state=root/'job.json'
if state.exists():raise SystemExit('Existing run; inspect evidence rather than repeat')
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','-i',os.environ['CUDA_VISIBLE_DEVICES']],text=True).strip())
if free<8000:raise SystemExit('Insufficient GPU memory')
started=time.time();s={'status':'running','started':started,'deadline':started+2700,'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'gpu_free_mib':free,'storage_limit':1000000000}
def record():
    temp=state.with_suffix('.tmp');temp.write_text(json.dumps(s,indent=2));temp.replace(state)
env=dict(os.environ,ENTRY_FILE='scripts/diagnose_original_aggregation.py');env.pop('ENTRY_MODULE',None)
with (root/'console.log').open('x') as log:
    child=subprocess.Popen([sys.executable,'/data/bmw/workers/e17.py'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    s['pid']=child.pid;record();allocated=0
    while child.poll() is None:
        allocated=sum(p.stat().st_blocks*512 for p in root.rglob('*') if p.is_file() and not p.is_symlink())
        if time.time()>=s['deadline'] or allocated>=s['storage_limit']:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            s['status']='budget_exhausted';break
        time.sleep(10)
s.update(exit_code=child.returncode,ended=time.time(),allocated_bytes=allocated)
if s['status']!='budget_exhausted':s['status']='completed' if child.returncode==0 and (root/'results/COMPLETE.json').exists() else 'failed'
record()
