"""One locked T1 recovery + head diagnosis; no second encoder run or history job."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

root=Path(os.environ['PROJECT_ROOT'])
statepath=root/'reports/job.json'
if statepath.exists():raise SystemExit('Existing run; do not reset original clock or repeat')
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','-i',os.environ['CUDA_VISIBLE_DEVICES']],text=True).strip())
if free<12000:raise SystemExit('Insufficient GPU memory')
started=time.time()
s={'started':started,'deadline':started+85500,'status':'running','gpu':os.environ['CUDA_VISIBLE_DEVICES'],'gpu_free_mib_at_launch':free,'storage_limit':12000000000,'phases':[]}
def record():
    temp=statepath.with_suffix('.tmp');temp.write_text(json.dumps(s,indent=2));temp.replace(statepath)
record()
phases=[('centering','scripts/basic_repair_centering.py',None),('original_T1',None,'fcl.train'),('fixed_encoder_heads','scripts/basic_repair_heads.py',None)]
for name,file,module in phases:
    env=dict(os.environ,RUN_CONFIG=str(root/'configs/t1.json'))
    env.pop('ENTRY_MODULE',None);env.pop('ENTRY_FILE',None);env.pop('RESUME',None)
    env['ENTRY_MODULE' if module else 'ENTRY_FILE']=module or file
    phase={'name':name,'started':time.time()};s['phases'].append(phase)
    with (root/'reports'/f'{name}.log').open('x') as log:
        child=subprocess.Popen([sys.executable,'/data/bmw/workers/e17.py'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        phase['pid']=child.pid;record()
        while child.poll() is None:
            # rglob does not traverse shared resource directory symlinks.
            allocated=sum(p.stat().st_blocks*512 for p in root.rglob('*') if p.is_file() and not p.is_symlink())
            if time.time()>=s['deadline'] or allocated>=s['storage_limit']:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                s.update(status='budget_exhausted',allocated_bytes=allocated);break
            time.sleep(10)
    phase.update(exit_code=child.returncode,ended=time.time())
    if child.returncode!=0 or s['status']=='budget_exhausted':
        if s['status']!='budget_exhausted':s['status']='failed'
        record();raise SystemExit('Phase failed; no automatic retry')
    if name=='original_T1':
        output=root/'runs/t1_original'
        assert (output/'COMPLETE.json').exists()
        events=[json.loads(line) for line in (output/'events.jsonl').read_text().splitlines()]
        clients=[e for e in events if e['kind']=='client']
        assert len(clients)==50 and all(e['visited']==5*e['N'] and e['steps']==5*((e['N']+15)//16) for e in clients)
        (root/'reports/T1_QUALIFICATION.json').write_text(json.dumps({'qualified':True,'client_records':50,'full_epoch_budget_verified':True,'exit_code':child.returncode},indent=2))
    record()
assert (root/'runs/head_diagnostics/COMPLETE.json').exists()
s.update(status='completed',ended=time.time());record()
