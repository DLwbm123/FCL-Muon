"""Shared experiment clock/storage across two seed lanes and all six runs."""
import json,os,signal,subprocess,sys,time
from pathlib import Path

root=Path(os.environ['PROJECT_ROOT']);receipt=root/'reports/job.json'
cfg=json.loads((root/'research/head_regmean/CONFIRMATION_PROTOCOL.json').read_text())
if receipt.exists():raise SystemExit('Existing experiment receipt; never duplicate/reset')
assert json.loads((root/'reports/PREFLIGHT.json').read_text())['passed']
for lane in cfg['lanes']:
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','-i',str(lane['gpu'])],text=True).strip())
    if free<24000:raise SystemExit('Insufficient lane memory before launch')
def size():return sum(p.stat().st_blocks*512 for p in root.rglob('*') if p.is_file() and not p.is_symlink())
started=time.time();s={'started':started,'deadline':started+85500,'status':'running','lanes':[],
    'storage_limit_bytes':20000000000,'baseline_allocated_bytes':size(),'protocol':'research/head_regmean/CONFIRMATION_PROTOCOL.json'}
def save():
    tmp=receipt.with_suffix('.tmp');tmp.write_text(json.dumps(s,indent=2));tmp.replace(receipt)
save();children=[];logs=[]
for lane in cfg['lanes']:
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(lane['gpu']),LANE_SEED=str(lane['seed']),ENTRY_FILE='scripts/head_confirmation_worker.py');env.pop('ENTRY_MODULE',None)
    log=(root/f"reports/worker_seed{lane['seed']}.log").open('x');logs.append(log)
    child=subprocess.Popen([sys.executable,'/data/bmw/workers/e17.py'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    children.append(child);s['lanes'].append({**lane,'pid':child.pid});save()
while any(p.poll() is None for p in children):
    if time.time()>=s['deadline'] or size()>=s['storage_limit_bytes']:
        for p in children:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        until=time.time()+15
        while any(p.poll() is None for p in children) and time.time()<until:time.sleep(1)
        for p in children:
            if p.poll() is None:os.killpg(p.pid,signal.SIGKILL);p.wait()
        s['status']='budget_exhausted';break
    time.sleep(15)
s.update(ended=time.time(),exit_codes=[p.poll() for p in children],allocated_bytes=size())
if s['status']=='running':s['status']='completed' if all(p.returncode==0 for p in children) else 'failed'
save()
if s['status']=='completed':(root/'reports/COMPLETE.json').write_text(json.dumps({'completed':time.time(),'six_runs':True}))
for log in logs:log.close()
