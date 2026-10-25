"""Three shared-state lanes, inherited earliest clock and conservative lineage cap."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
import torch

root=Path(os.environ['PROJECT_ROOT']);base=Path(os.environ['EXPERIMENT_ROOT'])
protocol=json.loads((root/'research/head_regmean/HEAD_LOSS_PROTOCOL.json').read_text())
path=root/'reports/job.json'
if path.exists():raise RuntimeError('Existing intervention receipt; no duplicate or clock reset')
assert json.loads((root/'reports/PREFLIGHT.json').read_text())['passed']
clocks=[]
for lane in protocol['lanes']:
    x=torch.load(base/lane['source'],map_location='cpu',weights_only=False);clocks.append(x['clock']);del x
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','-i',str(lane['gpu'])],text=True).strip())
    if free<24000:raise RuntimeError('Insufficient free GPU memory')
started=min(v['started'] for v in clocks);deadline=min(v['deadline'] for v in clocks)
if time.time()>=deadline:raise TimeoutError('Inherited clock exhausted')
lineage=[root,base/'head_confirmation_20261025',base/'head_regmean_pilot_20261025',base/'head_regmean_round_20261025',base/'basic_repair_20261025']
def size(folder):
    total=0
    for p in folder.rglob('*'):
        try:
            if p.is_file() and not p.is_symlink():total+=p.stat().st_blocks*512
        except FileNotFoundError:pass
    return total
s={'started':started,'deadline':deadline,'intervention_launched':time.time(),'status':'running','lanes':[],
   'storage_limit_bytes':20000000000,'lineage_limit_bytes':45000000000,
   'new_before_bytes':size(root),'lineage_before_bytes':sum(size(p) for p in lineage),'source_clocks':clocks}
def save():
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(s,indent=2));tmp.replace(path)
save();children=[];logs=[]
for lane in protocol['lanes']:
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(lane['gpu']),LANE_SEED=str(lane['seed']),
        CUBLAS_WORKSPACE_CONFIG=':4096:8',ENTRY_FILE='scripts/head_loss_worker.py');env.pop('ENTRY_MODULE',None)
    log=(root/f"reports/seed{lane['seed']}.log").open('x');logs.append(log)
    p=subprocess.Popen([sys.executable,'/data/bmw/workers/e17.py'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    children.append(p);s['lanes'].append({'seed':lane['seed'],'gpu':lane['gpu'],'pid':p.pid});save()
while any(p.poll() is None for p in children):
    new=size(root);total=sum(size(p) for p in lineage)
    if time.time()>=deadline or new>=s['storage_limit_bytes'] or total>=s['lineage_limit_bytes']:
        for p in children:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        until=time.time()+15
        while any(p.poll() is None for p in children) and time.time()<until:time.sleep(1)
        for p in children:
            if p.poll() is None:os.killpg(p.pid,signal.SIGKILL);p.wait()
        s['status']='budget_exhausted';break
    time.sleep(15)
s.update(ended=time.time(),exit_codes=[p.poll() for p in children],new_allocated_bytes=size(root),lineage_allocated_bytes=sum(size(p) for p in lineage))
if s['status']=='running':s['status']='completed' if all(p.returncode==0 for p in children) else 'failed'
save()
if s['status']=='completed':(root/'reports/COMPLETE.json').write_text(json.dumps({'completed':time.time(),'six_branches':True}))
for log in logs:log.close()
