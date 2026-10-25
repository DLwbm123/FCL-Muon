"""One LoRM comparison pilot with persisted deadline and allocated-storage guard."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
root=Path(os.environ['PROJECT_ROOT']);os.chdir(root)
statepath=root/'reports/lorm-state.json'
s=json.loads(statepath.read_text()) if statepath.exists() else {}
config='configs/pilot_lorm_official.json';cfg=json.loads((root/config).read_text());out=root/cfg['output']
# This job does not restart a failed scientific run automatically.
if s.get('status') in ['failed','completed','budget_exhausted']:raise SystemExit('Existing terminal LoRM state; inspect before any retry')
if s.get('status')=='running' and s.get('pid'):
    try:os.kill(s['pid'],0)
    except ProcessLookupError:pass
    else:raise SystemExit('Recorded LoRM process still exists; refusing duplicate launch')
smoke=root/'runs/smoke_lorm_official'
if not (smoke/'COMPLETE.json').exists():raise SystemExit('LoRM smoke completion required')
smoke_cfg=json.loads((smoke/'COMPLETE.json').read_text())['config']
for path,c in [(smoke,smoke_cfg)]:
    records=[json.loads(x) for x in (path/'events.jsonl').read_text().splitlines()]
    clients=[x for x in records if x['kind']=='client']
    assert len(clients)==c['tasks']*c['rounds']*c['clients']
    assert all(x['steps']>0 and x['train_matrix']==('B' if x['round']%2==0 else 'A') for x in clients)
    assert len(json.loads((path/'metrics.json').read_text())['matrix'])==c['tasks']
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','-i',os.environ['CUDA_VISIBLE_DEVICES']],text=True).strip())
if free<24000:raise SystemExit('Insufficient free GPU memory for upstream LoRM')
started=s.setdefault('started',time.time());remaining=started+85500-time.time()
if remaining<=0:raise SystemExit('Original LoRM deadline exhausted')
s.update(status='running',config=config,gpu=os.environ['CUDA_VISIBLE_DEVICES'],deadline=started+86400)
def record():
    tmp=statepath.with_suffix('.tmp');tmp.write_text(json.dumps(s,indent=2));tmp.replace(statepath)
out.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,ENTRY_MODULE='fcl.lorm_train',RUN_CONFIG=str(root/config),RUN_STARTED=str(started));env.pop('ENTRY_FILE',None)
if (out/'latest.pt').exists():env['RESUME']=str(out/'latest.pt')
else:env.pop('RESUME',None)
with (out/'console.log').open('a') as log:
    child=subprocess.Popen([sys.executable,'/data/bmw/workers/e17.py'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    s['pid']=child.pid;record();deadline=started+85500
    while child.poll() is None:
        allocated=sum(p.stat().st_blocks*512 for p in root.rglob('*') if p.is_file() and not p.is_symlink())
        if time.time()>=deadline or allocated>=40_000_000_000:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=60)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            s.update(status='budget_exhausted',allocated_bytes=allocated);break
        time.sleep(20)
s['exit_code']=child.returncode;s['ended']=time.time()
if (out/'COMPLETE.json').exists():
    records=[json.loads(x) for x in (out/'events.jsonl').read_text().splitlines()]
    clients=[x for x in records if x['kind']=='client']
    valid=len(clients)==cfg['tasks']*cfg['rounds']*cfg['clients'] and all(x['visited']==cfg['epochs']*x['N'] and x['steps']==cfg['epochs']*((x['N']+cfg['batch']-1)//cfg['batch']) for x in clients)
    qualification={'qualified':valid,'client_records':len(clients),'purpose':'same-data full-epoch budget and numerical completion check'}
    (out/'QUALIFICATION.json').write_text(json.dumps(qualification,indent=2))
    s['status']='completed' if valid else 'failed'
elif s['status']!='budget_exhausted':s['status']='failed'
record()
