"""Run only the never-started standard references after the exact-pair failure.

The original failed experiment receipt and pair gates remain unchanged. Both
references keep their original configurations, original clock and storage cap.
"""
import json,os,signal,subprocess,sys,time
from pathlib import Path

root=Path(os.environ['PROJECT_ROOT'])

def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def worker():
    import math
    import torch
    seed=int(os.environ['LANE_SEED']);cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());out=root/cfg['output']
    if out.exists():raise RuntimeError('Reference already started; no automatic retry')
    from fcl.lorm_train import main
    main()
    assert (out/'COMPLETE.json').exists()
    records=[json.loads(v) for v in (out/'events.jsonl').read_text().splitlines()]
    clients=[v for v in records if v['kind']=='client']
    assert len(clients)==150 and sum(v['visited'] for v in clients)==375000
    assert len({(v['task'],v['round'],v['client']) for v in clients})==150
    assert all(v['visited']==v['N']*5 and v['steps']==5*math.ceil(v['N']/16) and v['train_matrix']==('B' if v['round']%2==0 else 'A') for v in clients)
    cp=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
    assert cp['task']==3 and cp['round']==0 and not cp['initialized']
    assert cp['matrix']==json.loads((out/'metrics.json').read_text())['matrix']
    from scripts.head_confirmation_audit_lorm import verify
    audit=verify(root,out,cfg,cp)
    write(out/'QUALIFICATION.json',{'qualified':True,'client_rounds':150,'training_visits':375000,
        'optimizer_steps':sum(v['steps'] for v in clients),'checkpoint_boundary_complete':True,**audit,
        'original_exact_head_pair_gate':'FAILED and retained; independent official reference does not repair it'})

def supervisor():
    original=json.loads((root/'reports/job.json').read_text())
    assert original['status']=='failed'
    for seed in [1,2]:
        lane=json.loads((root/f'reports/lane_seed{seed}.json').read_text())
        assert lane['status']=='pair_identity_failed' and len(lane['runs'])==2
        assert all(v['status']=='completed' and v['exit_code']==0 for v in lane['runs'])
        assert not (root/f'runs/lorm_seed{seed}').exists()
    path=root/'reports/remaining_reference_job.json'
    if path.exists():raise RuntimeError('Existing remaining-reference receipt; do not duplicate')
    s={'started':original['started'],'deadline':original['deadline'],'launched':time.time(),
       'storage_limit_bytes':original['storage_limit_bytes'],'status':'running','runs':[],
       'reason':'Continue two original, unstarted standard references; no replay or relaxation of failed head-pair audit'}
    if time.time()>=s['deadline']:raise TimeoutError('Original clock exhausted')
    for gpu in [0,3]:
        free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','-i',str(gpu)],text=True).strip())
        if free<24000:raise RuntimeError('Insufficient GPU memory')
    def size():
        total=0
        for p in root.rglob('*'):
            try:
                if p.is_file() and not p.is_symlink():total+=p.stat().st_blocks*512
            except FileNotFoundError:pass
        return total
    s['allocated_before_bytes']=size();write(path,s);children=[];logs=[]
    for seed,gpu in [(1,0),(2,3)]:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),LANE_SEED=str(seed),REFERENCE_WORKER='1',
                 ENTRY_FILE='scripts/head_remaining_references.py',RUN_CONFIG=str(root/f'configs/lorm_seed{seed}.json'),RUN_STARTED=str(s['started']))
        env.pop('ENTRY_MODULE',None)
        log=(root/f'reports/lorm_seed{seed}.log').open('x');logs.append(log)
        p=subprocess.Popen([sys.executable,'/data/bmw/workers/e17.py'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(p);s['runs'].append({'seed':seed,'gpu':gpu,'pid':p.pid,'status':'running'});write(path,s)
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
    for row,p in zip(s['runs'],children):
        row.update(exit_code=p.poll(),status='completed' if p.returncode==0 and (root/f"runs/lorm_seed{row['seed']}/QUALIFICATION.json").exists() else 'failed')
    s.update(ended=time.time(),allocated_bytes=size())
    if s['status']=='running':s['status']='completed' if all(r['status']=='completed' for r in s['runs']) else 'failed'
    write(path,s)
    for log in logs:log.close()

if os.environ.get('REFERENCE_WORKER')=='1':worker()
else:supervisor()
