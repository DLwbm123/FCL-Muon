"""One frozen seed lane. No retry, tuning, or result-dependent branch selection."""
import json,os,subprocess,sys,time
from pathlib import Path
import torch

root=Path(os.environ['PROJECT_ROOT']);seed=int(os.environ['LANE_SEED'])
receipt=root/f'reports/lane_seed{seed}.json'
if receipt.exists():raise SystemExit('Existing lane receipt')
state={'seed':seed,'status':'running','runs':[]}
def save():
    temp=receipt.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(receipt)
save()
for method in ['regmean','fedavg','lorm']:
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','-i',os.environ['CUDA_VISIBLE_DEVICES']],text=True).strip())
    if free<(24000 if method=='lorm' else 14000):
        state.update(status='resource_blocked',blocked_method=method,free_mib=free);save();raise SystemExit(2)
    config=root/f'configs/{method}_seed{seed}.json';cfg=json.loads(config.read_text());out=root/cfg['output']
    if out.exists():raise RuntimeError('Existing run output; no automatic retry')
    env=dict(os.environ,RUN_CONFIG=str(config),RUN_STARTED=str(json.loads((root/'reports/job.json').read_text())['started']))
    if method=='lorm':
        env['ENTRY_MODULE']='fcl.lorm_train';env.pop('ENTRY_FILE',None)
    else:
        env['ENTRY_FILE']='scripts/head_confirmation_train.py';env.pop('ENTRY_MODULE',None)
    logpath=root/f'reports/{method}_seed{seed}.log'
    with logpath.open('x') as log:
        p=subprocess.Popen([sys.executable,'/data/bmw/workers/e17.py'],env=env,stdout=log,stderr=subprocess.STDOUT)
        run={'method':method,'seed':seed,'pid':p.pid,'started':time.time(),'status':'running'}
        state['runs'].append(run);save();code=p.wait()
    run.update(exit_code=code,ended=time.time(),status='completed' if code==0 and (out/'COMPLETE.json').exists() else 'failed');save()
    if run['status']!='completed':state['status']='failed';save();raise SystemExit(1)
    if method=='fedavg':
        a=torch.load(root/f'runs/regmean_seed{seed}/first_round_endpoints.pt',map_location='cpu',weights_only=False)
        b=torch.load(out/'first_round_endpoints.pt',map_location='cpu',weights_only=False)
        checks={k:len(a[k])==len(b[k]) and all(torch.equal(x,y) for x,y in zip(a[k],b[k])) for k in ['heads','B','A']}
        checks.update(sizes=a['sizes']==b['sizes'],torch_rng=torch.equal(a['rng']['torch'],b['rng']['torch']),
                      cuda_rng=all(torch.equal(x,y) for x,y in zip(a['rng']['cuda'],b['rng']['cuda'])),
                      python_rng=a['rng']['python']==b['rng']['python'])
        import numpy as np
        checks['numpy_rng']=a['rng']['numpy'][0]==b['rng']['numpy'][0] and np.array_equal(a['rng']['numpy'][1],b['rng']['numpy'][1]) and a['rng']['numpy'][2:]==b['rng']['numpy'][2:]
        (root/f'reports/PAIR_seed{seed}.json').write_text(json.dumps({'passed':all(checks.values()),'checks':checks},indent=2))
        if not all(checks.values()):state['status']='pair_identity_failed';save();raise SystemExit(1)
    if method=='lorm':
        records=[json.loads(v) for v in (out/'events.jsonl').read_text().splitlines()]
        clients=[v for v in records if v['kind']=='client']
        assert len(clients)==150 and sum(v['visited'] for v in clients)==375000
        assert all(v['visited']==v['N']*5 and v['train_matrix']==('B' if v['round']%2==0 else 'A') for v in clients)
        cp=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
        assert cp['task']==3 and cp['round']==0 and not cp['initialized']
        metrics=json.loads((out/'metrics.json').read_text());assert cp['matrix']==metrics['matrix']
        assert all((out/f'task{t}_fixed_rules.json').exists() for t in [1,2,3])
        from scripts.head_confirmation_audit_lorm import verify
        audit=verify(root,out,cfg,cp)
        (out/'QUALIFICATION.json').write_text(json.dumps({'qualified':True,'client_rounds':150,'training_visits':375000,
            'optimizer_steps':sum(v['steps'] for v in clients),'checkpoint_boundary_complete':True,
            **audit,'note':'Official standard learner; fixed-rule metrics from task-end temporal-merge logits'},indent=2))
state['status']='completed';state['ended']=time.time();save()
