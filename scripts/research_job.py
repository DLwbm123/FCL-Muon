"""Bounded research queue. No formal matrix submissions, no failed-run auto retry."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
root=Path(os.environ['PROJECT_ROOT']);os.chdir(root)
watch=root.parent
statepath=root/'reports/research-state.json'
if (root/'reports/USER_PAUSE.json').exists():raise SystemExit('User paused research experiments for Pro implementation review')
states=json.loads(statepath.read_text()) if statepath.exists() else {}
phase=os.environ.get('RESEARCH_PHASE','smoke')
queue=json.loads((root/'research/history_spectral'/f'{phase}_queue.json').read_text())

def record():
    tmp=statepath.with_suffix('.tmp');tmp.write_text(json.dumps(states,indent=2));tmp.replace(statepath)

def qualify(cfg,out):
    events=[json.loads(x) for x in (out/'events.jsonl').read_text().splitlines()]
    records={(e['task'],e['round'],e['client']):e for e in events if e['kind']=='client'}
    first_task=1 if cfg['method']!='mulora_fl_exact_polar' else 0
    expected=(cfg['tasks']-first_task)*cfg['rounds']*cfg['clients']
    assert len(records)==expected,(len(records),expected)
    if not cfg.get('train_batches'):
        assert all(e['steps']==cfg['epochs']*((e['N']+cfg['batch']-1)//cfg['batch']) and e['visited']==cfg['epochs']*e['N'] for e in records.values())
    diagnostics=[e for e in events if e['kind']=='history_solver_client']
    updates=sum(e['updates'] for e in diagnostics);nonconverged=sum(e['nonconverged'] for e in diagnostics)
    ratio=nonconverged/max(updates,1)
    result={'qualified':cfg['method']!='history_spectral' or ratio<=.01,'updates':updates,'nonconverged':nonconverged,
            'nonconverged_fraction':ratio,'zero_steps':sum(e['zero_steps'] for e in diagnostics),
            'max_relative_gap':max((e['max_relative_gap'] for e in diagnostics),default=0),'rule':'spectral QP <=1% actual-step nonconvergence; numerical qualification only'}
    (out/'QUALIFICATION.json').write_text(json.dumps(result,indent=2))
    return result['qualified']

for item in queue:
    if (root/'reports/USER_PAUSE.json').exists():raise SystemExit('User pause: research queue disabled')
    name=item['config'];cfg=json.loads((root/name).read_text());out=root/cfg['output'];s=states.setdefault(name,{'status':'not_started','phase':phase})
    if (out/'COMPLETE.json').exists():
        s['status']='completed' if qualify(cfg,out) else 'qualification_failed';record()
        if s['status']!='completed':break
        continue
    if s['status'] in ['failed','qualification_failed','budget_exhausted','blocked']:break
    if s.get('pid'):
        try:os.kill(s['pid'],0)
        except ProcessLookupError:pass
        else:raise SystemExit('Prior research child still exists; refusing duplicate')
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','-i',os.environ['CUDA_VISIBLE_DEVICES']],text=True).strip())
    if free<16000:s.update(status='blocked',reason='Insufficient free GPU memory');record();break
    env=dict(os.environ,ENTRY_MODULE='fcl.history_train',RUN_CONFIG=str(root/name));env.pop('ENTRY_FILE',None);env.pop('RESUME',None);env.pop('FORK_TASK1',None)
    if (out/'latest.pt').exists():env['RESUME']=str(out/'latest.pt')
    elif item.get('fork_task1'):
        fork=root/item['fork_task1']
        if not fork.exists():s.update(status='blocked',reason='Shared task1 checkpoint unavailable');record();break
        env['FORK_TASK1']=str(fork)
    started=s.setdefault('started',time.time());s.update(status='running',deadline=started+86400,gpu=os.environ['CUDA_VISIBLE_DEVICES'])
    out.mkdir(parents=True,exist_ok=True)
    with (out/'console.log').open('a') as log:
        child=subprocess.Popen([sys.executable,'/data/bmw/workers/e17.py'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        s['pid']=child.pid;record()
        while child.poll() is None:
            # Whole-project budget is deliberately stricter than per-run storage.
            allocated=sum(p.stat().st_blocks*512 for p in watch.rglob('*') if p.is_file() and not p.is_symlink())
            if time.time()>started+85500 or allocated>37_000_000_000:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=60)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                s.update(status='budget_exhausted',allocated_bytes=allocated);break
            time.sleep(15)
    s['exit_code']=child.returncode;s['ended']=time.time()
    if (out/'COMPLETE.json').exists():s['status']='completed' if qualify(cfg,out) else 'qualification_failed'
    elif s['status']!='budget_exhausted':s['status']='failed'
    record()
    if s['status']!='completed':break
