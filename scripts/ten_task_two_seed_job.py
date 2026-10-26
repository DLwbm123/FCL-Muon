"""Adopt live seed-zero workers without restarting training; run seed one on one GPU.
The amendment narrows scope only. The original experiment clock/storage remain.
"""
import json,os,signal,subprocess,sys,time
from pathlib import Path


def identity(pid):
    try:
        line=Path(f'/proc/{pid}/stat').read_text();parts=line[line.rfind(')')+2:].split()
        return {'pid':pid,'state':parts[0],'pgid':int(parts[2]),'start_ticks':int(parts[19])}
    except (FileNotFoundError,ProcessLookupError):return None


def alive(saved):
    now=identity(saved['pid'])
    return now is not None and now['start_ticks']==saved['start_ticks'] and now['state'] not in ['Z','X']


def next_seed_one_method(states):
    # Existing arms may run concurrently; no performance threshold is consulted.
    if any(v.get('status')=='failed' for v in states.values()):return None
    if ('regmean',1) not in states:return 'regmean'
    first=states[('regmean',1)]
    if first.get('status')=='completed' and first.get('qualified') and ('lorm',1) not in states:return 'lorm'
    return None


def main():
    root=Path(os.environ['PROJECT_ROOT']);report=root/'reports';receipt=report/'job.json'
    amendment=json.loads((root/'research/head_regmean/TWO_SEED_AMENDMENT.json').read_text())
    state=json.loads(receipt.read_text());handoff=json.loads((report/'TWO_SEED_HANDOFF.json').read_text())
    assert state['started']==handoff['original_started'] and state['deadline']==handoff['original_deadline']
    assert {(v['method'],v['seed']) for v in state['runs']}=={('regmean',0),('lorm',0)}
    assert all(v['seed']==0 for v in state['runs']) and amendment['allowed_seeds']==[0,1]
    old=identity(handoff['old_supervisor']['pid']);assert old and old['start_ticks']==handoff['old_supervisor']['start_ticks'] and old['state']=='T'
    owned={};children={};logs=[];stop_requested=False
    def save():
        tmp=receipt.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(receipt)
    def size():
        total=0
        for d,_,files in os.walk(root,followlinks=False):
            for name in files:
                p=Path(d)/name
                try:
                    if not p.is_symlink():total+=p.stat().st_blocks*512
                except FileNotFoundError:pass
        return total
    def stop(*_):
        nonlocal stop_requested
        stop_requested=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def terminate():
        groups=[]
        for saved in owned.values():
            if alive(saved):
                assert saved['pgid']==saved['pid'];groups.append(saved['pgid'])
                try:os.killpg(saved['pgid'],signal.SIGTERM)
                except ProcessLookupError:pass
        until=time.time()+30
        while any(alive(v) for v in owned.values()) and time.time()<until:time.sleep(1)
        for pgid in groups:
            try:os.killpg(pgid,signal.SIGKILL)
            except ProcessLookupError:pass
    for run in state['runs']:
        key=(run['method'],run['seed']);saved=handoff['adopted_workers'][run['method']]
        assert saved['pid']==run['worker_pid'] and saved['pgid']==saved['pid'] and alive(saved)
        owned[key]=saved
    # Readiness is written before taking ownership. The launcher retires only the
    # stopped old scheduler + its timeout, then commits the handoff. No worker signal.
    ack={'ready':True,'pid':os.getpid(),'identity':identity(os.getpid()),'original_started':state['started'],'original_deadline':state['deadline'],'adopted_workers':list(owned.values()),'time':time.time()}
    (report/'TWO_SEED_GUARD_READY.json').write_text(json.dumps(ack,indent=2))
    until=time.time()+45
    while not (report/'TWO_SEED_HANDOFF_COMMITTED.json').exists():
        if time.time()>until or stop_requested:raise RuntimeError('Handoff not committed; old scheduler retains ownership')
        time.sleep(.2)
    state.update(status='running',schedule='seed0 retained; seed1 regmean then lorm on GPU0; seed2 cancelled before launch',allowed_seeds=[0,1],expected_qualified_runs=4,supervisor_pid=os.getpid(),amendment='research/head_regmean/TWO_SEED_AMENDMENT.json')
    save()
    try:
        while True:
            used=size()
            if stop_requested or time.time()>=state['deadline'] or used>=state['write_stop_bytes']:
                state.update(status='budget_or_signal_stop',allocated_bytes=used);save();terminate();break
            statuses={};running=False
            for run in state['runs']:
                key=(run['method'],run['seed']);p=report/f"{run['method']}_seed{run['seed']}.json"
                s=json.loads(p.read_text()) if p.exists() else {'status':'starting'}
                live=alive(owned[key]);running|=live
                if not live and s.get('status') not in ['completed','failed']:
                    s.update(status='failed',error='Worker disappeared before terminal receipt');p.write_text(json.dumps(s,indent=2))
                if s.get('status')=='completed':
                    q=root/f"runs/{run['method']}_seed{run['seed']}/QUALIFICATION.json"
                    if not s.get('qualified') or not q.exists() or not json.loads(q.read_text()).get('qualified'):
                        s.update(status='failed',error='Missing completion qualification');p.write_text(json.dumps(s,indent=2))
                statuses[key]=s;run.update(status=s['status'],qualified=s.get('qualified',False))
                if not live:
                    run['ended']=s.get('ended',time.time())
                    if key in children:run['exit_code']=children[key].poll()
                    else:run['exit_code']=0 if s['status']=='completed' else None
            method=next_seed_one_method(statuses)
            if method and (('regmean',1) not in owned or not alive(owned[('regmean',1)])):
                free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','-i','0'],text=True).strip())
                need=24000 if method=='lorm' else 14000
                if free>=need:
                    assert not (report/f'{method}_seed1.json').exists() and not (root/f'runs/{method}_seed1').exists()
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',LANE_SEED='1',RUN_METHOD=method,ENTRY_FILE='scripts/ten_task_worker.py');env.pop('ENTRY_MODULE',None)
                    log=(report/f'worker_{method}_seed1.log').open('x');logs.append(log)
                    child=subprocess.Popen([sys.executable,'/data/bmw/workers/e17.py'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    saved=identity(child.pid);assert saved and saved['pgid']==child.pid
                    key=(method,1);owned[key]=saved;children[key]=child
                    state['runs'].append({'method':method,'seed':1,'gpu':0,'worker_pid':child.pid,'started':time.time(),'process_identity':saved})
                    state.pop('pending_gpu_admission',None);running=True
                else:state['pending_gpu_admission']={'method':method,'seed':1,'gpu':0,'free_mib':free,'required_mib':need}
            all_done=len(statuses)==4 and all(s.get('status')=='completed' and s.get('qualified') for s in statuses.values())
            if all_done and not running:
                state['status']='completed';save()
                (report/'COMPLETE.json').write_text(json.dumps({'completed':time.time(),'four_qualified_runs':True,'seeds':[0,1],'seed2_cancelled_by_user_before_launch':True}));break
            if any(s.get('status')=='failed' for s in statuses.values()) and not running:
                state['status']='technical_failure';break
            save();time.sleep(10)
    except BaseException as exc:
        state.update(status='supervisor_failure',error=repr(exc));terminate();raise
    finally:
        state.update(ended=time.time(),allocated_bytes=size());save()
        for log in logs:log.close()

if __name__=='__main__':main()
