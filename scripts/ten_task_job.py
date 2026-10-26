"""Six runs: complete both seed-zero arms, then both seed-one, then seed-two.
All children share this original clock and the independently authorized storage.
"""
import json,os,signal,subprocess,sys,time
from pathlib import Path

root=Path(os.environ['PROJECT_ROOT']);receipt=root/'reports/job.json'
protocol=json.loads((root/'research/head_regmean/TEN_TASK_PROTOCOL.json').read_text())
if receipt.exists():raise SystemExit('Existing experiment receipt; never duplicate/reset')
assert json.loads((root/'reports/PREFLIGHT.json').read_text())['passed']
def size():
    # os.walk does not follow symlinked shared directories; allocated bytes count
    # all actual new files, including temporary files, logs and checkpoints.
    total=0
    for directory,_,files in os.walk(root,followlinks=False):
        for name in files:
            p=Path(directory)/name
            try:
                if not p.is_symlink():total+=p.stat().st_blocks*512
            except FileNotFoundError:pass # an atomic replace completed during scan
    return total
started=time.time();state={'started':started,'deadline':started+85500,'status':'running','runs':[],
    'storage_limit_bytes':20000000000,'write_stop_bytes':16000000000,
    'baseline_allocated_bytes':size(),'protocol':'research/head_regmean/TEN_TASK_PROTOCOL.json'}
children=[];logs=[];stop_requested=False

def save():
    tmp=receipt.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(receipt)

def stopped(*_):
    global stop_requested
    stop_requested=True

signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)

def terminate():
    for child in children:
        if child.poll() is None:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
    until=time.time()+30
    while any(c.poll() is None for c in children) and time.time()<until:time.sleep(1)
    # Kill each owned process group even if its worker exited first: grandchildren
    # (including signal-saving LoRM) must not outlive the common hard deadline.
    for child in children:
        try:os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        child.wait()

save()
try:
    for seed in [0,1,2]:
        if stop_requested or time.time()>=state['deadline'] or size()>=state['write_stop_bytes']:
            state['status']='budget_or_signal_stop';break
        children=[];current=[]
        for method,gpu in [('regmean',3),('lorm',6)]:
            free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','-i',str(gpu)],text=True).strip())
            if free<(24000 if method=='lorm' else 14000):raise RuntimeError(f'GPU {gpu} admission failed')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),LANE_SEED=str(seed),RUN_METHOD=method,ENTRY_FILE='scripts/ten_task_worker.py')
            env.pop('ENTRY_MODULE',None)
            log=(root/f'reports/worker_{method}_seed{seed}.log').open('x');logs.append(log)
            child=subprocess.Popen([sys.executable,'/data/bmw/workers/e17.py'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children.append(child);run={'method':method,'seed':seed,'gpu':gpu,'worker_pid':child.pid,'started':time.time()}
            current.append(run);state['runs'].append(run);save()
        while any(c.poll() is None for c in children):
            used=size()
            if stop_requested or time.time()>=state['deadline'] or used>=state['write_stop_bytes']:
                state.update(status='budget_or_signal_stop',allocated_bytes=used);save();terminate();break
            time.sleep(10)
        for run,child in zip(current,children):run.update(exit_code=child.poll(),ended=time.time())
        save()
        if state['status']!='running':break
        # A numerical/data/qualification failure blocks later seeds; performance
        # never controls admission. The other arm of this pair may finish.
        if any(c.returncode!=0 for c in children):state['status']='technical_failure';break
        for run in current:
            r=json.loads((root/f"reports/{run['method']}_seed{seed}.json").read_text())
            assert r['status']=='completed' and r['qualified']
    if state['status']=='running':
        assert len(state['runs'])==6
        state['status']='completed'
        (root/'reports/COMPLETE.json').write_text(json.dumps({'completed':time.time(),'six_qualified_runs':True}))
except BaseException as exc:
    state.update(status='supervisor_failure',error=repr(exc));terminate();raise
finally:
    state.update(ended=time.time(),allocated_bytes=size());save()
    for log in logs:log.close()
