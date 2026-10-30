"""One common clock/storage supervisor and real GPU preflight for matched arms."""
import json,os,runpy,signal,subprocess,sys,time
from pathlib import Path


def main():
    root=Path(os.environ['PROJECT_ROOT']);report=root/'reports';receipt=report/'job.json'
    if receipt.exists():raise RuntimeError('Existing experiment: do not duplicate or reset its clock')
    assert json.loads((report/'CPU_TESTS.json').read_text())['passed']
    entry=os.environ['WORKER_ENTRY'];gpus=json.loads(os.environ['LANE_GPUS'])
    prefix=os.environ.get('CONFIG_PREFIX','spectra')
    seed=int(os.environ.get('EXPERIMENT_SEED','0'))
    if seed not in (0,1,2):raise ValueError('Unregistered seed')
    if prefix not in ('spectra','actual_update','actual_gate','actual_direction','actual_tangent','actual_matrix','actual_allseen','actual_rank8'):raise ValueError('Unknown configuration family')
    arms=('control','global') if prefix in ('actual_update','actual_gate','actual_direction','actual_tangent','actual_matrix','actual_allseen','actual_rank8') else ('control','mvp','full')
    if os.environ.get('RUN_ARMS')=='global':
        assert prefix in ('actual_update','actual_gate','actual_direction','actual_tangent','actual_matrix','actual_allseen','actual_rank8') and (root/'runs/control/QUALIFICATION_T5.json').exists()
        arms=('global',)
    cfg=json.loads((root/f'configs/{prefix}_{arms[-1]}_seed{seed}.json').read_text())
    if len(arms)==1:assert cfg.get('training_gate')
    budget_root=Path(os.environ.get('BUDGET_ROOT',str(root)))
    assert root.is_relative_to(budget_root)
    assert set(gpus)==set(arms) and len(set(gpus.values()))==len(arms)
    for arm in arms:
        other=json.loads((root/f'configs/{prefix}_{arm}_seed{seed}.json').read_text())
        assert other['seed']==seed
        assert all(other[k]==cfg[k] for k in ('tasks','maximum_seconds','storage_limit_bytes','write_stop_bytes'))
    for gpu in gpus.values():
        free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        if free<cfg['required_free_mib']:raise RuntimeError('Insufficient GPU admission memory')
    # Includes initial source setup/CPU qualification conservatively, not just
    # training. This immutable receipt is also used by the external timeout.
    origin=json.loads((report/'ORIGIN.json').read_text())
    started=origin['started'];deadline=started+cfg['maximum_seconds']
    if time.time()>=deadline:raise TimeoutError('Original budget exhausted')
    def size():
        total=0
        for directory,_,files in os.walk(budget_root,followlinks=False):
            for name in files:
                p=Path(directory)/name
                try:
                    if not p.is_symlink():total+=p.stat().st_blocks*512
                except FileNotFoundError:pass
        return total
    state={'started':started,'deadline':deadline,'status':'preflight','lanes':[],
           'storage_limit_bytes':cfg['storage_limit_bytes'],'write_stop_bytes':cfg['write_stop_bytes'],
           'baseline_allocated_bytes':size(),'supervisor_pid':os.getpid()}
    children=[];logs=[];stop=False
    def save():
        temp=receipt.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(receipt)
    def request_stop(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    def terminate():
        for p in children:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        until=time.time()+30
        while any(p.poll() is None for p in children) and time.time()<until:time.sleep(.5)
        for p in children:
            if p.poll() is None:os.killpg(p.pid,signal.SIGKILL);p.wait()
    def poll_budget():
        used=size();state.update(allocated_bytes=used,checked=time.time());save()
        if stop or time.time()>=deadline or used>=cfg['write_stop_bytes']:
            state['status']='budget_or_signal_stop';save();terminate();return False
        return True
    def spawn(arm,preflight=False):
        out=report/arm;out.mkdir(exist_ok=True)
        name='preflight' if preflight else arm
        log=(report/f'{name}.log').open('x');logs.append(log)
        env=dict(os.environ,ARM=arm,CUDA_VISIBLE_DEVICES=str(gpus[arm]),ENTRY_FILE='scripts/spectra_train.py')
        env.pop('ENTRY_MODULE',None)
        if preflight:env['PREFLIGHT_ONLY']='1'
        else:env.pop('PREFLIGHT_ONLY',None)
        p=subprocess.Popen([sys.executable,entry],env=env,stdin=subprocess.DEVNULL,
                           stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(p)
        state['lanes'].append({'arm':name,'gpu':gpus[arm],'pid':p.pid,'started':time.time()});save()
        return p
    save()
    try:
        preflight=spawn(arms[-1],True)
        while preflight.poll() is None:
            if not poll_budget():return
            time.sleep(10)
        if preflight.returncode!=0 or not json.loads((report/'REAL_PREFLIGHT.json').read_text())['passed']:
            raise RuntimeError('Real GPU preflight failed; formal training not launched')
        state['status']='running';save()
        workers=[spawn(arm) for arm in arms]
        while any(p.poll() is None for p in workers):
            if not poll_budget():return
            time.sleep(10)
        qualified=all(p.returncode==0 and (root/'runs'/arm/'COMPLETE.json').exists()
                      for arm,p in zip(arms,workers))
        state.update(status='completed' if qualified else 'failed',qualified=qualified,
                     exit_codes={arm:p.returncode for arm,p in zip(arms,workers)})
        if qualified:
            if cfg.get('report_entry'):runpy.run_path(str(root/cfg['report_entry']),run_name='__main__')
            (report/'COMPLETE.json').write_text(json.dumps({'qualified':True,'arms':len(arms),'tasks_per_arm':cfg['tasks'],'completed':time.time()}))
    except BaseException as exc:
        state.update(status='supervisor_failure',error=repr(exc));terminate();raise
    finally:
        state.update(ended=time.time(),allocated_bytes=size());save()
        for log in logs:log.close()


if __name__=='__main__':main()
