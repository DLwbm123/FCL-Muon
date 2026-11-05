"""Supervise paired training-only smoke on the original study budget."""
import json, os, signal, subprocess, sys, time
from pathlib import Path
from research.route_compare_20261102.preflight_supervisor import allocated, write


def main():
    root=Path(os.environ['PROJECT_ROOT']); report=root/'reports'; path=report/'job.json'
    with (report/'smoke_claim.json').open('x') as f: json.dump({'time':time.time()},f)
    state=json.loads(path.read_text())
    if state['status']!='preflight_passed' or state['optimizer']!='muon':
        raise ValueError('required GPU preflights not completed')
    used=float(state['gpu_seconds_used']);deadline=float(state['deadline'])
    if time.time() >= deadline-30 or used>=86400: raise TimeoutError('original budget exhausted')
    children=[]; stopped=False; previous=time.monotonic()
    def stop_signal(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop_signal);signal.signal(signal.SIGINT,stop_signal)
    try:
        for gpu in ('0','3'):
            free=int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free',
                     '--format=csv,noheader,nounits'],text=True).strip())
            if free < 40000: raise RuntimeError('insufficient GPU memory')
        for optimizer,gpu in (('adamw','0'),('muon','3')):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,LOCAL_OPTIMIZER=optimizer,
                     ENTRY_FILE='research/route_compare_20261102/smoke.py')
            log=(report/('smoke_'+optimizer+'.log')).open('x')
            child=subprocess.Popen([sys.executable,os.environ['WORKER_ENTRY']],env=env,
                                   stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children.append((optimizer,child,log))
        state.update(status='smoke_running',smoke_gpus=['0','3']);write(path,state)
        previously_active=len(children)
        while True:
            now=time.monotonic();delta=now-previous;previous=now
            # Count all lanes conservatively through each observation interval.
            used += delta*previously_active
            previously_active=sum(child.poll() is None for _,child,_ in children)
            state['gpu_seconds_used']=used;state['allocated_bytes']=allocated(root.parent)
            state['smoke_exit_codes']={name:child.poll() for name,child,_ in children}
            failed=any(child.poll() not in (None,0) for _,child,_ in children)
            active=any(child.poll() is None for _,child,_ in children)
            if failed or stopped or time.time()>=deadline-30 or used>=86400-30 or state['allocated_bytes']>=19_000_000_000:
                state['status']='smoke_failed' if failed else 'budget_or_signal_stop'
                for _,child,_ in children:
                    if child.poll() is None:
                        try: os.killpg(child.pid,signal.SIGTERM)
                        except ProcessLookupError: pass
                for _,child,_ in children:
                    if child.poll() is None:
                        try:child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid,signal.SIGKILL);child.wait()
                break
            if not active:
                state['status']='smoke_completed';break
            write(path,state);time.sleep(5)
    except BaseException as error:
        state.update(status='smoke_supervisor_failed',error=repr(error))
        for _,child,_ in children:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGKILL);child.wait()
        raise
    finally:
        for _,_,log in children:log.close()
        state.update(gpu_seconds_used=used,updated=time.time(),allocated_bytes=allocated(root.parent),
                     smoke_exit_codes={name:child.poll() for name,child,_ in children})
        write(path,state)

if __name__ == '__main__': main()
