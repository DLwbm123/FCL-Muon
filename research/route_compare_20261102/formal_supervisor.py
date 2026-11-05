"""Enforce shared original wall clock, aggregate GPU seconds and storage cap."""
import json, os, signal, subprocess, sys, time
from pathlib import Path
from research.route_compare_20261102.preflight_supervisor import allocated, write


def main():
    root=Path(os.environ['PROJECT_ROOT']);report=root/'reports';path=report/'job.json'
    with (report/'formal_claim.json').open('x') as f:json.dump({'time':time.time()},f)
    job=json.loads(path.read_text());frozen=json.loads((report/'FROZEN.json').read_text())
    if job['status']!='smoke_completed' or frozen['status']!='frozen_before_new_test':
        raise ValueError('pre-freeze lifecycle incomplete')
    deadline=float(job['deadline']);used=float(job['gpu_seconds_used'])
    if time.time()>=deadline-30 or used>=86400-30:raise TimeoutError('original budget exhausted')
    devices=('4','6')
    for gpu in devices:
        free=int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free',
               '--format=csv,noheader,nounits'],text=True).strip())
        if free<12000:raise RuntimeError('insufficient GPU memory')
    interrupted=False;children={};results={};last=time.monotonic();previously_active=0
    def stop(*_):
        nonlocal interrupted
        interrupted=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def start(seed,gpu):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,STUDY_SEED=str(seed),
                 ENTRY_FILE='research/route_compare_20261102/formal.py')
        log=(report/f'formal_seed{seed}.log').open('x')
        proc=subprocess.Popen([sys.executable,os.environ['WORKER_ENTRY']],env=env,
                              stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children[seed]=(proc,log)
    def terminate():
        for proc,_ in children.values():
            if proc.poll() is None:
                try:os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        for proc,_ in children.values():
            if proc.poll() is None:
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    try:
        start(0,devices[0]);previously_active=1
        job.update(status='formal_running',active_seeds=[0]);write(path,job)
        stage=0
        while True:
            now=time.monotonic();used+=(now-last)*previously_active;last=now
            for seed,(proc,log) in list(children.items()):
                if proc.poll() is not None:
                    results[seed]=proc.returncode;log.close();del children[seed]
            previously_active=len(children)
            job.update(gpu_seconds_used=used,allocated_bytes=allocated(root.parent),
                       active_seeds=list(children),seed_exit_codes=results,checked=time.time())
            if interrupted or time.time()>=deadline-30 or used>=86400-30 or job['allocated_bytes']>=19_000_000_000:
                job['status']='budget_or_signal_stop';terminate();break
            if any(code!=0 for code in results.values()):
                job['status']='formal_failed';terminate();break
            if stage==0 and results.get(0)==0:
                start(1,devices[0]);start(2,devices[1]);stage=1;previously_active=len(children)
            if stage==1 and not children:
                job['status']='formal_completed' if set(results)=={0,1,2} else 'formal_incomplete';break
            write(path,job);time.sleep(5)
    except BaseException as error:
        job.update(status='formal_supervisor_failed',error=repr(error));terminate();raise
    finally:
        for proc,log in children.values():
            if proc.poll() is None:proc.wait()
            log.close()
        job.update(gpu_seconds_used=used,allocated_bytes=allocated(root.parent),updated=time.time(),
                   seed_exit_codes=results)
        write(path,job)

if __name__=='__main__':main()
