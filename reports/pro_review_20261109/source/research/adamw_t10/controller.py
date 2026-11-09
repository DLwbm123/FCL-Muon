"""Exclusive durable supervisor. No score-dependent or seed-count gates."""
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from .storage import allocated,lock,write


def choose_gpu():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
    candidates=[(int(free),int(index)) for index,free in (row.split(',') for row in rows) if int(free)>=12000]
    return str(max(candidates,key=lambda x:(x[0],-x[1]))[1]) if candidates else None


def main():
    root=Path(os.environ['STUDY_ROOT']);reports=root/'src/reports'
    with (root/'.controller.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=reports/'job.json'
        if path.exists():raise RuntimeError('ledger exists; audit before any controller recovery')
        frozen=json.loads((reports/'FROZEN_PROTOCOL.json').read_text())
        if json.loads((reports/'checks.json').read_text())['status']!='passed':raise ValueError('CPU checks missing')
        started=frozen['started'];job={'started':started,'deadline':frozen['deadline'],'gpu_seconds_used':0.,
            'storage_limit_bytes':20_000_000_000,'write_stop_bytes':18_000_000_000,
            'status':'registering','active':False,'controller_pid':os.getpid(),'one_gpu_worker':True}
        write(root,path,job)
        stop=False
        def terminate(sig,frame):
            nonlocal stop
            stop=True
        signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
        for action in ('preflight','train'):
            gpu=None
            while action!='checks' and gpu is None:
                if stop or time.time()>=job['deadline']-300:
                    job.update(status='deadline_before_launch',active=False);write(root,path,job);return
                gpu=choose_gpu()
                if gpu is None:
                    job.update(status='waiting_for_gpu_memory',active=False);write(root,path,job);time.sleep(10)
            env=dict(os.environ,ACTION=action,CUDA_VISIBLE_DEVICES=gpu if gpu is not None else '')
            start=time.monotonic()
            with (reports/(action+'.log')).open('x') as log:
                proc=subprocess.Popen([sys.executable,str(root/'worker.py')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                job.update(status=action+'_running',active=True,worker_pid=proc.pid,gpu=gpu,action=action)
                write(root,path,job);kill_reason=None
                while proc.poll() is None:
                    now=time.monotonic();used=job['gpu_seconds_used']+(now-start if gpu is not None else 0.)
                    with lock(root):size=allocated(root)
                    job.update(gpu_seconds_live=used,allocated_bytes=size,updated=time.time())
                    write(root,path,job)
                    if stop or time.time()>=job['deadline']-120 or used>=86400-120 or size>=18_000_000_000:
                        kill_reason='signal_or_resource_hard_stop'
                        os.killpg(proc.pid,signal.SIGTERM)
                        try:proc.wait(timeout=25)
                        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                        break
                    time.sleep(5)
                if gpu is not None:job['gpu_seconds_used']+=time.monotonic()-start
                code=proc.wait();job.update(active=False,exit_code=code,status=action+'_completed' if code==0 else action+'_failed')
                job[action+'_exit']=code
                if kill_reason:job['reason']=kill_reason
                write(root,path,job)
                if code!=0:return
                receipt=reports/({'preflight':'preflight.json','train':'COMPLETE.json'}.get(action,'checks.json'))
                if not receipt.exists():raise ValueError('successful exit without receipt')
        job.update(status='completed',active=False,finished=time.time());write(root,path,job)


if __name__=='__main__':main()
