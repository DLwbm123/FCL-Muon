"""Monitor the cohort under one durable wall, compute and storage ceiling."""

import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from research.adamw_t10.storage import allocated,lock,write
from research.analytic_comparison.control import available_gpus


def main():
    root=Path(os.environ['COHORT_ROOT'])
    with (root/'.cohort.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        supervise(root)


def supervise(root):
    manifest=json.loads((root/'run.json').read_text())
    jobs={job['lane']:job for job in manifest['jobs']}
    pending=set(jobs)
    records={}
    live={}
    used=0.
    stop=False
    reason=None

    def request_stop(*_):
        nonlocal stop
        stop=True

    signal.signal(signal.SIGINT,request_stop)
    signal.signal(signal.SIGTERM,request_stop)
    while pending or live:
        now=time.monotonic()
        for lane,(p,started,gpu) in list(live.items()):
            code=p.poll()
            if code is None:
                continue
            used+=now-started
            receipt=root/lane/jobs[lane]['receipt']
            try:
                ok=code==0 and json.loads(receipt.read_text())['status']=='completed'
            except (OSError,ValueError,KeyError):
                ok=False
            records[lane].update(status='completed' if ok else 'failed',exit_code=code,finished=time.time())
            del live[lane]

        with lock(root):
            size=allocated(root)
        effective=used+sum(now-s for _,s,_ in live.values())
        budget=manifest['budgets']
        if stop:reason='signal'
        elif time.time()>=manifest['deadline']-120:reason='wall_deadline'
        elif effective>=budget['aggregate_gpu_seconds']-120*budget['max_gpu_workers']:reason='gpu_budget'
        elif size>=budget['write_stop_bytes']:reason='allocated_storage_budget'
        if reason:
            for p,_,_ in live.values():
                if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
            for lane,(p,s,_) in live.items():
                try:code=p.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid,signal.SIGKILL)
                    code=p.wait()
                used+=time.monotonic()-s
                records[lane].update(status='stopped',exit_code=code)
            break

        for lane in sorted(pending):
            job=jobs[lane]
            dependencies=job.get('depends',[])
            states=[records.get(d,{}).get('status') for d in dependencies]
            if any(state in ('failed','blocked') for state in states):
                records[lane]={'status':'blocked','reason':'dependency failed'}
                pending.remove(lane)
                continue
            if any(state!='completed' for state in states) or len(live)>=budget['max_gpu_workers']:
                continue
            busy={gpu for _,_,gpu in live.values()}
            choices=[g for g in available_gpus(job['required_mib']) if g not in busy]
            if not choices:
                continue
            gpu=choices[0]
            work=root/lane
            env=dict(os.environ,STUDY_ROOT=str(work),CUDA_VISIBLE_DEVICES=gpu,ARM='rrs_shared',
                     OPERATION=job['operation'],TMPDIR=str(work/'tmp'),XDG_CACHE_HOME=str(work/'cache'),
                     TORCH_HOME=str(work/'cache'),**job.get('settings',{}))
            with (work/'output.log').open('x') as log:
                worker=subprocess.Popen([sys.executable,str(root/'entry.py')],cwd=work,env=env,
                                        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                        start_new_session=True)
            live[lane]=worker,time.monotonic(),gpu
            records[lane]=dict(status='running',pid=worker.pid,gpu=gpu,started=time.time())
            pending.remove(lane)
        snapshot=dict(status='running',lanes=records,pending=sorted(pending),allocated_bytes=size,
                      gpu_seconds=effective,started=manifest['started'],deadline=manifest['deadline'])
        write(root,root/'status.json',snapshot)
        if pending or live:time.sleep(12)
    status='stopped' if reason else ('completed' if all(r['status']=='completed' for r in records.values()) else 'failed')
    write(root,root/'status.json',dict(status=status,reason=reason,lanes=records,pending=sorted(pending),
          allocated_bytes=size,gpu_seconds=used,started=manifest['started'],deadline=manifest['deadline'],finished=time.time()))


if __name__=='__main__':
    main()
