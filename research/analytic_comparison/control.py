"""Detached supervisor for two independent workers and a common hard budget."""

import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from research.adamw_t10.storage import allocated, lock, write


def available_gpus(required):
    lines = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True
    ).splitlines()
    candidates = [(int(free), int(index)) for index, free in (line.split(',') for line in lines)]
    return [str(index) for free, index in sorted(candidates, key=lambda x: (-x[0], x[1])) if free >= required]


def main():
    root = Path(os.environ['STUDY_ROOT'])
    with (root / '.controller.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        supervise(root)


def supervise(root):
    reports = root / 'src/reports'
    freeze = json.loads((reports / 'FROZEN_PROTOCOL.json').read_text())
    budget = freeze['budget']
    if (reports / 'job.json').exists():
        raise RuntimeError('Existing supervisor state requires an audited recovery')
    queues = {'rrs': [('preflight', 'preflight.json'), ('rrs', 'rrs_shared/COMPLETE.json')],
              'analytic': [('analytic', 'analytic/COMPLETE.json')]}
    live, records = {}, {}
    used = 0.0
    stopping = False
    reason = None

    def request_stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    status = 'running'
    while queues or live:
        now = time.monotonic()
        for lane, (process, start, gpu, action, receipt) in list(live.items()):
            code = process.poll()
            if code is None:
                continue
            used += now - start
            ok = code == 0 and (reports / receipt).exists()
            records[lane].update(state='completed' if ok else 'failed', exit_code=code, ended=time.time())
            del live[lane]
            if not ok or not queues[lane]:
                queues.pop(lane)
        with lock(root):
            size = allocated(root)
        live_seconds = used + sum(now - start for _, start, _, _, _ in live.values())
        if stopping:
            reason = 'signal'
        elif time.time() >= freeze['deadline'] - 120:
            reason = 'original_wall_deadline'
        elif live_seconds >= budget['aggregate_gpu_seconds'] - 120 * budget['max_workers']:
            reason = 'aggregate_gpu_budget'
        elif size >= budget['write_stop_bytes']:
            reason = 'allocated_storage_budget'
        if reason:
            status = 'stopped'
            for process, *_ in live.values():
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
            for lane, (process, start, _, _, _) in live.items():
                try:
                    code = process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    code = process.wait()
                used += time.monotonic() - start
                records[lane].update(state='stopped', exit_code=code, ended=time.time())
            break
        if any(steps for lane, steps in queues.items() if lane not in live):
            busy = {gpu for _, _, gpu, _, _ in live.values()}
            gpus = [gpu for gpu in available_gpus(budget['required_free_memory_mib']) if gpu not in busy]
            for lane, steps in queues.items():
                if lane in live or not steps or not gpus:
                    continue
                action, receipt = steps.pop(0)
                gpu = gpus.pop(0)
                env = dict(os.environ, ACTION=action, ARM='rrs_shared', CUDA_VISIBLE_DEVICES=gpu)
                with (reports / f'{action}.log').open('x') as log:
                    process = subprocess.Popen(
                        [sys.executable, str(root / 'entry.py')], env=env, stdout=log,
                        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True
                    )
                live[lane] = (process, time.monotonic(), gpu, action, receipt)
                records[lane] = dict(state='running', action=action, gpu=gpu, pid=process.pid, started=time.time())
        if not queues and not live:
            break
        write(root, reports / 'job.json', dict(status=status, updated=time.time(), lanes=records,
              gpu_seconds=live_seconds, allocated_bytes=size, started=freeze['started'], deadline=freeze['deadline']))
        time.sleep(10)
    if status == 'running':
        status = 'completed' if records and all(v['state'] == 'completed' for v in records.values()) else 'failed'
    write(root, reports / 'job.json', dict(status=status, reason=reason, lanes=records,
          gpu_seconds=used, allocated_bytes=size, started=freeze['started'], deadline=freeze['deadline'], finished=time.time()))
    if status != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
