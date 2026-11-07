"""Unregistered single-worker budget primitive; not a launch entry point."""
import os
import math
import signal
import stat
import subprocess
import time
from pathlib import Path


def allocated(root):
    """Count allocated blocks without following symlinks outside the run root."""
    root = Path(root)
    root_stat = root.lstat()
    if not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError('budget root must be a real directory')

    def fail_walk(error):
        # os.walk otherwise suppresses inaccessible subtrees and undercounts.
        raise error

    total = root_stat.st_blocks * 512
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=fail_walk):
        for name in dirs + files:
            path = Path(directory) / name
            total += path.lstat().st_blocks * 512
    return total


def run_guarded(command, env, log, root, job, persist, *, gpu_cap, space_stop,
                interval=5, margin=30, max_worker_seconds=None):
    """Charge one GPU for the full worker lifetime and stop its process group.

    This assumes the supervisor is the sole owner of the experiment's job file.
    A separately enforced parent-death signal and outer timeout are still required.
    """
    numbers = (job['deadline'], job['gpu_seconds_used'], gpu_cap, space_stop, interval, margin)
    if (any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
            for x in numbers) or job['gpu_seconds_used'] < 0 or gpu_cap <= 0 or
            space_stop <= 0 or interval <= 0 or margin < 0 or gpu_cap <= margin):
        raise ValueError('invalid budget limits')
    if (max_worker_seconds is not None and
            (isinstance(max_worker_seconds, bool) or not isinstance(max_worker_seconds, (int, float)) or
             not math.isfinite(max_worker_seconds) or max_worker_seconds <= 0)):
        raise ValueError('invalid worker timeout')
    if job['deadline'] - margin <= time.time() or job['gpu_seconds_used'] >= gpu_cap - margin:
        raise ValueError('original budget exhausted')
    if allocated(root) >= space_stop:
        raise ValueError('storage stop reached before start')
    # A crash from here onward must leave an ambiguous active ledger, so a
    # replacement supervisor cannot silently resume with missing GPU charges.
    job.update(active=True, stop_reason=None, checked=time.time())
    persist(job)
    last = time.monotonic()
    worker_started = last
    proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    reason = None
    try:
        while True:
            now = time.monotonic()
            job['gpu_seconds_used'] += now - last
            last = now
            job['allocated_bytes'] = allocated(root)
            code = proc.poll()
            if time.time() >= job['deadline'] - margin:
                reason = 'deadline'
            elif job['gpu_seconds_used'] >= gpu_cap - margin:
                reason = 'gpu_budget'
            elif job['allocated_bytes'] >= space_stop:
                reason = 'storage'
            elif max_worker_seconds is not None and now - worker_started >= max_worker_seconds:
                reason = 'worker_timeout'
            job.update(active=code is None, stop_reason=reason, checked=time.time())
            persist(job)
            if code is not None or reason:
                break
            time.sleep(interval)
    finally:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        job['gpu_seconds_used'] += time.monotonic() - last
        job.update(active=False, exit_code=proc.returncode, stop_reason=reason,
                   allocated_bytes=allocated(root), checked=time.time())
        persist(job)
    return proc.returncode, reason
