"""Draft independent timeout and allocated-storage supervisor."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def allocated(root):
    total = 0
    for directory, _, files in os.walk(root, followlinks=False):
        for name in files:
            path = Path(directory) / name
            try:
                if not path.is_symlink():
                    total += path.stat().st_blocks * 512
            except FileNotFoundError:
                continue
    return total


def main():
    root = Path(os.environ['PROJECT_ROOT'])
    source = Path(os.environ['SOURCE_ROOT'])
    report = root / 'reports'
    jobpath = report / 'job.json'
    job = json.loads(jobpath.read_text())
    if job['deadline'] - job['started'] != 10800 or time.time() >= job['deadline']:
        raise ValueError('invalid or exhausted original diagnostic clock')
    with (report / 'SUPERVISOR_CLAIM.json').open('x') as claim:
        json.dump({'started': time.time(), 'clock': {k: job[k] for k in ('started', 'deadline')}}, claim)
    stop_requested = False
    def signal_stop(*_):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGTERM, signal_stop)
    signal.signal(signal.SIGINT, signal_stop)
    child = None
    job.update(status='starting', supervisor_pid=os.getpid())
    write(jobpath, job)
    try:
        used = allocated(root)
        if used >= 6_000_000_000:
            raise ValueError('storage write-stop already reached')
        gpu = os.environ['CUDA_VISIBLE_DEVICES']
        free = int(subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-gpu=memory.free',
                                             '--format=csv,noheader,nounits'], text=True).strip())
        if free < 18000:
            raise ValueError('insufficient available GPU memory')
        env = dict(os.environ, ENTRY_FILE='scripts/real_boundary_worker.py')
        env.pop('ENTRY_MODULE', None)
        env['PYTHONPATH'] = str(source / 'deps') + os.pathsep + env.get('PYTHONPATH', '')
        with (report / 'worker.log').open('x') as log:
            child = subprocess.Popen([sys.executable, '/data/bmw/workers/e17.py'], env=env,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            job.update(status='running', worker_pid=child.pid, free_mib_at_admission=free)
            write(jobpath, job)
            while child.poll() is None:
                used = allocated(root)
                job.update(allocated_bytes=used, checked=time.time())
                if stop_requested or time.time() >= job['deadline'] or used >= 6_000_000_000:
                    job['status'] = 'budget_or_signal_stop'
                    write(jobpath, job)
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    break
                write(jobpath, job)
                time.sleep(5)
            job['worker_exit_code'] = child.returncode
            if job['status'] == 'running':
                result_path = report / 'RESULT.json'
                if child.returncode != 0 or not result_path.exists():
                    job['status'] = 'worker_failed'
                else:
                    result = json.loads(result_path.read_text())
                    job['status'] = 'completed' if result.get('status') == 'completed' else 'invalid_result'
    except BaseException as exc:
        job.update(status='supervisor_failure', error=repr(exc))
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        raise
    finally:
        job.update(ended=time.time(), allocated_bytes=allocated(root))
        write(jobpath, job)


if __name__ == '__main__':
    main()
