"""Independent hard-budget supervisor for the teacher-response study draft."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from scripts.real_boundary_supervisor import allocated, write


def main():
    root, source = Path(os.environ['PROJECT_ROOT']), Path(os.environ['SOURCE_ROOT'])
    report = root / 'reports';path = report / 'job.json'
    job = json.loads(path.read_text())
    if job['deadline'] - job['started'] != 10800 or time.time() >= job['deadline']:
        raise ValueError('invalid or exhausted original clock')
    with (report / 'SUPERVISOR_CLAIM.json').open('x') as claim:
        json.dump({'started': time.time(), 'clock': {k: job[k] for k in ('started', 'deadline')}}, claim)
    interrupted = False
    def request_stop(*_):
        nonlocal interrupted
        interrupted = True
    signal.signal(signal.SIGTERM, request_stop);signal.signal(signal.SIGINT, request_stop)
    child = None
    job.update(status='starting', supervisor_pid=os.getpid());write(path, job)
    try:
        if allocated(root) >= 6_000_000_000:
            raise ValueError('write-stop reached before launch')
        gpu = os.environ['CUDA_VISIBLE_DEVICES']
        free = int(subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-gpu=memory.free',
                                            '--format=csv,noheader,nounits'], text=True).strip())
        if free < 18000:
            raise ValueError('insufficient GPU memory')
        env = dict(os.environ, ENTRY_FILE='scripts/teacher_response_pair_draft.py', CUBLAS_WORKSPACE_CONFIG=':4096:8')
        env.pop('ENTRY_MODULE', None)
        env['PYTHONPATH'] = str(source / 'deps') + os.pathsep + env.get('PYTHONPATH', '')
        worker_timeout = int(job['deadline'] + 60 - time.time())
        if worker_timeout <= 0:
            raise TimeoutError('original worker deadline exhausted')
        with (report / 'worker.log').open('x') as log:
            child = subprocess.Popen(['/usr/bin/timeout', '-k', '30s', str(worker_timeout) + 's',
                                      sys.executable, '/data/bmw/workers/e17.py'], env=env,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            job.update(status='running', worker_pid=child.pid, free_mib_at_admission=free);write(path, job)
            while child.poll() is None:
                used = allocated(root)
                job.update(allocated_bytes=used, checked=time.time())
                if interrupted or time.time() >= job['deadline'] or used >= 6_000_000_000:
                    job['status'] = 'budget_or_signal_stop';write(path, job)
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL);child.wait()
                    break
                write(path, job);time.sleep(5)
            job['worker_exit_code'] = child.returncode
            if job['status'] == 'running':
                result = report / 'RESULT.json'
                job['status'] = ('completed' if child.returncode == 0 and result.exists() and
                                 json.loads(result.read_text()).get('status') == 'completed' else 'worker_failed')
    except BaseException as exc:
        job.update(status='supervisor_failure', error=repr(exc))
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL);child.wait()
        raise
    finally:
        job.update(ended=time.time(), allocated_bytes=allocated(root));write(path, job)


if __name__ == '__main__':
    main()
