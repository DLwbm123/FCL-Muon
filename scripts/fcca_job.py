"""Independent bounded worker for either explicitly authorized experiment."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def main():
    root = Path(os.environ['PROJECT_ROOT'])
    report = root / 'reports'
    arm='C'
    cfg = json.loads((root / 'configs' / 'fcca_three_seed0.json').read_text())
    receipt = report / 'job.json'
    resume = os.environ.get('RESUME_RUN') == '1'
    old = None
    if resume:
        old=json.loads(receipt.read_text())
        assert old['arm']==arm and old['gpu']==os.environ['CUDA_VISIBLE_DEVICES']
        assert not Path('/proc/'+str(old['worker_pid'])).exists()
        assert not Path('/proc/'+str(old['supervisor_pid'])).exists()
        assert old['storage_limit_bytes']==cfg['storage_limit_bytes']
        (report/('job_previous_'+str(int(time.time()))+'.json')).write_text(json.dumps(old,indent=2))
    elif receipt.exists() or (root / 'runs').exists():
        raise RuntimeError('Existing experiment: no duplicate launch, automatic retry or clock reset')
    started = old['started'] if resume else time.time()
    deadline = old['deadline'] if resume else started + cfg['maximum_seconds']
    assert time.time() < deadline, 'Original deadline exhausted'
    state = {'started': started, 'deadline': deadline,
             'status': 'starting', 'supervisor_pid': os.getpid(),
             'storage_limit_bytes': cfg['storage_limit_bytes'], 'write_stop_bytes': cfg['write_stop_bytes'],
             'seed': cfg['seed'], 'tasks': cfg['tasks'], 'gpu': os.environ['CUDA_VISIBLE_DEVICES'],
             'explicit_resume':resume,'arm':arm}

    def save():
        tmp = receipt.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(receipt)

    def size():
        total = 0
        for directory, _, files in os.walk(root, followlinks=False):
            for name in files:
                p = Path(directory) / name
                try:
                    if not p.is_symlink():
                        total += p.stat().st_blocks * 512
                except FileNotFoundError:
                    pass
        return total

    requested_stop = False

    def request_stop(*_):
        nonlocal requested_stop
        requested_stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    save()
    child = None
    try:
        free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits',
                                           '-i', os.environ['CUDA_VISIBLE_DEVICES']], text=True).strip())
        assert free >= cfg['required_free_mib'], 'Insufficient GPU admission memory'
        state['free_mib_at_admission'] = free
        state['baseline_allocated_bytes'] = old['baseline_allocated_bytes'] if resume else size()
        assert size() < cfg['write_stop_bytes']
        env = dict(os.environ, ENTRY_FILE='scripts/fcca_train.py')
        env.pop('ENTRY_MODULE', None)
        with (report / ('worker_resume_'+str(int(time.time()))+'.log' if resume else 'worker.log')).open('x') as log:
            child = subprocess.Popen([sys.executable, '/data/bmw/workers/e17.py'], env=env,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state.update(status='running', worker_pid=child.pid)
            save()
            while child.poll() is None:
                used = size()
                state.update(allocated_bytes=used, checked=time.time())
                if requested_stop or time.time() >= state['deadline'] or used >= cfg['write_stop_bytes']:
                    state['status'] = 'budget_or_signal_stop'
                    save()
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    break
                save()
                time.sleep(10)
        state['exit_code'] = child.returncode
        if state['status'] == 'running':
            qpath = root / 'runs/main/QUALIFICATION.json'
            success = child.returncode == 0 and qpath.exists() and (root / 'runs/main/COMPLETE.json').exists()
            if success:
                success = json.loads(qpath.read_text()).get('qualified', False)
            state.update(status='completed' if success else 'failed', qualified=bool(success))
    except BaseException as exc:
        state.update(status='supervisor_failure', error=repr(exc))
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        raise
    finally:
        state.update(ended=time.time(), allocated_bytes=size())
        save()


if __name__ == '__main__':
    main()
