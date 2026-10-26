"""Independent original-clock/storage guard for the three-task aggregation-utility pilot."""
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
    cfg = json.loads((root / 'configs/aggregation_utility_three_task_seed0.json').read_text())
    receipt = report / 'job.json'
    resume = os.environ.get('RESUME_RUN') == '1'
    old = None
    if resume:
        migration = json.loads((report / 'GPU0_MIGRATION_READY.json').read_text())
        old = json.loads((report / 'GPU0_MIGRATION_ORIGINAL_JOB.json').read_text())
        assert migration['clock'] == {k:old[k] for k in ['started','deadline']}
        assert os.environ['CUDA_VISIBLE_DEVICES'] == migration['authorized_gpu'] == '0'
        assert not Path('/proc/' + str(old['worker_pid'])).exists()
        assert not Path('/proc/' + str(old['supervisor_pid'])).exists()
        assert old['storage_limit_bytes'] == cfg['storage_limit_bytes'] and old['write_stop_bytes'] == cfg['write_stop_bytes']
    elif receipt.exists() or (root / 'runs').exists():
        raise RuntimeError('Existing experiment: no duplicate launch, automatic retry or clock reset')
    started = old['started'] if resume else time.time()
    deadline = old['deadline'] if resume else started + cfg['maximum_seconds']
    assert time.time() < deadline, 'Original deadline exhausted'
    state = {'started': started, 'deadline': deadline,
             'status': 'starting', 'supervisor_pid': os.getpid(),
             'storage_limit_bytes': cfg['storage_limit_bytes'], 'write_stop_bytes': cfg['write_stop_bytes'],
             'seed': cfg['seed'], 'tasks': cfg['tasks'], 'gpu': os.environ['CUDA_VISIBLE_DEVICES'],
             'user_requested_gpu_migration': resume}

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
        env = dict(os.environ, ENTRY_FILE='scripts/aggregation_utility_train.py')
        env.pop('ENTRY_MODULE', None)
        with (report / ('worker_gpu0_resume.log' if resume else 'worker.log')).open('x') as log:
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
            qpath = root / 'runs/STUDY_QUALIFICATION.json'
            success = child.returncode == 0 and qpath.exists() and (root / 'runs/COMPLETE.json').exists()
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
