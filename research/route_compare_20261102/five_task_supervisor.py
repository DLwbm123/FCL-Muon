"""One-clock GPU supervisor for the registered five-task continuation."""
import json
import os
import ctypes
import signal
import subprocess
import sys
import time
from pathlib import Path
from research.route_compare_20261102.five_task_worker import digest
from research.route_compare_20261102.preflight_supervisor import allocated, write


def idle_gpus():
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True).splitlines()
    return [index.strip() for index, memory, util in (row.split(',') for row in rows)
            if int(memory) < 100 and int(util) == 0]


def freeze(root, source, prior, job):
    import torch, timm, kornia
    original = json.loads((prior / 'reports/FROZEN.json').read_text())
    identities = {}
    for name, expected in original['source_sha256'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path) != expected:
            raise ValueError('original frozen source does not match: ' + name)
        identities[name] = expected
    for name in ('five_task_worker.py', 'five_task_supervisor.py', 'five_task_preflight.py', 'long_sequence_prefix.py'):
        key = 'research/route_compare_20261102/' + name
        identities[key] = digest(root / key)
    manifests = [source / 'manifests/seq-cifar100_n10_a0.05_s0.json',
                 prior / 'manifests/seed1.json', prior / 'manifests/seed2.json']
    mhashes = {str(i): digest(p) for i, p in enumerate(manifests)}
    if mhashes != original['manifest_sha256']:
        raise ValueError('original manifest mismatch')
    weights = json.loads((source / 'weights.json').read_text())
    if weights['sha256'] != original['weight_sha256'] or digest(Path(weights['path'])) != original['weight_sha256']:
        raise ValueError('original pretrained weight mismatch')
    ckpaths = [prior / 'runs/smoke_adamw/latest.pt'] + [prior / f'runs/seed{i}_adamw_prefix/latest.pt' for i in (1, 2)]
    pref = {str(i): digest(p) for i, p in enumerate(ckpaths)}
    proto = root / 'research/route_compare_20261102/FIVE_TASK_PROTOCOL_ZH.md'
    result = {'status': 'frozen_before_new_test', 'tasks': 5, 'seeds': [0, 1, 2],
              'arms': ['rrs', 'selected'], 'started': job['started'], 'deadline': job['deadline'],
              'frozen_at': time.time(), 'protocol_sha256': digest(proto), 'route': original['route'],
              'source_sha256': identities, 'manifest_sha256': mhashes, 'prefix_sha256': pref,
              'weight_sha256': original['weight_sha256'], 'environment': {'torch': str(torch.__version__),
              'timm': str(timm.__version__), 'kornia': str(kornia.__version__), 'cuda': str(torch.version.cuda)},
              'gpu_seconds_cap': 86400, 'write_stop_bytes': 19000000000,
              'scope': 'nonblind; shared AdamW T1, original A0/A1, tasks 2-5'}
    with (root / 'reports/FROZEN_FIVE_TASK.json').open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)


def main():
    root = Path(os.environ['PROJECT_ROOT']).resolve()
    source = Path(os.environ['SOURCE_ROOT']).resolve()
    prior = Path(os.environ['PRIOR_ROOT']).resolve()
    reports = root / 'reports'
    reports.mkdir(exist_ok=True)
    claim = reports / 'job.json'
    start = time.time()
    job = {'status': 'registering', 'started': start, 'deadline': start + 86400,
           'gpu_seconds_used': 0., 'allocated_bytes': allocated(root.parent), 'active': {}, 'exits': {}}
    with claim.open('x') as f:
        json.dump(job, f)
    children, results = {}, {}
    used, last = 0., time.monotonic()
    interrupted = False
    def stop(*_):
        nonlocal interrupted
        interrupted = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    def terminate():
        for proc, _, _ in children.values():
            if proc.poll() is None:
                try: os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError: pass
        for proc, _, _ in children.values():
            if proc.poll() is None:
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
    def start_child(name, device, entry, seed=None):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=device, ENTRY_FILE=entry)
        if seed is not None:
            env['STUDY_SEED'] = str(seed)
        log = (reports / (name + '.log')).open('x')
        command = [sys.executable, os.environ['WORKER_ENTRY']]
        if name == 'preflight':
            command = ['/usr/bin/timeout', '-k', '15s', '1800s'] + command
        parent = os.getpid()
        def parent_death_signal():
            if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM) != 0:
                os._exit(125)
            if os.getppid() != parent:
                os._exit(125)
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, preexec_fn=parent_death_signal)
        children[name] = (proc, log, device)
    try:
        freeze(root, source, prior, job)
        stage = 'preflight_waiting'
        while True:
            now = time.monotonic()
            used += (now - last) * len(children)
            last = now
            for name, (proc, log, device) in list(children.items()):
                if proc.poll() is not None:
                    results[name] = proc.returncode
                    log.close()
                    del children[name]
            space = allocated(root.parent)
            job.update(status=stage, gpu_seconds_used=used, allocated_bytes=space,
                       active={k: v[2] for k, v in children.items()}, exits=results, checked=time.time())
            if interrupted or time.time() >= job['deadline'] - 30 or used >= 86400 - 30 or space >= 19000000000:
                job['status'] = 'budget_or_signal_stop'
                terminate()
                break
            if any(v != 0 for v in results.values()):
                job['status'] = 'failed'
                terminate()
                break
            if stage == 'preflight_waiting':
                available = idle_gpus()
                if available:
                    start_child('preflight', available[0], 'research/route_compare_20261102/five_task_preflight.py')
                    stage = 'preflight_running'
            elif stage == 'preflight_running' and results.get('preflight') == 0:
                stage = 'formal_waiting'
            elif stage == 'formal_waiting':
                available = idle_gpus()
                if available:
                    start_child('seed0', available[0], 'research/route_compare_20261102/five_task_worker.py', 0)
                    stage = 'formal_running'
            elif stage == 'formal_running' and results.get('seed0') == 0:
                available = [g for g in idle_gpus() if g not in [v[2] for v in children.values()]]
                for seed in (1, 2):
                    if available and f'seed{seed}' not in children and f'seed{seed}' not in results:
                        start_child(f'seed{seed}', available.pop(0), 'research/route_compare_20261102/five_task_worker.py', seed)
                if all(results.get(f'seed{i}') == 0 for i in (0, 1, 2)):
                    job['status'] = 'formal_completed'
                    break
            write(claim, job)
            time.sleep(5)
    except BaseException as error:
        job.update(status='supervisor_failed', error=repr(error))
        terminate()
        raise
    finally:
        now = time.monotonic()
        used += (now - last) * len(children)
        for proc, log, _ in children.values():
            if proc.poll() is None:
                proc.wait()
            log.close()
        job.update(gpu_seconds_used=used, allocated_bytes=allocated(root.parent), updated=time.time())
        write(claim, job)


if __name__ == '__main__':
    main()
