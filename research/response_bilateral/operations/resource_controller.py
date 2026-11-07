"""One-off quiescent controller handoff, then original frozen study schedule.

Scientific worker/config/source files remain untouched. This wrapper records an
explicit resource-only amendment and resumes completed stages without re-running.
"""
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path


BASE = Path(__file__).resolve().parent


def stage_receipt(root, name, values):
    reports = root/'reports'
    if name == 'preflight':
        p = reports/'preflight_result.json'
        return p.exists() and json.loads(p.read_text()).get('status') == 'passed'
    if name.startswith('train_'):
        p = reports/(name+'.json')
        if not p.exists(): return False
        r = json.loads(p.read_text())
        return (r.get('status') == 'completed' and r['seed'] == int(values['SEED'])
                and r['arm'] == values['ARM'] and r['task'] == int(values['TARGET_TASK']))
    if name.startswith('prefix_'):
        seed = int(values['SEED'])
        p = reports/f'prefix_s{seed}_E0_T1.json'
        return (p.exists() and json.loads(p.read_text()).get('status') == 'completed'
                and (root/f'runs/prefix{seed}/latest.pt').is_file())
    return False


def choose_memory_gpu():
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                    '--format=csv,noheader,nounits'], text=True).splitlines()
    candidates = []
    for row in rows:
        index, free = (int(x.strip()) for x in row.split(','))
        if free >= 12000: candidates.append((-free, index))
    return str(min(candidates)[1]) if candidates else None


def adapted_source(source):
    replacements = [
        ("if old['status'] not in ('development_checked',):",
         "if old['status'] != 'waiting_for_idle_gpu' or old.get('active') is not False:"),
        ("job.update(status='registering', active=False); persist(job)",
         "job.update(status='resource_admission_resume', active=False); persist(job)"),
        ("def run(name, entry, values=None, limit=None):\n",
         "def run(name, entry, values=None, limit=None):\n"
         "            if name+'_exit' in job:\n"
         "                if job[name+'_exit'] != 0 or not stage_receipt(root, name, values):\n"
         "                    raise ValueError('completed-stage receipt missing or failed; audit required')\n"
         "                return True\n"),
        ("status='waiting_for_idle_gpu'", "status='waiting_for_gpu_memory'")]
    for before, after in replacements:
        if source.count(before) != 1: raise ValueError('controller source contract changed')
        source = source.replace(before, after)
    compile(source, '<resource-amended-supervisor>', 'exec')
    return source


def own_process(pid, name):
    path = Path(f'/proc/{pid}/cmdline')
    if not path.exists(): return False
    return str(BASE/name).encode() in path.read_bytes().split(b'\0')


@contextmanager
def stopped_scheduler(pid):
    if not own_process(pid, 'driver.py'): raise ValueError('supervisor identity lost')
    os.kill(pid, signal.SIGSTOP)
    try:
        limit = time.monotonic()+5
        while time.monotonic() < limit:
            status = Path(f'/proc/{pid}/status').read_text()
            if any(line.startswith('State:') and 'T' in line for line in status.splitlines()): break
            time.sleep(.01)
        else: raise RuntimeError('scheduler did not stop for boundary check')
        yield
    finally:
        # Also deliver an already-pending TERM after an intentional handoff.
        try: os.kill(pid, signal.SIGCONT)
        except ProcessLookupError: pass


def main():
    settings = json.loads((BASE/'settings.json').read_text())
    os.environ.update(settings)
    sys.path[:0] = settings['PYTHONPATH'].split(os.pathsep)
    root = Path(settings['PROJECT_ROOT'])
    reports = root/'reports'
    frozen = json.loads((reports/'FROZEN_PROTOCOL.json').read_text())
    source_path = root/'research/response_bilateral/supervisor.py'
    source = source_path.read_text()
    expected = frozen['source_sha256']['research/response_bilateral/supervisor.py']
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != expected:
        raise ValueError('original frozen controller changed')
    patched = adapted_source(source)
    old_outer = json.loads((BASE/'launch.json').read_text())['outer_pid']
    children = Path(f'/proc/{old_outer}/task/{old_outer}/children')
    pids = [int(x) for x in children.read_text().split()] if children.exists() else []
    candidates = [p for p in pids if own_process(p, 'driver.py')]
    if len(candidates) != 1: raise ValueError('cannot identify original supervisor')
    old_supervisor = candidates[0]
    print(json.dumps({'state': 'waiting_for_quiescent_resource_boundary',
                      'original_supervisor': old_supervisor}), flush=True)
    while time.time() < frozen['deadline']-300:
        job = json.loads((reports/'job.json').read_text())
        if any(job[k] != frozen[k] for k in ('started', 'deadline')):
            raise ValueError('original clock changed')
        if job['status'] == 'waiting_for_idle_gpu' and job.get('active') is False:
            if choose_memory_gpu() is None:
                time.sleep(10); continue
            # Freeze only our idle scheduler while checking the boundary; this
            # closes the spawn/check race. A raced-in child keeps running and
            # the original scheduler is immediately resumed, never terminated.
            with stopped_scheduler(old_supervisor):
                child = Path(f'/proc/{old_supervisor}/task/{old_supervisor}/children')
                active = [int(x) for x in child.read_text().split()] if child.exists() else []
                stopped_job = json.loads((reports/'job.json').read_text())
                if (active or stopped_job.get('active') is not False
                        or stopped_job.get('status') != 'waiting_for_idle_gpu'):
                    continue
                job = stopped_job
                # Stop only our idle scheduler. No training process is interrupted.
                os.kill(old_supervisor, signal.SIGTERM)
            limit = time.monotonic()+15
            while own_process(old_supervisor, 'driver.py') and time.monotonic() < limit:
                time.sleep(.1)
            if own_process(old_supervisor, 'driver.py'):
                raise RuntimeError('original scheduler did not exit')
            after = json.loads((reports/'job.json').read_text())
            if after != job: raise ValueError('ledger changed during resource handoff')
            from research.response_bilateral.runtime import write
            record = {'time': time.time(), 'authorization': 'GPU_RESOURCE_AMENDMENT_20261107.json',
                'original_frozen_controller_sha256': expected,
                'resource_wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'adapted_controller_sha256': hashlib.sha256(patched.encode()).hexdigest(),
                'original_clock': {k: job[k] for k in ('started', 'deadline')},
                'gpu_seconds_at_handoff': job['gpu_seconds_used'], 'pending': job['pending'],
                'old_supervisor': old_supervisor, 'new_supervisor': os.getpid(),
                'minimum_free_MiB': 12000, 'ignore_process_presence_and_utilization': True,
                'completed_stages_skip_only_with_success_receipts': True,
                'scientific_sources_config_order_and_budgets_unchanged': True}
            path = reports/'RESOURCE_CONTROLLER_AMENDMENT_1.json'
            if path.exists(): raise ValueError('handoff already registered')
            write(path, record)
            namespace = {'__name__': 'resource_amended_controller', '__file__': str(source_path),
                         'stage_receipt': stage_receipt}
            exec(compile(patched, str(source_path), 'exec'), namespace)
            namespace['idle_gpu'] = choose_memory_gpu
            namespace['freeze'] = lambda ledger: frozen
            print(json.dumps({'state': 'resource_controller_active', **record}), flush=True)
            namespace['main']()
            return
        if not own_process(old_supervisor, 'driver.py'):
            print(json.dumps({'state': 'original_controller_ended_without_resource_wait', 'job': job}), flush=True)
            return
        time.sleep(10)
    print('Original deadline near; resource wrapper did not start new work', flush=True)


if __name__ == '__main__': main()
