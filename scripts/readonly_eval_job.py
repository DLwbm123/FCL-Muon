"""Bounded, one-off evaluator supervisor; paths only arrive via environment."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

root = Path(os.environ['PROJECT_ROOT'])
record = root / 'job.json'
if record.exists():
    raise SystemExit('Existing evaluation job: inspect instead of launching again')
free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits', '-i', os.environ['CUDA_VISIBLE_DEVICES']], text=True).strip())
if free < 8000:
    raise SystemExit('Insufficient memory for read-only evaluation')
started = time.time()
state = {'started': started, 'deadline': started+1800, 'gpu': os.environ['CUDA_VISIBLE_DEVICES'], 'free_gpu_mib_at_launch': free, 'status': 'starting', 'storage_limit_bytes': 100_000_000}
record.write_text(json.dumps(state, indent=2))
env = dict(os.environ, ENTRY_FILE='scripts/diagnose_checkpoint.py')
output = Path(os.environ['EXPERIMENT_ROOT']) / 'runs/diagnostic_pilot_checkpoint_20261025'
with (root / 'console.log').open('x') as log:
    child = subprocess.Popen([sys.executable, '/data/bmw/workers/e17.py'], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    state.update(status='running', pid=child.pid)
    record.write_text(json.dumps(state, indent=2))
    allocated = 0
    while child.poll() is None:
        allocated = sum(p.stat().st_size for p in output.glob('*') if p.is_file()) + (root / 'console.log').stat().st_size
        if time.time() >= state['deadline'] or allocated >= state['storage_limit_bytes']:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            state['status'] = 'budget_exhausted'
            break
        time.sleep(10)
state.update(exit_code=child.returncode, ended=time.time(), output_bytes=allocated)
if state['status'] != 'budget_exhausted':
    state['status'] = 'completed' if child.returncode == 0 and (output / 'COMPLETE.json').exists() else 'failed'
record.write_text(json.dumps(state, indent=2))
