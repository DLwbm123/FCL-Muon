"""Bounded synthetic preflight, with durable original study clock."""
import json, os, signal, subprocess, sys, time
from pathlib import Path


def write(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    os.replace(tmp, path)


def allocated(root):
    total = 0
    for path in root.rglob('*'):
        try:
            if path.is_file() and not path.is_symlink(): total += path.stat().st_blocks * 512
        except FileNotFoundError:
            pass
    return total


def main():
    root = Path(os.environ['PROJECT_ROOT'])
    report = root/'reports'; report.mkdir(exist_ok=True)
    state_path = report/'job.json'
    attempt = int(os.environ.get('PREFLIGHT_ATTEMPT', '1'))
    with (report/f'preflight_claim_{attempt}.json').open('x') as claim:
        json.dump({'time':time.time(),'attempt':attempt},claim)
    if attempt == 1:
        with state_path.open('x') as f:
            started = time.time()
            json.dump({'started':started,'deadline':started+86400,'gpu_seconds_used':0,
                       'status':'preflight_starting','write_stop_bytes':19_000_000_000}, f)
    state = json.loads(state_path.read_text())
    if attempt > 1 and state['status'] not in ('preflight_failed', 'preflight_passed'):
        raise ValueError('previous preflight has not finished')
    prior_gpu = state['gpu_seconds_used']
    if time.time() >= state['deadline'] or prior_gpu >= 86400:
        raise TimeoutError('original budget exhausted')
    child = None
    begin = time.monotonic()
    try:
        gpu = os.environ['CUDA_VISIBLE_DEVICES']
        free = int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free',
                      '--format=csv,noheader,nounits'],text=True).strip())
        # The user permits shared devices when enough memory remains.
        # Keep admission conservative; this does not reserve other jobs' space.
        required_free = int(os.environ.get('PREFLIGHT_MIN_FREE_MIB', '12000'))
        if free < required_free: raise RuntimeError('insufficient free GPU memory')
        optimizer = os.environ.get('PREFLIGHT_OPTIMIZER', 'adamw')
        if optimizer not in ('adamw', 'muon'):
            raise ValueError('invalid optimizer')
        env = dict(os.environ, ENTRY_FILE='research/route_compare_20261102/preflight.py',
                   PREFLIGHT_CLIENTS='10', PREFLIGHT_ROUNDS='2', PREFLIGHT_OPTIMIZER=optimizer)
        with (report/f'preflight_{attempt}.log').open('x') as log:
            child = subprocess.Popen(['/usr/bin/timeout','-k','15s','1800s',sys.executable,
                                      os.environ['WORKER_ENTRY']],env=env,stdout=log,
                                     stderr=subprocess.STDOUT,start_new_session=True)
            state.update(status='preflight_running',free_mib=free,attempt=attempt,optimizer=optimizer);write(state_path,state)
            while child.poll() is None:
                state['gpu_seconds_used'] = prior_gpu + time.monotonic()-begin
                state['allocated_bytes'] = allocated(root)
                if (time.time() >= state['deadline'] or state['gpu_seconds_used'] >= 86400 or
                        state['allocated_bytes'] >= state['write_stop_bytes']):
                    os.killpg(child.pid,signal.SIGTERM)
                    try: child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGKILL);child.wait()
                    state['status']='budget_stop';break
                write(state_path,state);time.sleep(5)
            state.update(preflight_exit=child.returncode,
                         gpu_seconds_used=prior_gpu + time.monotonic()-begin,
                         status='preflight_passed' if child.returncode == 0 else 'preflight_failed')
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGKILL);child.wait()
        state.update(status='supervisor_failed',error=repr(error),gpu_seconds_used=prior_gpu + time.monotonic()-begin)
        raise
    finally:
        state['updated']=time.time();state['allocated_bytes']=allocated(root);write(state_path,state)

if __name__ == '__main__': main()
