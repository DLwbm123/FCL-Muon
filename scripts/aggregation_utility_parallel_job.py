"""User-authorized two-GPU scheduling, sharing the original clock and storage limit."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def main():
    root = Path(os.environ['PROJECT_ROOT'])
    report, out = root/'reports', root/'runs'
    ready = json.loads((report/'PARALLEL_READY.json').read_text())
    old = json.loads((report/'PARALLEL_ORIGINAL_JOB.json').read_text())
    cfg = json.loads((root/'configs/aggregation_utility_three_task_seed0.json').read_text())
    clock = {k:old[k] for k in ['started','deadline']}
    assert ready['clock'] == clock and ready['lanes'] == {'0':['rrs','central'],'7':['federated']}
    assert time.time() < clock['deadline']
    assert all(not Path('/proc/'+str(pid)).exists() for pid in ready['retired_pids'])
    assert json.loads((out/'PREFIX_READY.json').read_text())['clock'] == clock
    assert cfg['storage_limit_bytes'] == old['storage_limit_bytes']
    assert cfg['write_stop_bytes'] == old['write_stop_bytes']
    # An exclusive receipt prevents this one-time transition from launching twice.
    with (report/'PARALLEL_CLAIM.json').open('x') as f:
        json.dump({'supervisor_pid':os.getpid(),'started':time.time()},f)
    state = dict(clock, status='starting', supervisor_pid=os.getpid(),
                 storage_limit_bytes=cfg['storage_limit_bytes'], write_stop_bytes=cfg['write_stop_bytes'],
                 baseline_allocated_bytes=old['baseline_allocated_bytes'], lanes=[],
                 seed=cfg['seed'], tasks=cfg['tasks'], parallel_authorized=True)
    children, logs = [], []
    requested_stop = False

    def stop(*_):
        nonlocal requested_stop
        requested_stop = True

    def size():
        total = 0
        for directory, _, files in os.walk(root, followlinks=False):
            for name in files:
                p = Path(directory)/name
                try:
                    if not p.is_symlink():
                        total += p.stat().st_blocks*512
                except FileNotFoundError:
                    pass
        return total

    def save():
        write_json(report/'job.json',state)

    def stop_children():
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
        until = time.monotonic()+30
        for child in children:
            try:
                child.wait(timeout=max(.01,until-time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL)
                child.wait()

    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    save()
    try:
        assert size() < cfg['write_stop_bytes']
        for gpu,modes in ready['lanes'].items():
            free = int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free',
                       '--format=csv,noheader,nounits'],text=True).strip())
            assert free >= cfg['required_free_mib'], 'Insufficient memory on GPU'+gpu
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, ENTRY_FILE='scripts/aggregation_utility_train.py',
                       RESUME_RUN='1', PARALLEL_RUN='1', ARM_MODES=','.join(modes))
            env.pop('ENTRY_MODULE',None)
            log = (report/('worker_parallel_gpu'+gpu+'.log')).open('x');logs.append(log)
            child = subprocess.Popen([sys.executable,'/data/bmw/workers/e17.py'],env=env,
                                     stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children.append(child)
            state['lanes'].append({'gpu':gpu,'modes':modes,'worker_pid':child.pid,
                                   'started':time.time(),'free_mib_at_admission':free})
            save()
        state['status'] = 'running';save()
        while any(child.poll() is None for child in children):
            used = size();state.update(allocated_bytes=used,checked=time.time())
            for child,lane in zip(children,state['lanes']):
                lane['exit_code'] = child.poll()
                if lane['exit_code'] is not None and 'ended_observed' not in lane:
                    lane['ended_observed'] = time.time()
            failed = any(child.poll() not in (None,0) for child in children)
            if requested_stop or time.time() >= clock['deadline'] or used >= cfg['write_stop_bytes'] or failed:
                state['status'] = 'failed' if failed else 'budget_or_signal_stop'
                save();stop_children();break
            save();time.sleep(10)
        for child,lane in zip(children,state['lanes']):
            lane['exit_code'] = child.returncode
        if state['status'] == 'running':
            assert all(child.returncode == 0 for child in children)
            qualifications = {}
            for gpu,modes in ready['lanes'].items():
                lane = json.loads((out/('LANE_GPU'+gpu+'_COMPLETE.json')).read_text())
                assert lane['qualified'] and lane['clock'] == clock and set(lane['arms']) == set(modes)
                qualifications.update(lane['arms'])
            for mode in ['rrs','central','federated']:
                q = json.loads((out/mode/'QUALIFICATION.json').read_text())
                assert q['qualified'] and q['clock'] == clock and (out/mode/'COMPLETE.json').exists()
            write_json(out/'STUDY_QUALIFICATION.json',dict(qualified=True,
                prefix=json.loads((out/'PREFIX_READY.json').read_text()),arms=qualifications,
                completed_record_training_visits=875000,attributed_training_visits_including_prefix_per_arm=375000,
                user_requested_gpu_migration=json.loads((report/'GPU0_MIGRATION_READY.json').read_text()),
                user_requested_parallelization=ready,actual_client_rounds=350,actual_base_statistics_visits=210000,
                actual_task_eval_images=16000,actual_reload_images=9000,clock=clock))
            write_json(out/'COMPLETE.json',{'completed':time.time(),'qualified':True})
            state.update(status='completed',qualified=True)
    except BaseException as exc:
        state.update(status='supervisor_failure',error=repr(exc))
        stop_children()
        raise
    finally:
        for log in logs:
            log.close()
        state.update(ended=time.time(),allocated_bytes=size())
        save()


if __name__ == '__main__':
    main()
