"""One exclusive supervisor and one GPU worker, two complete T10 arms."""
import fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
from research.adamw_t10.controller import choose_gpu
from research.adamw_t10.storage import allocated,lock,write


def main():
    root=Path(os.environ['STUDY_ROOT']);reports=root/'src/reports'
    with (root/'.controller.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (reports/'job.json').exists():raise RuntimeError('existing ledger; audit required')
        f=json.loads((reports/'FROZEN_PROTOCOL.json').read_text());job={k:f[k] for k in ('started','deadline')}
        job.update(status='registered',active=False,gpu_seconds_used=0.,controller_pid=os.getpid())
        stopped=False
        def stop(sig,frame):
            nonlocal stopped
            stopped=True
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        for action,arm in [('preflight','gram_codec'),('diagnostic','unified'),('train','gram_codec'),('train','unified')]:
            gpu=None
            while gpu is None:
                if stopped or time.time()>=job['deadline']-300:
                    job.update(status='hard_stop_before_launch',active=False);write(root,reports/'job.json',job);return
                gpu=choose_gpu()
                if gpu is None:
                    job.update(status='waiting_for_gpu_memory',active=False);write(root,reports/'job.json',job);time.sleep(10)
            env=dict(os.environ,ACTION=action,ARM=arm,CUDA_VISIBLE_DEVICES=gpu)
            name=action+'_'+arm;begin=time.monotonic()
            with (reports/(name+'.log')).open('x') as log:
                p=subprocess.Popen([sys.executable,str(root/'worker.py')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                job.update(status=name+'_running',active=True,worker_pid=p.pid,gpu=gpu,action=action,arm=arm,updated=time.time())
                job.pop('exit_code',None);write(root,reports/'job.json',job)
                while p.poll() is None:
                    used=job['gpu_seconds_used']+time.monotonic()-begin
                    with lock(root):size=allocated(root)
                    job.update(gpu_seconds_live=used,allocated_bytes=size,updated=time.time());write(root,reports/'job.json',job)
                    if stopped or time.time()>=job['deadline']-120 or used>=86280 or size>=18_000_000_000:
                        os.killpg(p.pid,signal.SIGTERM)
                        try:p.wait(timeout=25)
                        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                        job['reason']='signal_or_resource_hard_stop';break
                    time.sleep(5)
                code=p.wait();job['gpu_seconds_used']+=time.monotonic()-begin
                job.update(active=False,exit_code=code,status=name+('_completed' if code==0 else '_failed'))
                write(root,reports/'job.json',job)
                if code:return
                receipt=reports/arm/'COMPLETE.json' if action=='train' else reports/(action+'.json')
                if not receipt.exists():raise RuntimeError('missing success receipt')
        job.update(status='completed',active=False,finished=time.time());write(root,reports/'job.json',job)
        results={a:json.loads((reports/a/'COMPLETE.json').read_text()) for a in ('gram_codec','unified')}
        write(root,reports/'COMPARISON.json',{a:{'raw_AIA':r['raw_AIA'],'centered_AIA':r['centered_AIA'],
            'raw_final':r['stages'][-1]['all_seen'],'centered_final':r['stages'][-1]['centered']['all_seen'],
            'communication':r['communication']} for a,r in results.items()})


if __name__=='__main__':main()
