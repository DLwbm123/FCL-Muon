import fcntl,json,os,signal,subprocess,sys,time,math
from pathlib import Path
from research.adamw_t10.storage import allocated,lock,write


def choose_gpu(required):
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
    choices=[(int(free),int(index)) for index,free in (r.split(',') for r in rows) if int(free)>=required]
    return str(max(choices,key=lambda x:(x[0],-x[1]))[1]) if choices else None


def main():
    root=Path(os.environ['STUDY_ROOT']);reports=root/'src/reports'
    with (root/'.controller.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        resume=(reports/'job.json').exists()
        if resume and not (reports/'ENTRY_FIX_1.json').exists():raise RuntimeError('ledger exists without audited amendment')
        f=json.loads((reports/'FROZEN_PROTOCOL.json').read_text())
        if json.loads((reports/'checks.json').read_text())['status']!='passed':raise RuntimeError('CPU checks missing')
        if resume:
            job=json.loads((reports/'job.json').read_text())
            if job['active'] or job['started']!=f['started'] or job['deadline']!=f['deadline']:raise RuntimeError('unsafe ledger recovery')
            for pid in (job.get('worker_pid',0),job.get('controller_pid',0)):
                path=Path('/proc')/str(pid)/'cmdline'
                if pid and path.exists() and path.read_bytes():raise RuntimeError('previous process still active')
            job.update(status='audited_preflight_recovery',controller_pid=os.getpid())
        else:
            job={k:f[k] for k in ('started','deadline')};job.update(active=False,status='registered',gpu_seconds_used=0.,controller_pid=os.getpid(),required_free_mib=4096)
        stopped=False
        def stop(sig,frame):
            nonlocal stopped
            stopped=True
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        for action,arm in [('preflight','rrs_shared'),('frozen','frozen'),('train','identity_selected'),('train','rrs_shared')]:
            gpu=None
            while gpu is None:
                if stopped or time.time()>=job['deadline']-300:
                    job.update(status='hard_stop_before_launch',active=False);write(root,reports/'job.json',job);return
                gpu=choose_gpu(job['required_free_mib'])
                if gpu is None:
                    job.update(status='waiting_for_gpu_memory',active=False);write(root,reports/'job.json',job);time.sleep(10)
            name=action+'_'+arm+('_entry1' if action=='preflight' and resume else '');begin=time.monotonic()
            env=dict(os.environ,ACTION=action,ARM=arm,CUDA_VISIBLE_DEVICES=gpu)
            with (reports/(name+'.log')).open('x') as log:
                p=subprocess.Popen([sys.executable,str(root/'worker.py')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                job.update(status=name+'_running',active=True,worker_pid=p.pid,gpu=gpu,action=action,arm=arm)
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
                job.update(active=False,exit_code=code,status=name+('_completed' if code==0 else '_failed'));write(root,reports/'job.json',job)
                if code:return
                receipt=reports/arm/'COMPLETE.json' if action=='train' else reports/('frozen_COMPLETE.json' if action=='frozen' else 'preflight.json')
                if not receipt.exists():raise RuntimeError('successful exit without receipt')
                if action=='preflight':
                    measured=json.loads(receipt.read_text())['peak_cuda_reserved_bytes']/(1024**2)
                    job['required_free_mib']=max(4096,math.ceil(measured*1.5)+512)
        job.update(status='completed',active=False,finished=time.time());write(root,reports/'job.json',job)
        write(root,reports/'COMPLETE.json',dict(status='completed',arms=['frozen_dense','frozen64','B_selected_identity','A_rrs_identity','C_rrs_full','M_rrs_mean_only'],reference_D='existing completed UT64 seed0; no retraining',gpu_seconds_used=job['gpu_seconds_used']))


if __name__=='__main__':main()
