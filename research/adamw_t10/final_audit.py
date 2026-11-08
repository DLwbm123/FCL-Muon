"""Read-only final merged-checkpoint audit, charged to the original study budget."""
import ctypes
import fcntl
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path


def main():
    root=Path(__file__).resolve().parent
    settings=json.loads((root/'settings.json').read_text());os.environ.update(settings)
    sys.path[:0]=settings['PYTHONPATH'].split(':')
    parent=os.getppid();ctypes.CDLL(None).prctl(1,signal.SIGKILL)
    if parent==1 or os.getppid()!=parent:raise RuntimeError('parent lost')
    import torch
    from research.adamw_t10.runtime import setup,build
    from research.adamw_t10.pilot import centered_logits
    from research.adamw_t10.storage import write,lock,allocated
    reports=root/'src/reports';path=reports/'job.json'
    with (root/'.controller.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        job=json.loads(path.read_text())
        if job['status']!='completed' or job['active'] or job['train_exit']!=0:raise ValueError('training not quiescent')
        if (reports/'FINAL_VALIDATION.json').exists():raise ValueError('already audited')
        if time.time()+180>=job['deadline'] or job['gpu_seconds_used']+180>=86400:raise TimeoutError('original budget insufficient')
        original={k:job[k] for k in ('started','deadline','gpu_seconds_used','finished')}
        write(root,reports/'BUDGET_BEFORE_FINAL_AUDIT.json',job)
        start=time.monotonic();job.update(status='final_audit_running',active=True,worker_pid=os.getpid())
        write(root,path,job);passed=False
        try:
            setup();p=build({k:job[k] for k in ('started','deadline')});p.restore()
            s=p.s
            if s['phase']!='complete' or s['task']!=10 or s['merges']!=list(range(10)) or s['history'].committed!=set(range(10)):
                raise ValueError('incomplete final checkpoint')
            if any(not a.merged or torch.count_nonzero(a.Bq) or torch.count_nonzero(a.Bv) for a in p.model.adapters.values()):
                raise ValueError('unmerged final checkpoint')
            saved=torch.load(p.out/'logits_T10.pt',map_location='cpu',weights_only=False)
            raw=torch.zeros(100,100,dtype=torch.int64);center=raw.clone();offset=0;error=0.
            restricted=0
            with torch.no_grad():
                for t in range(10):
                    for x,y in p.loader(p.raw[1],p.ids('test',t),p.cfg['batch']):
                        p.budget();z=p.model(p.test_tf(x.to(p.device))).cpu();end=offset+len(y)
                        if not torch.equal(y,saved['labels'][offset:end]):raise ValueError('test order changed')
                        error=max(error,float((z-saved['logits'][offset:end]).abs().max()))
                        raw+=torch.bincount(y*100+z.argmax(1),minlength=10000).reshape(100,100)
                        center+=torch.bincount(y*100+centered_logits(z,10).argmax(1),minlength=10000).reshape(100,100)
                        restricted+=int((z[:,t*10:(t+1)*10].argmax(1)+t*10==y).sum());offset=end
            if offset!=10000:raise ValueError('incomplete final test access')
            if raw.tolist()!=s['stages'][-1]['confusion'] or center.tolist()!=s['stages'][-1]['centered']['confusion']:
                raise ArithmeticError('final merged checkpoint predictions changed')
            visits=sum(v['visited'] for v in s['training_records']);steps=sum(v['steps'] for v in s['training_records'])
            if visits!=1250000 or len(s['training_records'])!=500:raise ValueError('formal training budget incomplete')
            route=s.get('route_records',[])
            if len(route)!=9 or any(v['optimizer']!='adamw' or v['route']!='selected' for v in route):raise ValueError('selected route missing')
            result={'status':'passed','read_only':True,'original_clock':{k:job[k] for k in ('started','deadline')},
                'checkpoint_phase':'complete','all_adapters_merged_zero_B':True,'formal_tasks':10,'formal_clients':500,
                'formal_training_visits':visits,'formal_training_steps':steps,'selected_route_tasks':len(route),
                'no_proximal_operator':not hasattr(p,'response_server_operator'),
                'final_readback_images':offset,'raw_confusion_equal':True,'centered_confusion_equal':True,
                'max_merged_vs_premerge_logit_difference':error,'restricted_accuracy_diagnostic':restricted/offset,
                'additional_test_loader_visits':p.accesses['test_loaded'],'additional_train_loader_visits':p.accesses['train_loaded'],
                'communication':s['communication'],'route_changed_layers':[v['changed_layers'] for v in route],
                'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
                'audit_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
            write(root,reports/'FINAL_VALIDATION.json',result);passed=True
        finally:
            elapsed=time.monotonic()-start;job['gpu_seconds_used']+=elapsed
            job.update(status='completed' if passed else 'final_audit_failed',active=False,final_audit_seconds=elapsed,
                       final_audit_passed=passed,original_training_finished=original['finished'],final_audit_finished=time.time())
            with lock(root):job['allocated_bytes']=allocated(root)
            write(root,path,job)
        print(json.dumps({'audit_passed':passed,'gpu_seconds':job['gpu_seconds_used']}),flush=True)


if __name__=='__main__':main()
