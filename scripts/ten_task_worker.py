"""One complete ten-task run; entry/config paths are environment-only."""
import json,os,subprocess,sys,time
from pathlib import Path

root=Path(os.environ['PROJECT_ROOT']);seed=int(os.environ['LANE_SEED']);method=os.environ['RUN_METHOD']
receipt=root/f'reports/{method}_seed{seed}.json';cfgpath=root/f'configs/ten_{method}_seed{seed}.json'
cfg=json.loads(cfgpath.read_text());out=root/cfg['output']
if receipt.exists() or out.exists():raise SystemExit('Existing run; no automatic retry or clock reset')
state={'seed':seed,'method':method,'status':'starting','started':time.time()}
def save():
    tmp=receipt.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(receipt)
save()
try:
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','-i',os.environ['CUDA_VISIBLE_DEVICES']],text=True).strip())
    assert free>=(24000 if method=='lorm' else 14000),'GPU admission memory insufficient'
    clock=json.loads((root/'reports/job.json').read_text())
    env=dict(os.environ,RUN_CONFIG=str(cfgpath),RUN_STARTED=str(clock['started']))
    env.pop('ENTRY_FILE',None);env.pop('ENTRY_MODULE',None)
    if method=='lorm':env['ENTRY_MODULE']='fcl.lorm_train'
    else:env['ENTRY_FILE']='scripts/head_confirmation_train.py'
    with (root/f'reports/train_{method}_seed{seed}.log').open('x') as log:
        child=subprocess.Popen([sys.executable,'/data/bmw/workers/e17.py'],env=env,stdout=log,stderr=subprocess.STDOUT)
        state.update(status='training',train_pid=child.pid,free_mib_at_admission=free);save();code=child.wait()
    state['training_exit_code']=code
    assert code==0 and (out/'COMPLETE.json').exists(),'Training failed or did not complete all tasks'
    state['status']='qualifying';save()
    from scripts.ten_task_audit import verify_lorm,report
    if method=='lorm':verify_lorm(root,out,cfg)
    else:
        q=json.loads((out/'QUALIFICATION.json').read_text())
        assert q['qualified'] and q['training_visits']==1250000 and q['client_rounds']==500 and q['stats_visits']==300000
        assert q['checkpoint_reload_confusion_equal'] and len(q['centered_matrix'])==10
        q.update(task_evaluation_images=55000,audit_images=10000)
        (out/'QUALIFICATION.json').write_text(json.dumps(q,indent=2))
    report(out,method)
    state.update(status='completed',ended=time.time(),qualified=True);save()
except BaseException as exc:
    state.update(status='failed',ended=time.time(),error=repr(exc));save();raise
