"""One-shot detached launch with a second, external wall-clock limit."""
import json,os,subprocess,time
from pathlib import Path
root=Path(os.environ['PROJECT_ROOT']);arm=os.environ['EXPERIMENT_ARM']
assert arm in ('A','B') and os.environ['CUDA_VISIBLE_DEVICES']==('2' if arm=='A' else '7')
assert not (root/'reports/job.json').exists() and not (root/'reports/LAUNCH.json').exists()
free=int(subprocess.check_output(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
assert free>=18000
# Mark claim before spawning; failure is explicit and never loops automatically.
with (root/'reports/LAUNCH.json').open('x') as f:json.dump({'claimed':time.time(),'arm':arm,'free_mib':free},f)
env=dict(os.environ,ENTRY_FILE='scripts/boundary_job.py',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='0',
         PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',
         HF_HOME=str(root/'cache'),TORCH_HOME=str(root/'cache'),TMPDIR=str(root/'tmp'),
         PYTHONPATH=str(root/'deps')+':'+str(root))
env.pop('ENTRY_MODULE',None)
with (root/'reports/supervisor.log').open('x') as log:
    p=subprocess.Popen(['timeout','-k','60s','86300s','/data/bmw/envs/n3/bin/python','/data/bmw/workers/e17.py'],
                        env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(root/'reports/LAUNCH.json').write_text(json.dumps({'launched':time.time(),'arm':arm,'free_mib':free,'timeout_pid':p.pid,
        'hard_timeout_seconds':86300,'external_kill_grace_seconds':60},indent=2))
print(json.dumps({'arm':arm,'timeout_pid':p.pid,'gpu':os.environ['CUDA_VISIBLE_DEVICES']}))
