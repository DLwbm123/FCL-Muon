"""Single detached launch; external time limit shares the original clock."""
import json,os,subprocess,time
from pathlib import Path

root=Path(os.environ['PROJECT_ROOT']);report=root/'reports'
assert not (report/'LAUNCH.json').exists() and not (report/'job.json').exists()
origin=json.loads((report/'ORIGIN.json').read_text())
remaining=int(origin['started']+int(os.environ.get('EXTERNAL_MAX_SECONDS','86200'))-time.time())
assert remaining>0
env=dict(os.environ,ENTRY_FILE='scripts/spectra_job.py',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='0',
         PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',
         HF_HOME=str(root/'cache'),TORCH_HOME=str(root/'cache'),TMPDIR=str(root/'tmp'),
         PYTHONPATH=str(root/'deps')+':'+str(root))
env.pop('ENTRY_MODULE',None)
with (report/'LAUNCH.json').open('x') as f:json.dump({'claimed':time.time(),'original_started':origin['started']},f)
with (report/'supervisor.log').open('x') as log:
    child=subprocess.Popen(['timeout','-k','60s',str(remaining)+'s',os.environ['PYTHON_EXECUTABLE'],os.environ['WORKER_ENTRY']],
          env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(report/'LAUNCH.json').write_text(json.dumps({'launched':time.time(),'original_started':origin['started'],
    'timeout_pid':child.pid,'hard_timeout_seconds_remaining':remaining,'external_kill_grace_seconds':60},indent=2))
print(json.dumps({'timeout_pid':child.pid,'original_started':origin['started'],'remaining_seconds':remaining}))
