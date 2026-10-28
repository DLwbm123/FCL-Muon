"""Launch explicitly requested C T4-T10, without resetting time or T3 evidence."""
import json,os,subprocess,time
from pathlib import Path
import torch
from research.fcca.lifecycle import config_hash

root=Path(os.environ['PROJECT_ROOT']);report=root/'reports/ten';source=root/'runs/main/latest.pt'
assert os.environ['CUDA_VISIBLE_DEVICES']=='3'
assert not (report/'LAUNCH.json').exists() and not (report/'job.json').exists() and not (root/'runs/ten').exists()
assert json.loads((report/'SYNTHETIC_PREFLIGHT.json').read_text())['exit_code']==0
old=json.loads((root/'reports/job.json').read_text());q=json.loads((root/'runs/main/QUALIFICATION.json').read_text())
assert old['status']=='completed' and old['qualified'] and old['exit_code']==0 and q['qualified']
assert not Path('/proc/'+str(old['worker_pid'])).exists() and not Path('/proc/'+str(old['supervisor_pid'])).exists()
clock={k:old[k] for k in ('started','deadline')};assert q['clock']==clock and time.time()<clock['deadline']
auth=json.loads((report/'AUTHORIZATION.json').read_text())
assert auth['explicit_user_T10_authorization'] and auth['clock']==clock
assert q['reload_images']==3000 and q['reload_raw_predictions_equal'] and q['reload_centered_predictions_equal']
assert q['client_rounds']==150 and q['training_visits']==375000 and q['certificates_passed']==15
cfg=json.loads((root/'configs/fcca_ten_seed0.json').read_text())
state=torch.load(source,map_location='cpu',weights_only=False)
assert state['config']['tasks']==3 and cfg['tasks']==10 and state['clock']==clock
assert q['config_sha256']==config_hash(state['config']) and state['state']['phase']=='complete'
for k in set(cfg)|set(state['config']):
    if k not in ('tasks','output'):assert cfg.get(k)==state['config'].get(k),k
assert state['state']['stages']==q['stages'] and len(state['state']['fcca_records'])==10
expected={'source_commit':json.loads((root/'reports/SOURCE.json').read_text())['source_commit'],
          'completed_tasks':3,'selection_mode':'federated','config_sha256':q['config_sha256'],
          'identity':state['identity'],'clock':clock}
(report/'IMPORT_SOURCE.json').write_text(json.dumps(expected,indent=2));del state
free=int(subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
assert free>=cfg['required_free_mib']
used=0
for directory,_,files in os.walk(root,followlinks=False):
    for name in files:
        p=Path(directory)/name
        if not p.is_symlink():used+=p.stat().st_blocks*512
assert used<cfg['write_stop_bytes'] and os.statvfs(root).f_bavail*os.statvfs(root).f_frsize>cfg['storage_limit_bytes']
probe=root/'tmp/continuation_storage_probe';probe.write_text('read-write-check');assert probe.read_text()=='read-write-check';probe.unlink()
remaining=int(clock['deadline']-time.time()+60);assert remaining>60
with (report/'LAUNCH.json').open('x') as f:json.dump({'claimed':time.time(),'clock':clock},f)
env=dict(os.environ,ENTRY_FILE='scripts/fcca_continue_job.py',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='0',
         PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',HF_HOME=str(root/'cache'),
         TORCH_HOME=str(root/'cache'),TMPDIR=str(root/'tmp'),PYTHONPATH=str(root/'deps')+':'+str(root),SOURCE_CHECKPOINT=str(source))
env.pop('ENTRY_MODULE',None)
with (report/'supervisor.log').open('x') as log:
    child=subprocess.Popen(['timeout','-k','60s',str(remaining)+'s','/data/bmw/envs/n3/bin/python','/data/bmw/workers/e17.py'],
                           env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'launched':time.time(),'timeout_pid':child.pid,'gpu':3,'clock':clock,'remaining_external_timeout_seconds':remaining,
         'free_mib':free,'total_root_allocated_bytes':used,'storage_probe_passed':True,'expected_peak_root_bytes':3000000000,
         'explicit_user_T10_authorization':True,'T3_scientific_progress_gate_passed':False}
(report/'LAUNCH.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
