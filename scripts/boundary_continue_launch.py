"""Audit the final T3 gate and launch B T4-T10 on the original resource clock."""
import hashlib,json,os,subprocess,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from research.boundary_repair.lifecycle import config_hash
from research.boundary_repair.continuation import validate_continuation
from research.basic_repair.fcl_basic_repair_tools import center_task_logits

root=Path(os.environ['PROJECT_ROOT']);report=root/'reports/ten';source=root/'runs/main/latest.pt'
assert os.environ['CUDA_VISIBLE_DEVICES']=='7' and os.environ['EXPERIMENT_ARM']=='B'
assert not (report/'LAUNCH.json').exists() and not (report/'job.json').exists() and not (root/'runs/ten').exists()
assert json.loads((report/'SYNTHETIC_PREFLIGHT.json').read_text())['exit_code']==0
old=json.loads((root/'reports/job.json').read_text());q=json.loads((root/'runs/main/QUALIFICATION.json').read_text())
assert old['status']=='completed' and old['qualified'] and old['exit_code']==0 and q['qualified']
assert not Path('/proc/'+str(old['worker_pid'])).exists() and not Path('/proc/'+str(old['supervisor_pid'])).exists()
clock={k:old[k] for k in ('started','deadline')};assert q['clock']==clock and time.time()<clock['deadline']
assert q['reload_images']==3000 and q['reload_raw_predictions_equal'] and q['reload_centered_predictions_equal']
assert q['client_rounds']==150 and q['training_visits']==375000 and q['certificates_passed']==15
reference=json.loads((report/'LORM_T3_REFERENCE.json').read_text())
assert reference['centered']['n']==3000 and reference['centered']['all_seen_accuracy']==.908
with np.load(root/'runs/main/task3_private_logits.npz') as f:
    indices=f['indices'];labels=f['labels'];scores=f['logits']
    assert len(labels)==3000 and len(np.unique(indices))==3000
    correct=int((center_task_logits(scores,10).argmax(1)==labels).sum())
    raw_correct=int((scores.argmax(1)==labels).sum())
assert correct==round(q['stages'][-1]['centered']['all_seen_accuracy']*3000)
source_state=torch.load(source,map_location='cpu',weights_only=False)
cfg=json.loads((root/'configs/boundary_ten_seed0.json').read_text())
assert q['config_sha256']==config_hash(source_state['config'])
gate={'passed':correct>2724,'test_images':3000,'B_centered_correct':correct,'B_raw_correct':raw_correct,
      'LoRM_centered_correct':2724,'LoRM_raw_correct':round(reference['raw']['all_seen_accuracy']*3000),
      'source_config_sha256':q['config_sha256'],'clock':clock,'checked':time.time(),
      'readout':'fixed block mean subtraction','requires_strict_exceedance':True,
      'reload_qualification_passed':True,'maximum_reload_logit_difference':q['maximum_reload_logit_difference'],
      'selection_disclosure':'Horizon extended conditional on the same development evaluation set; not independent confirmation.'}
validate_continuation(SimpleNamespace(cfg=cfg,clock=clock),source_state,gate)
expected={'source_commit':json.loads((root/'reports/SOURCE.json').read_text())['source_commit'],
          'completed_tasks':3,'selection_mode':'federated','config_sha256':q['config_sha256'],
          'identity':source_state['identity'],'clock':clock,'prior_T1_import':source_state['state']['boundary_import']}
(report/'IMPORT_SOURCE.json').write_text(json.dumps(expected,indent=2))
with (report/'GATE.json').open('x') as f:json.dump(gate,f,indent=2)
del source_state
free=int(subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
assert free>=cfg['required_free_mib']
used=0
for directory,_,files in os.walk(root,followlinks=False):
    for name in files:
        p=Path(directory)/name
        if not p.is_symlink():used+=p.stat().st_blocks*512
assert used<cfg['write_stop_bytes'] and os.statvfs(root).f_bavail*os.statvfs(root).f_frsize>cfg['storage_limit_bytes']
probe=root/'tmp/continuation_storage_probe';probe.write_text('read-write-check');assert probe.read_text()=='read-write-check';probe.unlink()
# The external timeout is the remaining interval, never a fresh 24-hour budget.
remaining=int(clock['deadline']-time.time()+60);assert remaining>60
with (report/'LAUNCH.json').open('x') as f:json.dump({'claimed':time.time(),'clock':clock},f)
env=dict(os.environ,ENTRY_FILE='scripts/boundary_continue_job.py',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='0',
         PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',HF_HOME=str(root/'cache'),
         TORCH_HOME=str(root/'cache'),TMPDIR=str(root/'tmp'),PYTHONPATH=str(root/'deps')+':'+str(root),SOURCE_CHECKPOINT=str(source))
env.pop('ENTRY_MODULE',None)
with (report/'supervisor.log').open('x') as log:
    child=subprocess.Popen(['timeout','-k','60s',str(remaining)+'s','/data/bmw/envs/n3/bin/python','/data/bmw/workers/e17.py'],
                           env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'launched':time.time(),'timeout_pid':child.pid,'gpu':7,'clock':clock,'remaining_external_timeout_seconds':remaining,
         'free_mib':free,'total_root_allocated_bytes':used,'storage_probe_passed':True,'gate':gate}
(report/'LAUNCH.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
