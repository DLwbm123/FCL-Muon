"""Freeze an independent no-prox ten-task protocol, preserving prior records."""
import hashlib
import json
import time
from pathlib import Path

root=Path(__file__).resolve().parent;reports=root/'src/reports'
prior=json.loads((reports/'PRIOR_SOURCE_METADATA.json').read_text())
if (reports/'FROZEN_PROTOCOL.json').exists():raise RuntimeError('already frozen')
cfg=prior['base_config'];cfg.update(tasks=10,total_tasks=10,manifest='manifests/seed0.json',
    output='runs/main',selection_mode='selected',required_free_mib=12000,storage_limit_bytes=20_000_000_000,
    write_stop_bytes=18_000_000_000,source_revision='f061aa601b817b86c2bd462c42faff8fb74bde96',
    inference_rule='primary raw all-seen argmax; secondary fixed per-task-block mean subtraction')
cfg['route']=dict(prior['route'],mode='selected',optimizer='adamw')
cfg['retain_round_checkpoints']=False;cfg['retain_task_checkpoints']=False
sources={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
         for base in (root/'src/fcl',root/'src/research') for p in base.rglob('*.py')}
for name in ('settings.json','worker.py','controller.py','configure.py','register.py','launch.py'):
    sources[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
assets={k:v for k,v in prior['asset_stat'].items() if k==prior['weights']['path'] or '/sources/' in k}
for name in ('train','test','meta'):
    p=root/'src/data/cifar-100-python'/name;s=p.stat();assets[str(p.resolve())]=[s.st_size,s.st_mtime_ns,s.st_ino]
# Recheck recorded immutable weights once by stat; retain the earlier full SHA provenance.
for path,stat in assets.items():
    s=Path(path).stat()
    if [s.st_size,s.st_mtime_ns,s.st_ino]!=stat:raise ValueError('asset identity changed')
started=(reports/'PRIOR_SOURCE_METADATA.json').stat().st_mtime
f={'status':'frozen_before_new_real_training','registered_at':time.time(),'started':started,'deadline':started+86400,
   'authorization':'2026-11-08 direct user: current AdamW full T10 versus LoRM AIA, no prox, no three-seed gate',
   'config':cfg,'weights':prior['weights'],'asset_stat':assets,'environment':prior['environment'],
   'manifest_sha256':hashlib.sha256((root/'src/manifests/seed0.json').read_bytes()).hexdigest(),
   'manifest_content_equal_historical_LoRM_seed0':True,'source_sha256':sources,
   'code_identity':hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest(),
   'primary':'AIA = arithmetic mean of raw all-seen accuracies after T1..T10; units fractions in worker, percent in report',
   'secondary':'fixed block mean subtraction AIA; raw and centered final accuracy; forgetting and full task matrices',
   'training':'fresh T1..T10 seed0, 5 rounds x 10 clients x 5 epochs/task, no old T1/checkpoint reuse',
   'selection':'original frozen selected route on current-task train support/query only; no test-score selection',
   'gates':'no proximal arm, no three-seed gate, no score-dependent T10 admission',
   'readout_selection':'raw primary predeclared for this new experiment; earlier centered-primary study retained separately',
   'LoRM':'reuse verified official pinned historical seed0 full T10; whole-scheme historical comparison, not one-module ablation',
   'budget':{'wall_seconds':86400,'conservative_gpu_seconds':86400,'allocated_limit_bytes':20_000_000_000,
             'write_stop_bytes':18_000_000_000,'atomic_write_reserve_bytes':1_500_000_000,'minimum_free_MiB':12000,
             'one_gpu_worker':True,'shared_gpu_allowed':True,'save_and_exit_reserve_seconds':120},
   'failures':'retain failures and original new-study clock; do not automatically restart scientific/numeric failures',
   'cpu_preparation_failure':{'attempt':0,'issue':'neutral entry lacked main guard for multiprocessing spawn',
                              'resolution':'added main guard; selected trajectory and readout checks unchanged; 0 GPU seconds'},
   'weights_sha_provenance':'earlier frozen full SHA reused; current stat identity checked, no repeated large-file hash',
   'stop':'complete after T10 readbacks/merge/history; preserve latest and all stage logits; hard limits never extended'}
(reports/'FROZEN_PROTOCOL.json').write_text(json.dumps(f,indent=2))
print(json.dumps({'frozen_sources':len(sources),'started':started,'deadline':f['deadline'],'checks':json.loads((reports/'checks.json').read_text())}))
