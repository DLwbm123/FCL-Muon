"""Freeze one rank-64 codec experiment before any new real-data outcomes."""
import copy
import hashlib
import json
import time
from pathlib import Path

root=Path(__file__).resolve().parent;reports=root/'src/reports'
if (reports/'FROZEN_PROTOCOL.json').exists():raise RuntimeError('already frozen')
baseline=json.loads((reports/'BASELINE_PROTOCOL.json').read_text())
result=json.loads((reports/'BASELINE_T10.json').read_text())
clock=json.loads((reports/'PREPARATION_CLOCK.json').read_text())
checks=json.loads((reports/'checks.json').read_text())
if checks['status']!='passed':raise ValueError('CPU checks not passed')
cfg=copy.deepcopy(baseline['config'])
cfg['source_revision']='0604b844faf364c4ceb2d211fe94c75f0852ab3a'
cfg['rrs_compression']={'mode':'lowrank_diag','rank':64,'factor_dtype':'float32','residual_dtype':'float32',
                        'wire_header':'little-endian uint32 version/dimension/rank + 4 float64 diagnostics',
                        'server_statistics':'decoded Gram for RRS, selected pool, history and historical exposure constraints'}
sources={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
         for base in (root/'src/fcl',root/'src/research') for p in base.rglob('*.py')}
for name in ('settings.json','worker.py','controller.py','configure.py','register.py','launch.py'):
    sources[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
for path,stat in baseline['asset_stat'].items():
    s=Path(path).stat()
    if [s.st_size,s.st_mtime_ns,s.st_ino]!=stat:raise ValueError('immutable asset changed')
manifest_sha=hashlib.sha256((root/'src/manifests/seed0.json').read_bytes()).hexdigest()
if manifest_sha!=baseline['manifest_sha256']:raise ValueError('baseline manifest changed')
d=768;k=64;messages=12*10*10;wire_per_message=44+(d*k+d)*4
before=12*10*10*d*d*8;after=messages*wire_per_message
f={'status':'frozen_before_new_real_training','registered_at':time.time(),'started':clock['started'],
   'deadline':clock['started']+86400,'authorization':'User: compress RRS statistics, run full ten tasks, compare communication and AIA',
   'config':cfg,'weights':baseline['weights'],'asset_stat':baseline['asset_stat'],'environment':baseline['environment'],
   'manifest_sha256':manifest_sha,'source_sha256':sources,
   'code_identity':hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest(),
   'baseline':{'source_commit':'0604b844faf364c4ceb2d211fe94c75f0852ab3a','tasks':10,'seed':0,
               'raw_AIA':result['raw_AIA'],'centered_AIA':result['centered_AIA'],
               'protocol_code_identity':baseline['code_identity'],'reuse_verified_result_no_retraining':True},
   'changes':'Only RRS client-statistic wire encoding and server decoded statistics change; original AdamW/head/route formal hyperparameters unchanged',
   'codec':'Client forms dense local Gram, eigendecomposes it, sends top-64 square-root eigenfactor + nonnegative diagonal residual in FP32. Server decodes a PSD approximate Gram. No full client Gram enters server decisions/history.',
   'limitations':['PSD and near-exact diagonal/trace preservation are not a Loewner upper bound on the original dense Gram.',
                 'RRS/generalized eigenspace, candidate pool, history and risk-repair reference now use approximate decoded statistics; original dense-risk conservation is not guaranteed.',
                 'Low-rank approximation and FP32 quantization are tested as a single fixed codec, not isolated causal effects.',
                 'This is a mechanism/efficiency experiment, not a claim that the compression codec itself is novel.',
                 'Server reconstructs full matrices; this experiment reduces wire bytes, not necessarily server memory or client computation.'],
   'primary':'Full T1..T10 raw all-seen AIA difference from the completed uncompressed seed0 method, in percentage points',
   'secondary':'fixed centered AIA, final raw/centered accuracy, forgetting, task matrices, compression error and compute overhead',
   'communication_primary':{'scope':'RRS actual serialized logical upload bytes, including codec headers and four scalar diagnostics',
                            'baseline_dense_FP64_tensor_bytes':before,'planned_compressed_wire_bytes':after,
                            'planned_RRS_reduction_fraction':1-after/before,
                            'planned_formal_total_bytes_if_other_counts_unchanged':10768588800-before+after},
   'communication_limits':'Formal protocol logical payloads are counted separately from selected-route calls. Full distributed selector/outer-derivative traffic and transport/initial-backbone distribution are not measured; do not call formal counters complete network traffic.',
   'training':'Fresh same pretrained weights T1..T10 seed0, unchanged 10 clients x 5 rounds x 5 epochs, batch16; no old prefix reuse',
   'selection':'original current-task train support/query response selection; compressed H metric; no test-score tuning',
   'gates':'no three-seed or proximal-gain gate; no rank/epoch/round/seed changes or early stopping by intermediate test scores',
   'budget':baseline['budget'],
   'stop':'Complete all ten tasks, reload each evaluation, audit final merged checkpoint, preserve all results; hard limits stop first',
   'failures':'Keep evidence/costs and original new-study clock; numeric/scientific failure requires audit, no automatic restart',
   'source_adaptation':'Original head initializer extracts a client-statistic aggregation hook. Default hook preserves dense arithmetic order; compressed override adds only decoded wire matrices. Runtime factory dispatches by frozen config; no proximal operator.'}
(reports/'FROZEN_PROTOCOL.json').write_text(json.dumps(f,indent=2))
print(json.dumps({'frozen_sources':len(sources),'clock':clock,'planned_RRS_bytes':after,'reduction':1-after/before,'checks':checks}))
