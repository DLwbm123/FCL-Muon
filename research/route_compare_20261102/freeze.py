"""Freeze paired T3 protocol after training-only smoke and before new scores."""
import hashlib, json, os, time
from pathlib import Path
import torch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root=Path(os.environ['PROJECT_ROOT']);source=Path(os.environ['SOURCE_ROOT']);report=root/'reports'
    job=json.loads((report/'job.json').read_text())
    if job['status']!='smoke_completed' or time.time()>=job['deadline']:
        raise ValueError('smoke incomplete or original clock expired')
    elapsed=[];route=None;weight=None
    for optimizer in ('adamw','muon'):
        receipt=json.loads((report/f'smoke_{optimizer}.json').read_text())
        if receipt['status']!='completed_training_only' or receipt['test_evaluations']!=0:
            raise ValueError('not a training-only smoke')
        x=torch.load(root/f'runs/smoke_{optimizer}/latest.pt',map_location='cpu',weights_only=False)
        if (x['state']['task'],x['state']['phase'])!=(0,'ready_evaluation'):
            raise ValueError('checkpoint not at the pre-test boundary')
        r=dict(x['config']['route']);r.pop('optimizer');r.pop('mode')
        if route is None:route=r
        elif route!=r:raise ValueError('optimizer route settings differ')
        if weight is None:weight=x['identity']['weights_sha256']
        elif weight!=x['identity']['weights_sha256']:raise ValueError('weight identity mismatch')
        elapsed.append(float(receipt['elapsed']))
        del x
    if list((root/'runs').glob('**/metrics.json')) or list(report.glob('result_seed*_private.json')):
        raise ValueError('new score artifact exists before freeze')
    # Conservative planning estimate: 4 remaining prefixes, 12 two-task arms,
    # plus up to 108 responses using a 2.5x inflation over the larger GPU
    # preflight; actual resource guard remains the only hard limit.
    estimate=2*sum(elapsed)+12*sum(elapsed)+108*107*2.5
    remaining=86400-job['gpu_seconds_used']
    paths=list((root/'fcl').rglob('*.py'))+list((root/'research').rglob('*.py'))+[root/'configs/aggregation_utility_three_task_seed0.json']
    hashes={str(p.relative_to(root)):digest(p) for p in sorted(paths)}
    manifests=[source/'manifests/seq-cifar100_n10_a0.05_s0.json',root/'manifests/seed1.json',root/'manifests/seed2.json']
    data={'status':'frozen_before_new_test','frozen_at':time.time(),'started':job['started'],
          'deadline':job['deadline'],'tasks':3,'seeds':[0,1,2],
          'arms':{'A0':'adamw+rrs','A1':'adamw+selected','M0':'muon+rrs','M1':'muon+selected'},
          'route':route,'weight_sha256':weight,'manifest_sha256':{str(i):digest(p) for i,p in enumerate(manifests)},
          'source_sha256':hashes,'resources':{'gpu_seconds_cap':86400,'wall_seconds_cap':86400,
             'write_stop_bytes':19000000000,'estimate_gpu_seconds':estimate,
             'remaining_gpu_seconds_before_formal':remaining,'estimate_is_not_a_guarantee':True},
          'selection':'historical trace non-increase at deployed FP32; per-layer spectral repair; actual all-seen/current CE acceptance',
          'trial':'2 short rounds; each client 4 updates per round, batch4, S/Q at most32; paired RNG; sampled Gram scaled by formal N/sample N',
          'evaluation':'raw argmax over all seen classes without task ID','test_scores_before_freeze':0,
          'scope':'current-task query is selection data; Head-RegMean backward is surrogate; T3 only for common endpoints'}
    with (report/'FROZEN.json').open('x') as f:json.dump(data,f,indent=2,allow_nan=False)
    print(json.dumps({'status':data['status'],'estimate_gpu_seconds':estimate,'remaining_gpu_seconds':remaining,'tasks':3}))

if __name__=='__main__':main()
