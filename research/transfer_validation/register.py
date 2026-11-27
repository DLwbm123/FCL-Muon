"""Freeze all lane inputs and launch the detached cohort controller."""

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from fcl.data import load_raw,manifest


def write(path,value):
    with path.open('x') as f:
        json.dump(value,f,indent=2)
        f.write('\n')


def main():
    root=Path.cwd()
    prior=Path(os.environ['PRIOR_ROOT'])
    scientific=json.loads((root/'protocol.json').read_text())
    src=json.loads((prior/'src/reports/FROZEN_PROTOCOL.json').read_text())
    budget=scientific['budgets']
    started=time.time()
    deadline=started+budget['wall_seconds']
    previous_manifest=prior/'src/manifests'/src.get('manifest_name','seed0.json')
    ref_checkpoint=prior/'src/runs/analytic/latest.pt'
    prior_rrs=prior/'src/runs/rrs_shared/latest.pt'
    source_data=(prior/'src/data').resolve()
    asset_stat=dict(src['asset_stat'])
    common_sources=root/'src/sources/fed-mammoth'
    for p in (ref_checkpoint,prior_rrs,previous_manifest,
              *(common_sources/'imagenet_r_utils').glob('*.yaml')):
        s=p.stat()
        asset_stat[str(p)]=[s.st_size,s.st_mtime_ns,s.st_ino]
    identifiers=[p.relative_to(root) for base in ('src/fcl','src/research')
                 for p in (root/base).rglob('*.py')]
    identifiers.extend(Path(p) for p in ('protocol.json','entry.py','control.py','register.py'))
    code_hash={str(p):hashlib.sha256((root/p).read_bytes()).hexdigest() for p in identifiers}
    base_config=dict(src['config'])
    common=dict(config=base_config,weights=src['weights'],environment=src['environment'],
                asset_stat=asset_stat,source_sha256={str(root/p):value for p,value in code_hash.items()},
                code_identity=os.environ['SOURCE_REVISION'],started=started,deadline=deadline,
                rrs_branch_modes=scientific['rrs_branch_modes'],status='frozen',
                task_end_statistics=scientific['task_end_statistics'],
                retain_stage_checkpoints=scientific['retain_stage_checkpoints'],
                manifest_name='frozen.json')
    # The pre-existing validator resolves absolute paths without depending on worker cwd.
    cohorts=root/'splits'
    cohorts.mkdir(exist_ok=False)
    jobs=[]

    def create_split(tag,config):
        out=cohorts/(tag+'.json')
        raw,cpt,n=load_raw(config['dataset'],source_data,download=False)
        if cpt!=config['cpt'] or n!=config['tasks']*cpt:
            raise ValueError('Invalid configured class count')
        m=manifest(raw,cpt,n,config,out)
        for name, labels in m['labels'].items():
            if set(labels)!=set(range(n)):
                raise ValueError('Dataset has an uncovered class')
            for t, groups in enumerate(m['partitions'][name]):
                if len(groups)!=config['clients'] or (name=='train' and not all(groups)):
                    raise ValueError('Manifest has wrong client coverage')
                actual={labels[i] for g in groups for i in g}
                if actual!=set(range(t*cpt,(t+1)*cpt)):
                    raise ValueError('Manifest task differs from fixed class order')
        return out

    def lane(name,operation,config,split,spec=None,depends=(),settings=None):
        folder=root/name
        for rel in ('src/reports','src/runs','src/manifests','tmp','cache'):
            (folder/rel).mkdir(parents=True,exist_ok=False)
        (folder/'src/data').symlink_to(source_data,target_is_directory=True)
        (folder/'reference').symlink_to(root/'reference',target_is_directory=True)
        (folder/'src/manifests/frozen.json').symlink_to(split)
        freeze=dict(common,config=config,manifest_sha256=hashlib.sha256(split.read_bytes()).hexdigest())
        if spec:
            freeze['analytic_baselines']=spec
        write(folder/'src/reports/FROZEN_PROTOCOL.json',freeze)
        receipt={
            'replay':'src/reports/analytic/COMPLETE.json',
            'fit':'src/reports/rrs_shared/COMPLETE.json',
        }.get(operation,'src/reports/COMPLETE.json')
        jobs.append(dict(lane=name,operation=operation,receipt=receipt,depends=list(depends),
                         required_mib=18000 if operation=='replay' and not settings else 4096,
                         settings=settings or {}))

    analytic=dict(src['analytic_baselines'],matched_lda=True)
    lane('a0','check',base_config,previous_manifest)
    lane('a1','replay',base_config,previous_manifest,
         dict(analytic,fedran_ridge_controls={'fedran_ridge_1e5':1e5,'fedran_ridge_1e7':1e7}),
         settings={'WARMSTART_CHECKPOINT':str(ref_checkpoint)})
    lane('a2','audit',base_config,previous_manifest,settings={'INPUT_CHECKPOINT':str(prior_rrs)})
    for idx,group in enumerate(scientific['groups'],1):
        tag=f'b{idx}'
        cfg=dict(base_config,dataset=group['dataset'],seed=group['seed'],cpt=group['classes_per_task'])
        split=create_split(tag,cfg)
        adap=dict(analytic,seed=group['seed'],rounds=group['analytic_first_task_rounds'])
        lane(tag+'0','fit',cfg,split,depends=('a0',))
        lane(tag+'1','replay',cfg,split,adap,depends=('a0',))
        lane(tag+'2','frozen',cfg,split,depends=(tag+'0',),
             settings={'INPUT_CHECKPOINT':str(root/(tag+'0')/'src/runs/rrs_shared/T1.pt')})
    write(root/'run.json',dict(jobs=jobs,budgets=budget,started=started,deadline=deadline))
    env=dict(os.environ,COHORT_ROOT=str(root),PYTHONPATH=str(root/'src')+':'+str(prior/'src/deps'),
             PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',
             HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    remaining=math.floor(deadline-time.time()-35)
    if remaining<=120:
        raise TimeoutError('Registration exhausted the frozen deadline')
    with (root/'controller.log').open('x') as log:
        run=subprocess.Popen(['timeout','--signal=TERM','--kill-after=30s',str(remaining),sys.executable,
                              str(root/'control.py')],env=env,stdout=log,stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL,start_new_session=True)
    receipt=dict(timeout_pid=run.pid,started=started,deadline=deadline)
    write(root/'launch.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
