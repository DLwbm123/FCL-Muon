"""Use the same LDA on the RRS representation frozen after task one."""

import os

import torch

from research.adamw_t10.storage import write
from research.stat_ablation.frozen import step_frozen
from research.stat_ablation.runtime import setup,build,validate


def restore_model(pilot,path):
    state=torch.load(path,map_location='cpu',weights_only=False)
    cfg=state['config']
    verify=('dataset','seed','cpt','backbone','rank')
    if any(cfg[key]!=pilot.cfg[key] for key in verify):
        raise ValueError('Checkpoint dataset, seed or model mismatch')
    if state['identity']['weights_sha256']!=pilot.identity['weights_sha256']:
        raise ValueError('Checkpoint pretrained identity mismatch')
    if state['identity']['manifest_sha256']!=pilot.identity['manifest_sha256']:
        raise ValueError('Checkpoint manifest mismatch')
    if state['state']['phase'] not in ('task_start','complete') or not all(state['merged_flags'].values()):
        raise ValueError('Expected a committed merged checkpoint')
    for _ in range(state['head_count']):
        pilot.model.add_head(pilot.cfg['cpt'])
    pilot.model.load_state_dict(state['model'],strict=True)
    for name,adapter in pilot.model.adapters.items():
        adapter.merged=state['merged_flags'][name]
    return state['state']


def main():
    setup()
    root,f=validate()
    pilot=build({k:f[k] for k in ('started','deadline')})
    endpoint=restore_model(pilot,os.environ['INPUT_CHECKPOINT'])
    if endpoint['task']!=1 or endpoint['merges']!=[0]:
        raise ValueError('The frozen control requires the merged T1 endpoint')
    if int(pilot.model.lda_classes)!=pilot.cfg['cpt']:
        raise ValueError('Unexpected source head')
    pilot.cfg.update(primary_mode='dense',branch_modes=['dense','compressed'])
    pilot.out=root/'src/runs/frozen'
    pilot.out.mkdir(exist_ok=False)
    pilot.s['frozen_stats']={mode:None for mode in pilot.cfg['branch_modes']}
    pilot.s['frozen_communication']={mode:{} for mode in pilot.cfg['branch_modes']}
    # Keep the same T1 backbone while refreshing only the analytic classifier.
    for task in range(pilot.cfg['tasks']):
        result=step_frozen(pilot,task)
        print(f'T{task+1} frozen readout {result["branch_readouts"]["compressed"]["all_seen"]:.6f}',flush=True)
    stages=pilot.s['stages']
    write(root,root/'src/reports/COMPLETE.json',dict(
        status='completed',tasks=len(stages),stages=stages,
        training_visits=0,source='same-seed RRS committed T1',
        classification_stats_upload_and_head_download=pilot.s['frozen_communication'],
        all_checkpoint_reloads_verified=True))


if __name__=='__main__':
    main()
