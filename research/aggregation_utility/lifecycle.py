"""Explicit same-experiment prefix fork with exact RNG and lifecycle checks."""
import copy
import torch
from fcl.checkpoint import restore_rng


def import_prefix(pilot, checkpoint):
    if checkpoint['schema']!='cosine-head-regmean-pilot-1':
        raise ValueError('Unexpected prefix schema')
    state=checkpoint['state']
    if (state['task'],state['round'],state['phase'])!=(1,0,'task_start'):
        raise ValueError('Prefix must be exactly the committed T1 boundary')
    if state['merges']!=[0] or state['history'].committed!={0} or checkpoint['head_count']!=1:
        raise ValueError('Prefix lifecycle incomplete')
    if checkpoint['clock']!=pilot.clock or checkpoint['identity']!=pilot.identity:
        raise ValueError('Prefix clock or data identity mismatch')
    for key,value in pilot.cfg.items():
        if key not in {'selection_mode','output'} and checkpoint['config'].get(key)!=value:
            raise ValueError('Prefix config mismatch: '+key)
    if checkpoint['config']['selection_mode']!='prefix':
        raise ValueError('Source is not the shared original-RRS prefix')
    pilot.model.add_head(pilot.cfg['cpt'])
    pilot.model.load_state_dict(checkpoint['model'],strict=True)
    pilot.s=copy.deepcopy(state)
    for name,a in pilot.model.adapters.items():
        a.merged=checkpoint['merged_flags'][name]
        if not a.merged or torch.count_nonzero(a.Bq) or torch.count_nonzero(a.Bv):
            raise ValueError('Prefix not merged and zeroed')
    pilot.s['prefix_provenance']={'shared_tasks':[0],'reused_training_visits':sum(v['visited'] for v in state['training_records']),
                                 'source':'runs/prefix/latest.pt','same_clock':True}
    restore_rng(checkpoint['rng'])
    pilot.save()
