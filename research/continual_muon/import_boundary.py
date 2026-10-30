"""Independent diagnostic import of a committed T1 prefix, without clock forgery."""
import copy
import torch
from fcl.checkpoint import restore_rng


def import_t1(pilot, checkpoint, expected_identity):
    state = checkpoint['state']
    if checkpoint['schema'] != 'cosine-head-regmean-pilot-1':
        raise ValueError('wrong checkpoint schema')
    if (state['task'], state['round'], state['phase']) != (1, 0, 'task_start'):
        raise ValueError('source is not the committed T1 boundary')
    if (state['merges'] != [0] or state['history'].committed != {0} or
            checkpoint['head_count'] != 1 or state['client_states']):
        raise ValueError('incomplete or unexpected source state')
    if checkpoint['identity'] != expected_identity:
        raise ValueError('source identity mismatch')
    if checkpoint['config'].get('selection_mode') != 'prefix':
        raise ValueError('source is not the RRS prefix')
    for key, value in checkpoint['config'].items():
        if key != 'output' and pilot.cfg.get(key) != value:
            raise ValueError('source config mismatch: ' + key)
    if set(pilot.cfg) != set(checkpoint['config']):
        raise ValueError('source config keys differ')
    if not {'started', 'deadline'} <= set(checkpoint['clock']) or not {'started', 'deadline'} <= set(pilot.clock):
        raise ValueError('missing clock provenance')
    if checkpoint['clock'] == pilot.clock:
        raise ValueError('diagnostic must use a new independent clock')
    pilot.model.add_head(pilot.cfg['cpt'])
    pilot.model.load_state_dict(checkpoint['model'], strict=True)
    for name, adapter in pilot.model.adapters.items():
        adapter.merged = checkpoint['merged_flags'][name]
        if not adapter.merged or torch.count_nonzero(adapter.Bq) or torch.count_nonzero(adapter.Bv):
            raise ValueError('source adapters are not merged and zeroed')
    pilot.s = copy.deepcopy(state)
    restore_rng(checkpoint['rng'])
    if not torch.equal(torch.get_rng_state(), checkpoint['rng']['torch']):
        raise ValueError('source CPU RNG was not restored')
    return {'source_clock': copy.deepcopy(checkpoint['clock']), 'diagnostic_clock': copy.deepcopy(pilot.clock),
            'shared_training_visits': sum(v['visited'] for v in state['training_records'])}
