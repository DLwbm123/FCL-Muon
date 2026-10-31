"""Strict, independent-clock import of an actual captured T2 boundary."""
import copy
import numpy as np
import torch
from fcl.checkpoint import restore_rng


def import_boundary(pilot, checkpoint, snapshot, expected_identity):
    state = checkpoint['state']
    if checkpoint['schema'] != 'cosine-head-regmean-pilot-1' or checkpoint['head_count'] != 2:
        raise ValueError('unexpected source model schema')
    if (state['task'], state['round'], state['phase']) != (1, pilot.cfg['rounds'], 'ready_evaluation'):
        raise ValueError('source is not the completed T2 before evaluation')
    if state['history'].committed != {0} or len(state['pending']) != len(pilot.model.adapters):
        raise ValueError('source history is not T1 plus pending T2')
    if checkpoint['identity'] != expected_identity or snapshot['identity'] != expected_identity:
        raise ValueError('source identity mismatch')
    if snapshot['clock'] != checkpoint['clock'] or pilot.clock == checkpoint['clock']:
        raise ValueError('source or independent diagnostic clock mismatch')
    a, b = snapshot['rng'], checkpoint['rng']
    if (a['python'] != b['python'] or a['numpy'][0] != b['numpy'][0] or
            not np.array_equal(a['numpy'][1], b['numpy'][1]) or a['numpy'][2:] != b['numpy'][2:] or
            not torch.equal(a['torch'], b['torch']) or len(a['cuda']) != len(b['cuda']) or
            any(not torch.equal(x, y) for x, y in zip(a['cuda'], b['cuda']))):
        raise ValueError('snapshot and endpoint RNG mismatch')
    if (snapshot['schema'] != 'paired-private-t2-boundary-v1' or snapshot['task'] != 1 or
            snapshot['round'] != pilot.cfg['rounds'] or snapshot['adapter_names'] != list(pilot.model.adapters)):
        raise ValueError('private momentum snapshot identity mismatch')
    if set(state['client_states']) != set(range(pilot.cfg['clients'])) or set(snapshot['momenta']) != set(state['client_states']):
        raise ValueError('source client state mismatch')
    for k, v in checkpoint['config'].items():
        if k != 'output' and pilot.cfg.get(k) != v:
            raise ValueError('source config mismatch: ' + k)
    if set(pilot.cfg) != set(checkpoint['config']):
        raise ValueError('source config keys mismatch')
    for k in state['client_states']:
        a, b = state['client_states'][k]['m'], snapshot['momenta'][k]
        if len(a) != 2 * len(pilot.model.adapters) or len(a) != len(b) or any(not torch.equal(x, y) for x, y in zip(a, b)):
            raise ValueError('saved client momentum differs')
    for _ in range(2):pilot.model.add_head(pilot.cfg['cpt'])
    pilot.model.load_state_dict(checkpoint['model'], strict=True)
    for index, (name, adapter) in enumerate(pilot.model.adapters.items()):
        adapter.merged = checkpoint['merged_flags'][name]
        if adapter.merged or not torch.equal(adapter.A.cpu(), snapshot['old_bases'][index]):
            raise ValueError('source T2 basis or merge state mismatch')
    pilot.s = copy.deepcopy(state)
    restore_rng(checkpoint['rng'])
    if not torch.equal(torch.get_rng_state(), checkpoint['rng']['torch']):
        raise ValueError('source CPU RNG restore mismatch')
    return {'source_clock': copy.deepcopy(checkpoint['clock']),
            'source_training_visits': sum(v['visited'] for v in state['training_records'])}
