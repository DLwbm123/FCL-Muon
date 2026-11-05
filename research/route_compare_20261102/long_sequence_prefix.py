"""Unregistered five-task continuation: validate and derive a shared T1 fork."""
import copy


def derive_prefix(checkpoint, target_config, identity, old_clock, new_clock):
    """Change only the task horizon, route flag and experiment clock in a copy."""
    source = checkpoint['config']
    state = checkpoint['state']
    if (checkpoint['schema'] != 'cosine-head-regmean-pilot-1' or
            checkpoint['clock'] != old_clock or checkpoint['identity'] != identity or
            source['tasks'] != 3 or source['selection_mode'] != 'prefix' or
            source['route']['mode'] != 'prefix' or source['route']['optimizer'] != 'adamw' or
            identity['optimizer'] != 'adamw' or
            (state['task'], state['round'], state['phase']) != (1, 0, 'task_start') or
            state['history'].committed != {0} or state['merges'] != [0] or checkpoint['head_count'] != 1 or
            len(state['training_records']) != 50 or sum(v['visited'] for v in state['training_records']) != 125000):
        raise ValueError('invalid source T1 prefix')
    if (target_config['tasks'] != 5 or target_config['route']['mode'] not in ('rrs', 'selected') or
            target_config['selection_mode'] != target_config['route']['mode']):
        raise ValueError('expected registered five-task AdamW arm')
    expected = copy.deepcopy(source)
    expected['tasks'] = 5
    expected['route']['mode'] = target_config['route']['mode']
    for key in ('output', 'selection_mode'):
        expected[key] = target_config[key]
    if expected != target_config:
        raise ValueError('configuration differs beyond the registered horizon and route')
    fork = copy.deepcopy(checkpoint)
    fork['config']['tasks'] = 5
    fork['config']['route']['mode'] = target_config['route']['mode']
    fork['clock'] = copy.deepcopy(new_clock)
    return fork
