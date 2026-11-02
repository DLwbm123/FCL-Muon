"""Unregistered derivation of a guarded pilot from a verified T1 prefix.

The caller must independently verify source files, original clock, dataset and
model provenance. This routine does not make a prior test set blind.
"""
import copy
from research.aggregation_utility.lifecycle import import_prefix


def import_guarded_prefix(pilot, checkpoint, source_manifest, source_clock):
    if checkpoint['clock'] != source_clock:
        raise ValueError('source clock mismatch')
    if checkpoint['config'].get('selection_mode') != 'prefix':
        raise ValueError('source is not a shared RRS prefix')
    identity = dict(pilot.identity)
    if not identity.pop('guarded_partition', None) or identity != checkpoint['identity']:
        raise ValueError('source identity mismatch')
    filtered = pilot.manifest
    source = source_manifest['partitions']['train']
    if (source[0] != filtered['partitions']['train'][0] or
            source_manifest['partitions']['test'] != filtered['partitions']['test'] or
            source_manifest['labels'] != filtered['labels'] or source_manifest['cpt'] != filtered['cpt']):
        raise ValueError('source T1 or labels/test changed')
    for task, groups in enumerate(source):
        for client, original in enumerate(groups):
            kept = filtered['partitions']['train'][task][client]
            held = pilot.reserved.get(task, [[] for _ in groups])[client]
            if len(set(kept + held)) != len(original) or set(kept + held) != set(original):
                raise ValueError('derived training split mismatch')
    allowed = {'selection_mode', 'output', 'server_correction', 'bilevel'}
    keys = set(checkpoint['config']) | set(pilot.cfg)
    if any(checkpoint['config'].get(k) != pilot.cfg.get(k) for k in keys - allowed):
        raise ValueError('source scientific configuration mismatch')
    fork = copy.deepcopy(checkpoint)
    fork['identity'] = copy.deepcopy(pilot.identity)
    fork['clock'] = copy.deepcopy(pilot.clock)
    fork['config'] = copy.deepcopy(pilot.cfg)
    # import_prefix requires a shared RRS prefix, regardless of the target mode.
    fork['config']['selection_mode'] = 'prefix'
    import_prefix(pilot, fork)
    pilot.s['prefix_provenance'].update(source_clock=copy.deepcopy(source_clock),
                                        same_clock=False, guarded_partition=True)
    pilot.save()
