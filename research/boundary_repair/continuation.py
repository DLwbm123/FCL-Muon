"""Strict same-clock B T3-to-T10 continuation, after the fixed performance gate."""
from .lifecycle import config_hash


def validate_continuation(p,source,gate):
    if not gate['passed'] or gate['B_centered_correct']<=gate['LoRM_centered_correct']:
        raise ValueError('Continuation gate failed')
    if gate['test_images']!=3000 or gate['LoRM_centered_correct']!=2724:
        raise ValueError('Unexpected gate reference')
    if source['clock']!=p.clock or gate['clock']!=p.clock:
        raise ValueError('Continuation must preserve original clock')
    if source['config']['tasks']!=3 or p.cfg['tasks']!=10 or source['state']['phase']!='complete':
        raise ValueError('Requires completed B T3 and ten-task target')
    if not source['config'].get('boundary_repair',{}).get('enabled'):
        raise ValueError('Source is not B')
    for key in set(p.cfg)|set(source['config']):
        if key not in ('tasks','output') and p.cfg.get(key)!=source['config'].get(key):
            raise ValueError('Continuation scientific config mismatch: '+key)
    if gate['source_config_sha256']!=config_hash(source['config']):
        raise ValueError('Continuation source config mismatch')
    state=source['state']
    if len(state['training_records'])!=150 or sum(v['visited'] for v in state['training_records'])!=375000:
        raise ValueError('Incomplete T3 training')
    if round(state['stages'][-1]['centered']['all_seen_accuracy']*3000)!=gate['B_centered_correct']:
        raise ValueError('Gate metric differs from checkpoint')
    if len(state['repair_records'])!=100 or not all(v['numerical_checks_passed'] for v in state['repair_records']):
        raise ValueError('Missing or invalid repair evidence')
