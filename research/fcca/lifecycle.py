"""Strict shared-T1 import for the explicitly authorized FCCA pilot."""
import copy
import hashlib
import json
import torch
from fcl.checkpoint import restore_rng


def config_hash(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def import_boundary(p,source,expected,continuation=False):
    if source['schema']!='cosine-head-regmean-pilot-1':raise ValueError('Schema mismatch')
    if config_hash(source['config'])!=expected['config_sha256']:raise ValueError('Source config hash mismatch')
    if source['identity']!=expected['identity'] or source['identity']!=p.identity:raise ValueError('Identity mismatch')
    if source['clock']!=expected['clock']:raise ValueError('Source clock mismatch')
    t=expected['completed_tasks'];s=source['state'];mode=source['config']['selection_mode']
    if continuation:
        if source['clock']!=p.clock:raise ValueError('Continuation must preserve original clock')
        if not source['config'].get('fcca',{}).get('enabled'):raise ValueError('Source is not FCCA')
        if s['phase']!='complete' or source['config']['tasks']!=t:raise ValueError('Incomplete continuation source')
        if not expected.get('toy') and (t,p.cfg['tasks'])!=(3,10):raise ValueError('Requires T3 to T10 continuation')
        records=s['training_records']
        if len({(v['task'],v['round'],v['client']) for v in records})!=len(records):raise ValueError('Duplicate training records')
        if not expected.get('toy') and (len(records),sum(v['visited'] for v in records))!=(150,375000):
            raise ValueError('Incomplete T3 training')
        if len(s['certificates'])!=t*p.cfg['rounds'] or not all(v['deployed_fp32']['passed'] for v in s['certificates']):
            raise ValueError('Missing head certificates')
        rr=s.get('fcca_records',[])
        if len(rr)!=(t-1)*p.cfg['rounds'] or len({(v['task'],v['round']) for v in rr})!=len(rr) or s.get('fcca_endpoints'):
            raise ValueError('Incomplete FCCA aggregation records')
    elif t!=1:raise ValueError('FCCA must start from shared T1, never from future state')
    if mode!=expected['selection_mode'] or p.cfg['selection_mode']!='federated':raise ValueError('Selection mode mismatch')
    allowed={'output','tasks'} if continuation else {'output','tasks','fcca','selection_mode','inference_rule'}
    for key in set(source['config'])|set(p.cfg):
        if key not in allowed and source['config'].get(key)!=p.cfg.get(key):raise ValueError('Scientific config mismatch: '+key)
    if source['config']['tasks']!=3 and not expected.get('toy'):raise ValueError('Unexpected source horizon')
    if p.cfg['tasks']<=t:raise ValueError('Target has no remaining tasks')
    if (s['task'],s['round'])!=(t,0) or s['phase'] not in ('complete','task_start'):raise ValueError('Not a committed boundary')
    if s['merges']!=list(range(t)) or s['history'].committed!=set(range(t)) or source['head_count']!=t:raise ValueError('Incomplete lifecycle')
    if s['pending'] or s['client_states'] or s['heads'] or s['grams'] or s['agg_B']:raise ValueError('Uncommitted local state')
    if len(s['stages'])!=t or len(s['matrix'])!=t:raise ValueError('Missing historical evaluations')
    if set(source['rng'])!={'python','numpy','torch','cuda'}:raise ValueError('Incomplete RNG')
    if set(s['history'].raw)!=set(p.model.adapters):raise ValueError('Incomplete historical Gram')
    for _ in range(t):p.model.add_head(p.cfg['cpt'])
    p.model.load_state_dict(source['model'],strict=True)
    for name,a in p.model.adapters.items():
        a.merged=source['merged_flags'][name]
        if not a.merged or torch.count_nonzero(a.Bq) or torch.count_nonzero(a.Bv):raise ValueError('Not merged and zeroed')
        h=s['history'].raw[name]
        if h.shape!=(a.A.shape[1],)*2 or not torch.isfinite(h).all():raise ValueError('Invalid history')
    p.s=copy.deepcopy(s);p.s['phase']='task_start'
    p.s['boundary_import']={'source':expected,'new_clock':p.clock,'target_config_sha256':config_hash(p.cfg),
        'reused_client_rounds':len(s['training_records']),'reused_training_visits':sum(v['visited'] for v in s['training_records']),
        'allowed_changes':sorted(allowed),'prefix_schedule_independent_of_horizon':True}
    if continuation:
        p.s['boundary_import'].update(explicit_user_continuation=True,prior_T1_import=copy.deepcopy(s.get('boundary_import')))
    restore_rng(source['rng']);p.save()
    return p.s['boundary_import']
