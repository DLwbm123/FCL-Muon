import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from fcl.checkpoint import rng_state
from research.fcca.math import directions,candidate,directional,historical_gram,historical_energy,propose,finite_acceptance
from research.fcca.pilot import FccaPilot,probe_split
from research.fcca.lifecycle import import_boundary,config_hash
from test_aggregation_utility import toy

SETTINGS=json.loads((Path(__file__).parents[1]/'configs/fcca_three_seed0.json').read_text())['fcca']


def build(path,enabled=True,restore=False,force_reject=False):
    p=toy(path,kind='federated');p.__class__=FccaPilot
    p.cfg['fcca']=dict(SETTINGS,enabled=enabled,probe_per_split=2)
    if force_reject:p.cfg['fcca']['minimum_relative_raw_gain']=100.
    if restore:p.restore()
    return p


def test_cross_client_terms_can_reverse_diagonal_prediction():
    # Each own-client derivative is +1 (bad), but cross-client effects are -2.
    ds=[torch.tensor([[[1.],[-2.]],[[-2.],[1.]]],dtype=torch.float64)]
    matrix=torch.stack([directional([g],ds) for g in [torch.tensor([[1.],[0.]]),torch.tensor([[0.],[1.]])]])
    p=torch.tensor([.5,.5],dtype=torch.float64)
    assert float((p@matrix)@p)==-.5
    assert float(p@matrix.diag())==1.


def test_affine_anchor_history_and_exact_noop():
    torch.manual_seed(77);ep=[[torch.randn(4,2)] for _ in range(3)];p=torch.tensor([.2,.3,.5],dtype=torch.float64)
    origin=[torch.randn(4,2)];base=[sum(float(v)*e[0] for v,e in zip(p,ep))]
    ds=directions(ep,base,origin,p);q=[torch.tensor([[2.,.3],[.3,1.]],dtype=torch.float64)]
    h=historical_gram(ds,q)
    assert torch.equal(candidate(base,ds,p,p)[0],base[0])
    for alpha in [p,torch.tensor([.3,.3,.4],dtype=torch.float64)]:
        actual=candidate(base,ds,p,alpha)
        assert abs(historical_energy(actual,origin,q)-float(alpha@h@alpha))<2e-5


def test_solver_feasible_improvement_and_safe_flat_fallback():
    p=np.array([.2,.3,.5]);v=np.array([1.,-2.,.5]);h=np.eye(3)
    a,r=propose(p,v,h,SETTINGS)
    assert r['success'] and v@(a-p)<0 and abs(a.sum()-1)<1e-9
    assert np.all(a>=.5*p-1e-9) and np.all(a<=1.5*p+1e-9)
    assert a@h@a<=r['history_limit']+1e-12
    a,r=propose(p,np.ones(3),h,SETTINGS)
    assert np.array_equal(a,p) and r['reason']=='no_client_direction_contrast'


def test_probe_split_disjoint_deterministic_and_rng_neutral():
    before=rng_state();a,b=probe_split(list(range(100)),2,5,SETTINGS)
    assert len(a)==len(b)==32 and not(set(a)&set(b))
    assert (a,b)==probe_split(list(range(100)),2,5,SETTINGS)
    assert torch.equal(before['torch'],rng_state()['torch'])
    assert probe_split([8],2,5,SETTINGS)==([8],[])


def test_finite_acceptance_requires_raw_and_within_on_both_splits():
    b={'raw_ce':1.,'within_ce':.4};good={'raw_ce':.9,'within_ce':.4}
    assert finite_acceptance(b,good,b,good,SETTINGS)[0]
    assert not finite_acceptance(b,good,b,{'raw_ce':1.01,'within_ce':.4},SETTINGS)[0]
    assert not finite_acceptance(b,{'raw_ce':.9,'within_ce':.5},b,good,SETTINGS)[0]
    assert not finite_acceptance(b,good,None,None,SETTINGS)[0]


def test_disabled_T1_and_all_rejected_preserve_original_trajectory(tmp_path):
    a=toy(tmp_path/'a',kind='federated');a.run();end=rng_state()
    for label,enabled,reject in [('disabled',False,False),('rejected',True,True)]:
        b=build(tmp_path/label,enabled=enabled,force_reject=reject);b.run()
        for k,v in a.model.state_dict().items():assert torch.equal(v,b.model.state_dict()[k]),(label,k)
        assert a.s['training_records']==b.s['training_records'] and a.s['stages']==b.s['stages']
        assert torch.equal(end['torch'],rng_state()['torch'])
        if enabled:
            assert len(b.s['fcca_records'])==4 and not any(r['accepted'] for r in b.s['fcca_records'])
    a=toy(tmp_path/'t1a',kind='federated')
    while a.s['task']==0:a.step()
    b=build(tmp_path/'t1b')
    while b.s['task']==0:b.step()
    assert not b.s.get('fcca_records')
    for k,v in a.model.state_dict().items():assert torch.equal(v,b.model.state_dict()[k])


def test_full_resume_every_transition_and_probe_head_isolation(tmp_path):
    a=build(tmp_path/'full');a.run();end=rng_state()
    b=build(tmp_path/'resumed')
    while b.step():b=build(b.out,restore=True)
    for k,v in a.model.state_dict().items():assert torch.equal(v,b.model.state_dict()[k]),k
    assert a.s['stages']==b.s['stages'] and a.s['training_records']==b.s['training_records']
    assert a.s['communication']==b.s['communication']
    assert torch.equal(end['torch'],rng_state()['torch'])
    assert all(r['fixed_actual_HeadRegMean'] and r['check_used_once'] for r in b.s['fcca_records'])
    for x,y in zip(a.s['fcca_records'],b.s['fcca_records']):
        xx={k:v for k,v in x.items() if k!='seconds'};yy={k:v for k,v in y.items() if k!='seconds'}
        assert xx==yy


def test_shared_T1_import_and_strict_other_config(tmp_path):
    src=toy(tmp_path/'src',kind='prefix')
    while src.s['task']==0:src.step()
    state=torch.load(src.out/'latest.pt',weights_only=False)
    expected={'completed_tasks':1,'config_sha256':config_hash(state['config']),'identity':state['identity'],
              'clock':state['clock'],'selection_mode':'prefix','toy':True}
    p=build(tmp_path/'new');import_boundary(p,state,expected)
    assert p.s['task']==1 and p.s['boundary_import']['reused_training_visits']==16
    p.run();assert len(p.s['fcca_records'])==4
    bad=build(tmp_path/'bad');bad.cfg['lr']*=2
    with pytest.raises(ValueError,match='Scientific'):import_boundary(bad,state,expected)
    with pytest.raises(ValueError,match='shared T1'):import_boundary(build(tmp_path/'future'),state,dict(expected,completed_tasks=2))


def test_accepted_branch_deploys_candidate_and_original_head(tmp_path):
    from research.head_regmean.solver import aggregate_heads
    p=build(tmp_path/'accept')
    while not (p.s['task']==1 and p.s['phase']=='local_done'):p.step()
    baseline=[v.clone() for v in p.s['agg_B']]
    fixed_head,_,_=aggregate_heads(p.s['heads'],p.s['grams'],p.s['sizes'],True,alpha=.5)
    # Controlled risks isolate the acceptance/deployment path. Real probe
    # gradients/state are exercised by lifecycle tests and GPU preflight.
    calls=[]
    def measure(part,ds=None):
        calls.append(part)
        r={'raw_ce':1. if len(calls)<=2 else .8,'within_ce':.4,'probe_images':2,
           'forward_calls':1,'backward_calls':int(ds is not None)}
        if ds is not None:r['global_raw_derivative']=[1.,-1.]
        return r
    p.measure=measure
    p.history_metrics=lambda:[torch.zeros(b.shape[1],b.shape[1],dtype=torch.float64) for b in p.model.b_parameters()]
    p.fuse();r=p.s['fcca_records'][-1]
    assert calls==[0,1,0,1] and r['accepted']
    assert r['deployed_weights']!=r['sample_weights']
    assert any(not torch.equal(b,a) for b,a in zip(p.model.b_parameters(),baseline))
    assert torch.equal(p.model.rows[-1].detach().cpu(),fixed_head)
    assert p.s['fcca_endpoints']==[] and p.s['phase']=='round_start'
