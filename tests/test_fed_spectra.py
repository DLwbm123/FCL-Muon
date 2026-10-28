import copy
import json
from pathlib import Path
import pytest
import torch
from torch.nn import functional as F
from fcl.checkpoint import rng_state
from research.fed_spectra.math import (barycenter,bures_distance,candidate_pool,fisher_coefficients,
                                      gaussian_map,orthogonal_pool,rebase_adapter,signed_disagreement,
                                      transport_increments,update_history)
from research.fed_spectra.pilot import SpectraPilot
from test_aggregation_utility import toy

SETTINGS=json.loads((Path(__file__).parents[1]/'configs/spectra_full_seed0.json').read_text())['fed_spectra']


def build(path,method='fed_spectra_full',restore=False):
    p=toy(path,plain=True)
    p.cfg.update(method=method,protocol_version='test-spectra',fed_spectra=dict(SETTINGS,candidate_dim=4,candidate_current_rank=1,sketch_dim=2,probe_images=2))
    s=SpectraPilot(p.model,p.raw,p.manifest,p.loader,p.train_tf,p.test_tf,p.cfg,p.out,p.identity,p.clock)
    if restore:s.restore()
    return s


def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items() if k not in ('seconds','time')}
    if isinstance(value,list):return [clean(v) for v in value]
    return value


def test_bures_diagonal_solution_and_explicit_fallback():
    a=torch.diag(torch.tensor([.8,.2],dtype=torch.float64));b=torch.diag(torch.tensor([.3,.7],dtype=torch.float64))
    weights=torch.tensor([.25,.75],dtype=torch.float64)
    actual,receipt=barycenter([a,b],weights,ridge=1e-6)
    aa=(1-1e-6)*a+1e-6*torch.eye(2)/2;bb=(1-1e-6)*b+1e-6*torch.eye(2)/2
    expected=(.25*aa.diag().sqrt()+.75*bb.diag().sqrt()).square();expected/=expected.sum()
    assert not receipt['fallback']
    torch.testing.assert_close(actual,torch.diag(expected),atol=1e-8,rtol=1e-7)
    assert bures_distance(a,a)<1e-12
    _,fallback=barycenter([a,b],weights,iterations=1,tolerance=0.)
    assert fallback['fallback'] and fallback['kind']=='arithmetic'


def test_signed_sketch_detects_opposites_hidden_by_gram():
    g=torch.tensor([[1.,2.],[3.,1.]],dtype=torch.float64)
    assert torch.equal(g.T@g,(-g).T@(-g))
    d=signed_disagreement([g,-g],[.5,.5])
    torch.testing.assert_close(d,g.T@g)
    assert float(d.trace())>0
    assert torch.equal(signed_disagreement([g,g],[.5,.5]),torch.zeros_like(d))


def test_fisher_nonzero_at_zero_kl_and_matches_softmax_covariance():
    z=torch.tensor([[.2,-.3,.8]],requires_grad=True)
    p=z.detach().softmax(-1)
    kl=F.kl_div(z.log_softmax(-1),p,reduction='batchmean')
    grad=torch.autograd.grad(kl,z)[0]
    assert float(grad.norm())<1e-7
    coefficient=fisher_coefficients(z,72,1.)
    score=(coefficient*z).sum()
    assert float(torch.autograd.grad(score,z)[0].norm())>1e-4
    # Enumerate all Rademacher signs, avoiding a noisy Monte-Carlo assertion.
    signs=torch.tensor([[a,b,c] for a in [-1.,1.] for b in [-1.,1.] for c in [-1.,1.]])
    v=p.sqrt()*signs;a=v-p*v.sum(-1,keepdim=True)
    torch.testing.assert_close(a.T@a/8,torch.diag(p[0])-p.T@p,atol=1e-7,rtol=1e-6)


def test_transport_rotation_equivariance_zero_and_norm_bound():
    torch.manual_seed(21)
    ds=[torch.randn(7,3,dtype=torch.float64) for _ in range(3)];weights=[.2,.3,.5]
    x=torch.randn(3,3,dtype=torch.float64);current=x.T@x;current/=current.trace()
    h=torch.eye(3,dtype=torch.float64)*.2;d=torch.diag(torch.tensor([.1,.7,.4],dtype=torch.float64))
    result,receipt=transport_increments(ds,weights,current,h,d,SETTINGS)
    q=torch.linalg.qr(torch.randn(3,3,dtype=torch.float64)).Q
    rotated,_=transport_increments([v@q for v in ds],weights,q.T@current@q,q.T@h@q,q.T@d@q,SETTINGS)
    torch.testing.assert_close(rotated,result@q,atol=1e-8,rtol=1e-7)
    assert float(result.norm())<=2*float(sum(p*v for p,v in zip(weights,ds)).norm())+1e-10
    assert all(.5-1e-10<=v['scale_min']<=v['scale_max']<=2+1e-10 for v in receipt['clients'])
    zero,record=transport_increments([torch.zeros_like(ds[0])],[1.],current,h,d,SETTINGS)
    assert torch.count_nonzero(zero)==0 and record['zero_update']
    opposite,_=transport_increments([ds[0],-ds[0]],[.5,.5],current,h,d,SETTINGS)
    assert torch.count_nonzero(opposite)==0


def test_rebase_retains_offspace_update_and_preserves_logits(tmp_path):
    p=build(tmp_path);p.step();a=p.model.adapters['0'];torch.manual_seed(3)
    with torch.no_grad():a.Bq.normal_();a.Bv.normal_()
    inputs=p.raw[0].tensors[0][:4]
    before=p.model(inputs).detach()
    old_q=a.base.weight[:5].detach().double()+a.Bq.detach().double()@a.A.double()
    new=torch.linalg.qr(torch.randn(5,2,dtype=torch.float64)).Q
    report=rebase_adapter(a,new)
    after=p.model(inputs).detach()
    reconstructed=a.base.weight[:5].detach().double()+a.Bq.detach().double()@a.A.double()
    torch.testing.assert_close(old_q,reconstructed,atol=3e-7,rtol=1e-7)
    torch.testing.assert_close(before,after,atol=1e-5,rtol=1e-5)
    assert sum(v['residual_fro'] for v in report)>0


def test_candidate_contains_previous_space_and_history_uses_ambient_coordinates():
    torch.manual_seed(3);previous=torch.linalg.qr(torch.randn(7,2,dtype=torch.float64)).Q
    history=torch.eye(7,dtype=torch.float64)/7
    sketches=[torch.randn(7,2,dtype=torch.float64) for _ in range(2)]
    pool=candidate_pool(previous,history,sketches,[.4,.6],4,1,42)
    torch.testing.assert_close(pool.T@pool,torch.eye(4,dtype=torch.float64),atol=1e-9,rtol=0)
    torch.testing.assert_close(pool@pool.T@previous,previous,atol=1e-8,rtol=1e-8)
    small=torch.diag(torch.tensor([.1,.2,.3,.4],dtype=torch.float64))
    q=torch.linalg.qr(torch.randn(4,4,dtype=torch.float64)).Q
    a=update_history(history,pool,small,.9)
    b=update_history(history,pool@q,q.T@small@q,.9)
    torch.testing.assert_close(a,b,atol=1e-9,rtol=1e-9)


@pytest.mark.parametrize('method',['controlled_adam','fed_spectra_mvp','fed_spectra_full'])
def test_complete_and_resume_at_every_transition_are_identical(tmp_path,method):
    full=build(tmp_path/'full',method);full.run();rng=rng_state()
    resumed=build(tmp_path/'resumed',method)
    while resumed.step():resumed=build(resumed.out,method,restore=True)
    for key,value in full.model.state_dict().items():assert torch.equal(value,resumed.model.state_dict()[key]),key
    for key in ['stages','training_records','communication','spectra_rounds','spectra_geometry','spectra_commits']:
        assert clean(full.s[key])==clean(resumed.s[key]),key
    assert torch.equal(rng['torch'],rng_state()['torch'])
    assert full.s['merges']==[0,1,2] and full.s['history'].committed=={0,1,2}
    assert len(full.s['training_records'])==12 and sum(v['visited'] for v in full.s['training_records'])==48
    assert full.s['anchor_forward_images']==32
    assert all(v['deployed_fp32']['passed'] for v in full.s['certificates'])
    if method!='controlled_adam':
        assert len(full.s['spectra_geometry'])==6 and len(full.s['spectra_rounds'])==6
        assert len(full.s['spectra_commits'])==3 and len(full.s['spectra_probes'])==9
        assert any(v['old_fisher_nonzero_clients'] for r in full.s['spectra_geometry'][2:] for v in r['layers'].values())


def test_probe_restores_rng_flags_and_old_heads_remain_frozen(tmp_path):
    p=build(tmp_path)
    while p.s['task']==0:p.step()
    p.step();old=p.model.rows[0].detach().clone()
    rng=rng_state();flags=[v.requires_grad for v in p.model.parameters()]
    p.probe_gradients()
    assert torch.equal(rng['torch'],rng_state()['torch'])
    assert flags==[v.requires_grad for v in p.model.parameters()]
    with pytest.raises(ValueError,match='Old/future'):p.check_labels(torch.tensor([0]))
    while p.step():assert torch.equal(old,p.model.rows[0])
    assert not p.model.rows[0].requires_grad


def test_no_implicit_configuration_or_clock_change_on_resume(tmp_path):
    p=build(tmp_path);p.step();p.step()
    ckpt=torch.load(p.out/'latest.pt',weights_only=False);ckpt['clock']['started']=2
    torch.save(ckpt,p.out/'latest.pt')
    with pytest.raises(ValueError,match='clock'):build(tmp_path,restore=True)


def test_actual_checkpoint_qualification_preserves_state_and_can_continue(tmp_path):
    from research.fed_spectra.lifecycle import qualify
    p=build(tmp_path)
    while p.s['task']<2:p.step()
    before=rng_state();weights={k:v.clone() for k,v in p.model.state_dict().items()}
    q=qualify(p,2)
    assert q['qualified'] and q['reload_images']==16
    assert q['reload_raw_predictions_equal'] and q['reload_centered_predictions_equal']
    assert torch.equal(before['torch'],rng_state()['torch'])
    for key,value in weights.items():assert torch.equal(value,p.model.state_dict()[key])
    p.run();final=qualify(p,3)
    assert final['qualified'] and (p.out/'COMPLETE.json').exists()
