import copy,json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch.nn import functional as F
from fcl.checkpoint import rng_state
from research.fed_spectra.alignment_diagnostic import measure
from research.fed_spectra.loss_gate import GatedActualUpdatePilot,current_loss_safe
from research.fed_spectra.lifecycle import qualify
from test_matrix_gate import build as matrix_build
from test_actual_update import build as control_build
from test_fed_spectra import clean


def build(path,restore=False):
    p=matrix_build(path);p.cfg['all_seen_gate']={'current_loss_tolerance':1e-6}
    p=GatedActualUpdatePilot(p.model,p.raw,p.manifest,p.loader,p.train_tf,p.test_tf,p.cfg,p.out,p.identity,p.clock)
    if restore:p.restore()
    return p


def test_all_seen_client_gradient_oracle_and_current_loss():
    torch.manual_seed(27)
    class Model(torch.nn.Linear):
        def b_parameters(self):return [self.weight]
    model=Model(2,4,bias=False);teacher=copy.deepcopy(model).requires_grad_(False)
    with torch.no_grad():model.weight.add_(torch.tensor([[.1,.2],[.2,.3],[.4,.1],[.2,.4]]))
    data=torch.tensor([[1.,0.],[0.,1.],[1.,1.],[.5,-1.]])
    labels=torch.tensor([2,3,2,3]);groups=[[0],[1,2,3]]
    p=SimpleNamespace(model=model,raw=[None,None],device=torch.device('cpu'),s={'task':1},cfg={'clients':2,'batch':2,'cpt':2},
        settings={'anchor_weight':1.,'anchor_temperature':2.},ids=lambda split,t,k:groups[k],get_teacher=lambda:teacher,
        test_tf=lambda x:x,budget=lambda:None,check_labels=lambda y:None,loader=lambda raw,ids,batch:[(data[ids],labels[ids])])
    old,_=measure(p,groups);state=rng_state();row,grad=measure(p,groups,True,full_client_gradients=True,all_seen_objective=True)
    expected=torch.zeros_like(grad[0])
    for ids in groups:
        z=model(data[ids]);target=(teacher(data[ids])[:,:2]/2).softmax(-1)
        loss=F.cross_entropy(z,labels[ids])+F.kl_div((z[:,:2]/2).log_softmax(-1),target,reduction='batchmean')*4
        expected.add_(torch.autograd.grad(loss,model.weight)[0].double(),alpha=len(ids)/len(data))
    torch.testing.assert_close(grad[0],expected,rtol=0,atol=0)
    assert row['weighted']['current_loss']==old['weighted']['loss']
    assert row['weighted']['current_ce']==old['weighted']['ce'] and row['weighted']['ce']>row['weighted']['current_ce']
    assert row['student_forward_images']==row['backward_images']==4
    assert torch.equal(state['torch'],rng_state()['torch']) and model.weight.grad is None


@pytest.mark.parametrize('reject_by',['primary','current'])
def test_forced_rejection_exact_control(tmp_path,monkeypatch,reject_by):
    ref=control_build(tmp_path/'ref',eta=0.);ref.run();rng=rng_state()
    if reject_by=='primary':monkeypatch.setattr('research.fed_spectra.loss_gate.gate_accept',lambda *a:False)
    else:
        monkeypatch.setattr('research.fed_spectra.loss_gate.gate_accept',lambda *a:True)
        monkeypatch.setattr('research.fed_spectra.loss_gate.current_loss_safe',lambda *a:False)
    p=build(tmp_path/'test');p.run()
    assert all(torch.equal(v,p.model.state_dict()[k]) for k,v in ref.model.state_dict().items())
    assert clean(ref.s['stages'])==clean(p.s['stages']) and torch.equal(rng['torch'],rng_state()['torch'])
    assert all(not v['accepted'] for v in p.s['training_gate_records'])


def test_natural_resume_and_scalar_cost(tmp_path):
    full=build(tmp_path/'full');full.run();rng=rng_state();p=build(tmp_path/'resume')
    while p.step():p=build(p.out,True)
    assert all(torch.equal(v,p.model.state_dict()[k]) for k,v in full.model.state_dict().items())
    assert clean(full.s['training_gate_records'])==clean(p.s['training_gate_records']) and torch.equal(rng['torch'],rng_state()['torch'])
    for g in full.s['training_gate_records']:
        assert g['accepted']==(g['loss_difference']< -1e-6 and g['all_seen_objective']['current_loss_difference']<=1e-6)
    q=qualify(full,3)
    assert q['logical_communication_bytes']['all_seen_guard_scalar_upload']==sum(len(v['baseline']['clients']) for v in full.s['training_gate_records'])*2*8
    assert q['extra_probe_backward_images']==q['gate_backward_images']


def test_guard_and_config_drift(tmp_path):
    assert current_loss_safe(1.,1.0000005,1e-6) and not current_loss_safe(1.,1.000002,1e-6)
    with pytest.raises(ArithmeticError):current_loss_safe(1.,float('nan'),1e-6)
    from scripts.actual_update_report import validate_comparison_configs
    root=Path(__file__).resolve().parents[1]
    for seed in range(3):
        a=json.loads((root/f'configs/actual_update_control_seed{seed}.json').read_text());b=json.loads((root/f'configs/actual_allseen_global_seed{seed}.json').read_text())
        assert validate_comparison_configs(a,b)
        b['all_seen_gate']['current_loss_tolerance']=1e-3
        with pytest.raises(AssertionError):validate_comparison_configs(a,b)
    p=build(tmp_path/'invalid');p.cfg.pop('matrix_tangent')
    with pytest.raises(AssertionError):GatedActualUpdatePilot(p.model,p.raw,p.manifest,p.loader,p.train_tf,p.test_tf,p.cfg,p.out,p.identity,p.clock)
