import copy,json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from fcl.checkpoint import rng_state
from research.fed_spectra.alignment_diagnostic import measure
from research.fed_spectra.loss_gate import GatedActualUpdatePilot
from research.fed_spectra.lifecycle import qualify
from test_loss_gate import build as previous_build
from test_actual_update import build as control_build
from test_fed_spectra import clean


def build(path,restore=False):
    p=previous_build(path);p.cfg['matrix_tangent']={'reference':'original_eta_angle','gradient':'client_fp32_server_fp64'}
    p=GatedActualUpdatePilot(p.model,p.raw,p.manifest,p.loader,p.train_tf,p.test_tf,p.cfg,p.out,p.identity,p.clock)
    if restore:p.restore()
    return p


def test_client_full_aggregation_matches_weighted_gradient():
    torch.manual_seed(13)
    class Model(torch.nn.Linear):
        def b_parameters(self):return [self.weight]
    model=Model(2,4,bias=False);teacher=copy.deepcopy(model).requires_grad_(False)
    data=torch.tensor([[1.,0.],[0.,1.],[1.,1.],[.5,-1.]])
    labels=torch.tensor([2,3,2,3]);groups=[[0],[1,2,3]]
    p=SimpleNamespace(model=model,raw=[None,None],device=torch.device('cpu'),s={'task':1},
      cfg={'clients':2,'batch':2,'cpt':2},settings={'anchor_weight':1.,'anchor_temperature':2.},
      ids=lambda split,t,k:groups[k],get_teacher=lambda:teacher,test_tf=lambda x:x,budget=lambda:None,
      check_labels=lambda y:None,loader=lambda raw,ids,batch:[(data[ids],labels[ids])])
    u,_,vh=torch.linalg.svd(model.weight.detach().double(),full_matrices=False)
    r,g=measure(p,groups,True);state=rng_state()
    rr,m=measure(p,groups,True,full_client_gradients=True)
    assert r==rr and m[0].shape==model.weight.shape and m[0].dtype==torch.float64
    from torch.nn import functional as F
    expected=torch.zeros_like(m[0])
    for ids in groups:
        z=model(data[ids]);target=(teacher(data[ids])[:,:2]/2).softmax(-1)
        loss=F.cross_entropy(z[:,2:],labels[ids]-2)+F.kl_div((z[:,:2]/2).log_softmax(-1),target,reduction='batchmean')*4
        local=torch.autograd.grad(loss,model.weight)[0]
        expected.add_(local.double(),alpha=len(ids)/len(data))
    torch.testing.assert_close(m[0],expected,rtol=0,atol=0)
    assert torch.equal(state['torch'],rng_state()['torch']) and model.weight.grad is None
    with pytest.raises(ValueError):measure(p,groups,False,full_client_gradients=True)
    with pytest.raises(ValueError):measure(p,groups,True,[(u,vh)],True)


def test_rejected_exact_control(tmp_path,monkeypatch):
    ref=control_build(tmp_path/'ref',eta=0.);ref.run();rng=rng_state()
    monkeypatch.setattr('research.fed_spectra.loss_gate.gate_accept',lambda *a:False)
    p=build(tmp_path/'test');p.run()
    for k,v in ref.model.state_dict().items():assert torch.equal(v,p.model.state_dict()[k])
    assert clean(ref.s['stages'])==clean(p.s['stages']) and torch.equal(rng['torch'],rng_state()['torch'])


def test_reference_operator_exact_previous_gate(tmp_path,monkeypatch):
    ref=previous_build(tmp_path/'ref');ref.run();rng=rng_state()
    monkeypatch.setattr('research.fed_spectra.loss_gate.matrix_candidate',lambda o,b,v,g:(v.clone(),{'identity_reason':None}))
    p=build(tmp_path/'test');p.run()
    for k,v in ref.model.state_dict().items():assert torch.equal(v,p.model.state_dict()[k])
    assert clean(ref.s['stages'])==clean(p.s['stages']) and torch.equal(rng['torch'],rng_state()['torch'])


def test_natural_resume_cost_and_actual_deployment(tmp_path):
    full=build(tmp_path/'full');full.run();rng=rng_state();p=build(tmp_path/'resume')
    while p.step():p=build(p.out,True)
    for k,v in full.model.state_dict().items():assert torch.equal(v,p.model.state_dict()[k])
    assert clean(full.s['training_gate_records'])==clean(p.s['training_gate_records']) and torch.equal(rng['torch'],rng_state()['torch'])
    for g,a in zip(full.s['training_gate_records'],full.s['actual_update_rounds']):
        assert g['accepted']==(g['loss_difference']< -1e-6)
        assert all('mode_gradient' not in op for op in g['matrix_tangent']['operators'])
        for layer,parts in a['layers'].items():
            for i,v in enumerate(parts):
                if g['accepted']:assert v==g['candidate_round']['layers'][layer][i]
                else:assert v['actual_FP32_relative_correction']==0
        for op in g['matrix_tangent']['operators']:
            assert op['used_angle']<=op['reference_angle']+1e-12 and op['norm_relative_error']<=1e-10
    q=qualify(full,3)
    assert q['gate_backward_images']==q['extra_probe_backward_images']>0
    assert q['logical_communication_bytes']['matrix_full_gradient_upload']==sum(len(v['baseline']['clients']) for v in full.s['training_gate_records'])*sum(b.numel()*4 for b in full.model.b_parameters())
    assert 'tangent_svd_bases_broadcast' not in q['logical_communication_bytes']
    assert q['extra_probe_forward_images']==q['gate_student_forward_images']+q['gate_teacher_forward_images']


def test_config_and_mutual_exclusion(tmp_path):
    from scripts.actual_update_report import validate_comparison_configs
    root=Path(__file__).resolve().parents[1]
    for seed in range(3):
        a=json.loads((root/f'configs/actual_update_control_seed{seed}.json').read_text());b=json.loads((root/f'configs/actual_matrix_global_seed{seed}.json').read_text())
        assert validate_comparison_configs(a,b)
        b['matrix_tangent']['reference']='larger_angle'
        with pytest.raises(AssertionError):validate_comparison_configs(a,b)
    p=build(tmp_path/'mixed');p.cfg['direction_selection']={'type':'gradient_reflection','tie_tolerance':1e-12}
    with pytest.raises(AssertionError):GatedActualUpdatePilot(p.model,p.raw,p.manifest,p.loader,p.train_tf,p.test_tf,p.cfg,p.out,p.identity,p.clock)
