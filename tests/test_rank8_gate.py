import copy,json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset
from fcl.model import MuLoRAFL
from fcl.checkpoint import rng_state
from research.fed_spectra.alignment_diagnostic import measure
from research.fed_spectra.rank8_transport import encode_decode
from research.fed_spectra.loss_gate import GatedActualUpdatePilot
from research.fed_spectra.lifecycle import qualify
from research.fed_spectra.actual_update import ActualUpdatePilot
from test_all_seen_gate import build as original_build
from test_head_regmean_pilot import Backbone,Block
from test_fed_spectra import clean


def build(path,restore=False,control=False):
    p=original_build(path);torch.manual_seed(31)
    model=Backbone();model.num_features=16
    model.blocks[0].attn.qkv=nn.Linear(16,48);model.blocks[0].norm=nn.LayerNorm(16)
    x=torch.randn(24,3,16,generator=torch.Generator().manual_seed(8));y=torch.arange(24)//4
    p.raw=[TensorDataset(x,y),TensorDataset(x.clone(),y.clone())]
    p.cfg['rank']=8;p.cfg['rank8_transport']={'rank':8,'wire_dtype':'float32'}
    if control:
        for k in ['rank8_transport','all_seen_gate','matrix_tangent','training_gate']:p.cfg.pop(k)
        p.cfg['actual_update']['operator']['eta']=0.
    cls=ActualUpdatePilot if control else GatedActualUpdatePilot
    p=cls(MuLoRAFL(model,6,8,16),p.raw,p.manifest,p.loader,p.train_tf,p.test_tf,p.cfg,p.out,p.identity,p.clock)
    if restore:p.restore()
    return p


def test_independent_client_factor_oracle_and_rng():
    torch.manual_seed(37)
    class Model(nn.Linear):
        def b_parameters(self):return [self.weight]
    model=Model(16,12,bias=False);teacher=copy.deepcopy(model).requires_grad_(False)
    x=torch.randn(26,16);y=torch.arange(26)%6+6;groups=[list(range(9)),list(range(9,26))]
    p=SimpleNamespace(model=model,raw=[None,None],device=torch.device('cpu'),s={'task':1},cfg={'clients':2,'batch':32,'cpt':6},settings={'anchor_weight':1.,'anchor_temperature':2.},ids=lambda split,t,k:groups[k],get_teacher=lambda:teacher,test_tf=lambda x:x,budget=lambda:None,check_labels=lambda y:None,loader=lambda raw,ids,batch:[(x[ids],y[ids])])
    state=rng_state();r,g=measure(p,groups,True,full_client_gradients=True,all_seen_objective=True,rank8_transport=True)
    expected=torch.zeros_like(g[0])
    for ids in groups:
        z=model(x[ids]);target=(teacher(x[ids])[:,:6]/2).softmax(-1)
        loss=F.cross_entropy(z,y[ids])+4*F.kl_div((z[:,:6]/2).log_softmax(-1),target,reduction='batchmean')
        grad=torch.autograd.grad(loss,model.weight)[0];u,s,vh=torch.linalg.svd(grad.double(),full_matrices=False)
        expected.add_((u[:,:8]*s[:8]).float().double()@vh[:8].float().double(),alpha=len(ids)/len(x))
    assert torch.equal(g[0],expected) and torch.equal(state['torch'],rng_state()['torch']) and model.weight.grad is None
    assert len(r['rank8_transport']['records'])==2 and r['backward_images']==26
    with pytest.raises(ValueError):measure(p,groups,rank8_transport=True)


def test_resume_and_wire_cost(tmp_path):
    full=build(tmp_path/'full');full.run();state=rng_state();p=build(tmp_path/'resume')
    while p.step():p=build(p.out,True)
    assert all(torch.equal(v,p.model.state_dict()[k]) for k,v in full.model.state_dict().items())
    assert clean(full.s['training_gate_records'])==clean(p.s['training_gate_records']) and torch.equal(state['torch'],rng_state()['torch'])
    q=qualify(full,3);records=[r for g in q['training_gate_records'] for r in g['baseline']['rank8_transport']['records']]
    assert q['logical_communication_bytes']['rank8_gradient_factor_upload']==sum(r['wire_bytes'] for r in records)
    assert 'matrix_full_gradient_upload' not in q['logical_communication_bytes'] and q['rank8_client_SVDs']==len(records)


def test_reject_exact_control_and_config_drift(tmp_path,monkeypatch):
    ref=build(tmp_path/'ref',control=True);ref.run();state=rng_state()
    monkeypatch.setattr('research.fed_spectra.loss_gate.gate_accept',lambda *a:False)
    p=build(tmp_path/'gate');p.run()
    assert all(torch.equal(v,p.model.state_dict()[k]) for k,v in ref.model.state_dict().items())
    assert clean(p.s['stages'])==clean(ref.s['stages']) and torch.equal(state['torch'],rng_state()['torch'])
    from scripts.actual_update_report import validate_comparison_configs
    root=Path(__file__).resolve().parents[1]
    for seed in range(3):
        a=json.loads((root/f'configs/actual_update_control_seed{seed}.json').read_text());b=json.loads((root/f'configs/actual_rank8_global_seed{seed}.json').read_text())
        assert validate_comparison_configs(a,b)
        b['rank8_transport']['rank']=9
        with pytest.raises(AssertionError):validate_comparison_configs(a,b)
