import json
from pathlib import Path
import pytest
import torch
from fcl.checkpoint import rng_state
from research.fed_spectra.loss_gate import GatedActualUpdatePilot,gate_accept
from research.fed_spectra.lifecycle import qualify
from test_actual_update import build as base_build
from test_fed_spectra import clean


def build(path,restore=False):
    p=base_build(path)
    p.cfg.update(seed=0,training_gate={'minimum_improvement':1e-6,'probe_per_client':32,'selection_seed_base':61029})
    result=GatedActualUpdatePilot(p.model,p.raw,p.manifest,p.loader,p.train_tf,p.test_tf,p.cfg,p.out,p.identity,p.clock)
    if restore:result.restore()
    return result


def test_gate_rule_and_config_rejects_scientific_drift():
    assert gate_accept(1.,.99,1e-6) and not gate_accept(1.,1.,1e-6) and not gate_accept(1.,1.01,1e-6)
    with pytest.raises(ArithmeticError):gate_accept(1.,float('nan'),1e-6)
    from scripts.actual_update_report import validate_comparison_configs
    root=Path(__file__).resolve().parents[1]
    a=json.loads((root/'configs/actual_update_control_seed0.json').read_text());b=json.loads((root/'configs/actual_gate_global_seed0.json').read_text())
    assert validate_comparison_configs(a,b)
    b['lr']*=2
    with pytest.raises(AssertionError):validate_comparison_configs(a,b)


@pytest.mark.parametrize('accept',[False,True])
def test_forced_gate_preserves_control_or_proposal_trajectory(tmp_path,monkeypatch,accept):
    reference=base_build(tmp_path/'ref',eta=.1 if accept else 0.);reference.run();rng=rng_state()
    monkeypatch.setattr('research.fed_spectra.loss_gate.gate_accept',lambda *args:accept)
    p=build(tmp_path/'gate');p.run()
    for key,v in reference.model.state_dict().items():assert torch.equal(v,p.model.state_dict()[key]),key
    for key in ['stages','training_records']:assert clean(reference.s[key])==clean(p.s[key]),key
    assert torch.equal(rng['torch'],rng_state()['torch'])
    assert all(v['accepted']==accept for v in p.s['training_gate_records'])
    for record in p.s['actual_update_rounds']:
        if not accept:assert all(v['actual_FP32_relative_correction']==0 for pair in record['layers'].values() for v in pair)


def test_natural_gate_resume_and_cost_qualification(tmp_path):
    full=build(tmp_path/'full');full.run()
    p=build(tmp_path/'resumed')
    while p.step():p=build(p.out,True)
    for key,v in full.model.state_dict().items():assert torch.equal(v,p.model.state_dict()[key]),key
    assert clean(full.s['training_gate_records'])==clean(p.s['training_gate_records'])
    for g in full.s['training_gate_records']:
        assert g['accepted']==(g['loss_difference'] < -1e-6)
        assert g['baseline']['backward_images']==g['candidate']['backward_images']==0
    q=qualify(full,3)
    assert len(q['training_gate_records'])==6 and q['extra_probe_forward_images']>0 and q['extra_probe_backward_images']==0
    assert q['extra_probe_forward_images']==q['gate_student_forward_images']+q['gate_teacher_forward_images']
    assert q['logical_communication_bytes']['gate_B_candidates_broadcast']>0
