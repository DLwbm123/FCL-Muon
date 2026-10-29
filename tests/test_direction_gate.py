import json
from pathlib import Path
import torch
import pytest
from fcl.checkpoint import rng_state
from research.fed_spectra.loss_gate import GatedActualUpdatePilot
from research.fed_spectra.lifecycle import qualify
from research.fed_spectra.direction_diagnostic import reflected_record,reflected_proposal
from test_loss_gate import build as original_build
from test_actual_update import build as base_build
from test_fed_spectra import clean


def build(path,restore=False):
    p=original_build(path);p.cfg['direction_selection']={'type':'gradient_reflection','tie_tolerance':1e-12}
    p=GatedActualUpdatePilot(p.model,p.raw,p.manifest,p.loader,p.train_tf,p.test_tf,p.cfg,p.out,p.identity,p.clock)
    if restore:p.restore()
    return p


def test_all_rejected_exact_control_and_rng(tmp_path,monkeypatch):
    ref=base_build(tmp_path/'ref',eta=0.);ref.run();rng=rng_state()
    monkeypatch.setattr('research.fed_spectra.loss_gate.gate_accept',lambda *args:False)
    p=build(tmp_path/'directed');p.run()
    for k,v in ref.model.state_dict().items():assert torch.equal(v,p.model.state_dict()[k]),k
    assert clean(ref.s['stages'])==clean(p.s['stages']) and torch.equal(rng['torch'],rng_state()['torch'])
    assert all(v['actual_FP32_relative_correction']==0 for r in p.s['actual_update_rounds'] for pair in r['layers'].values() for v in pair)


def test_positive_orientation_exact_old_gate(tmp_path,monkeypatch):
    ref=original_build(tmp_path/'ref');ref.run();rng=rng_state()
    monkeypatch.setattr('research.fed_spectra.loss_gate.choose_direction',lambda *args:('positive',{'positive':0.,'negative':0.}))
    p=build(tmp_path/'positive');p.run()
    for k,v in ref.model.state_dict().items():assert torch.equal(v,p.model.state_dict()[k]),k
    assert clean(ref.s['stages'])==clean(p.s['stages']) and torch.equal(rng['torch'],rng_state()['torch'])


def test_negative_deployment_record_and_resume_every_transition(tmp_path,monkeypatch):
    monkeypatch.setattr('research.fed_spectra.loss_gate.choose_direction',lambda *args:('negative',{'positive':1.,'negative':-1.}))
    monkeypatch.setattr('research.fed_spectra.loss_gate.gate_accept',lambda *args:True)
    full=build(tmp_path/'full');full.run();p=build(tmp_path/'resumed')
    while p.step():p=build(p.out,True)
    for k,v in full.model.state_dict().items():assert torch.equal(v,p.model.state_dict()[k]),k
    assert clean(full.s['training_gate_records'])==clean(p.s['training_gate_records'])
    for gate,actual in zip(full.s['training_gate_records'],full.s['actual_update_rounds']):
        assert clean(gate['candidate_round'])==clean(actual)
        assert gate['direction']['selected']=='negative'
        assert all(v['direction']=='negative_reflection' and v['actual_FP32_norm_relative_error']<5e-4 for pair in actual['layers'].values() for v in pair)
    q=qualify(full,3)
    assert q['negative_direction_rounds']==6 and q['extra_probe_backward_images']==q['gate_backward_images']>0
    assert q['extra_probe_forward_images']==q['gate_student_forward_images']+q['gate_teacher_forward_images']
    assert q['logical_communication_bytes']['direction_derivative_scalar_upload']>0


def test_natural_direction_and_comparison_config(tmp_path):
    p=build(tmp_path/'natural');p.run()
    for g in p.s['training_gate_records']:
        dots=g['direction']['directional_derivatives']
        assert g['direction']['selected']==('negative' if dots['negative']<dots['positive']-1e-12 else 'positive')
        assert g['accepted']==(g['loss_difference']< -1e-6)
    from scripts.actual_update_report import validate_comparison_configs
    root=Path(__file__).resolve().parents[1]
    for seed in range(3):
        a=json.loads((root/f'configs/actual_update_control_seed{seed}.json').read_text());b=json.loads((root/f'configs/actual_direction_global_seed{seed}.json').read_text())
        assert validate_comparison_configs(a,b)
        b['direction_selection']['tie_tolerance']=1e-6
        with pytest.raises(AssertionError):validate_comparison_configs(a,b)
