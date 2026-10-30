import copy
import pytest
import torch
from research.fed_spectra.rank8_transport import encode_decode,conditions


def test_known_spectrum_and_actual_wire_quantization():
    g=torch.diag(torch.arange(32,0,-1,dtype=torch.float32))
    got,r=encode_decode(g)
    expected=g.double();expected[8:,8:]=0
    torch.testing.assert_close(got,expected,atol=1e-12,rtol=1e-12)
    assert r['wire_bytes']==8*64*4 and r['retained_energy_fraction']==float(g.diag()[:8].double().square().sum()/g.double().square().sum())
    h=torch.randn(40,32,generator=torch.Generator().manual_seed(7))
    u,s,vh=torch.linalg.svd(h.double(),full_matrices=False)
    actual,_=encode_decode(h)
    assert torch.equal(actual,(u[:,:8]*s[:8]).float().double()@vh[:8].float().double())


def test_zero_and_bad_inputs():
    got,r=encode_decode(torch.zeros(40,32));assert got.count_nonzero()==0 and r['retained_energy_fraction']==1
    with pytest.raises(ValueError):encode_decode(torch.zeros(3,2))
    with pytest.raises(ValueError):encode_decode(torch.zeros(40,32,dtype=torch.float64))
    with pytest.raises(ValueError):encode_decode(torch.full((40,32),float('nan')))


def test_conditions_do_not_hide_bad_seed_or_no_decision_gain():
    p={'paired_decisions':{'coverage':1.},'combined_delta_vs_baseline':-.01,'original_objective_delta_vs_baseline':-.001,'all_seen_accuracy_delta_vs_original':0.,'all_seen_accuracy_delta_vs_baseline':.001}
    rows=[{'seed':s,'engineering_passed':True,'wire_bytes':26,'full_bytes':100,'selection_gate_passed':True,'compressed_candidate':{'probes':{'confirmation':copy.deepcopy(p)}},'full_candidate':{'probes':{'confirmation':copy.deepcopy(p)}}} for s in range(3)]
    assert all(conditions(rows).values())
    rows[1]['compressed_candidate']['probes']['confirmation']['all_seen_accuracy_delta_vs_original']=-.001
    assert not conditions(rows)['validation_accuracy_not_worse_than_full']
    for r in rows:r['compressed_candidate']['probes']['confirmation']['all_seen_accuracy_delta_vs_baseline']=0.
    assert not conditions(rows)['mean_validation_accuracy_gain_positive']
    rows[0]['compressed_candidate']['probes']['confirmation']['combined_delta_vs_baseline']=-.0089
    assert not conditions(rows)['validation_retains_90percent_loss_gain']
