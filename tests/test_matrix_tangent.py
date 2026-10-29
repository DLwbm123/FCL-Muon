import math
import pytest
import torch
from research.fed_spectra.matrix_tangent import tangent_candidate, gradient_coverage


def inputs():
    b=torch.tensor([[3.,0.],[0.,1.],[0.,0.]])
    h=torch.tensor([[0.,1.],[0.,0.],[0.,0.]])
    ref=b*math.cos(.02)+b.norm()*h*math.sin(.02)
    return torch.zeros_like(b),b,ref,-h


def test_norm_angle_and_off_diagonal_descent():
    o,b,r,g=inputs();out,d=tangent_candidate(o,b,r,g)
    assert d['predicted_delta']<0 and out[0,1]>0
    assert d['norm_relative_error']<1e-10 and d['actual_FP32_norm_relative_error']<5e-4
    assert abs(d['used_angle']-.02)<1e-7
    torch.testing.assert_close(out,r)
    cov=gradient_coverage(o,b,g,g,out)
    assert cov['spectral_fraction_of_tangent_energy']<1e-12
    assert abs(cov['fit_confirmation_tangent_cosine']-1)<1e-12
    assert cov['confirmation_predicted_delta']<0


def test_equivariance_and_rng():
    o,b,r,g=inputs();state=torch.get_rng_state().clone()
    q=torch.tensor([[0.,-1.],[1.,0.]])
    a,_=tangent_candidate(o,b,r,g);c,_=tangent_candidate(o@q,b@q,r@q,g@q)
    torch.testing.assert_close(c,a@q)
    assert torch.equal(state,torch.get_rng_state())


def test_identity_and_invalid():
    o,b,r,g=inputs()
    for bb,rr,gg in [(o,o,g),(b,b,g),(b,r,b)]:
        a,d=tangent_candidate(o,bb,rr,gg)
        assert torch.equal(a,bb) and d['identity_reason']
    with pytest.raises(ArithmeticError):tangent_candidate(o,b,r,g*float('nan'))
    with pytest.raises(ValueError):tangent_candidate(o,b,r,g[:1])


def test_coverage_diagonal_and_opposite_confirmation():
    o,b,r,g=inputs();g=torch.tensor([[1.,0.],[0.,-3.],[0.,0.]])
    a,_=tangent_candidate(o,b,r,g);v=gradient_coverage(o,b,g,-g,a)
    assert abs(v['spectral_fraction_of_tangent_energy']-1)<1e-12
    assert abs(v['fit_confirmation_tangent_cosine']+1)<1e-12
    assert v['confirmation_predicted_delta']>0


def test_deployed_spectrum_includes_new_support():
    from research.fed_spectra.direction_diagnostic import deployed_record
    b=torch.diag(torch.tensor([1.,0.]));o=torch.zeros_like(b)
    n=torch.diag(torch.tensor([math.cos(.1),math.sin(.1)]))
    proposal={'layers':{'0':[{'singular_values_before':[1.,0.]}]}}
    record=deployed_record(proposal,[o],[b],[n],'matrix_tangent')['layers']['0'][0]
    assert record['retained_rank_before']==1 and record['retained_rank_after']==2
    assert len(record['energy_after'])==2 and abs(sum(record['energy_after'])-1)<1e-12
