import math
import pytest
import torch
from research.fed_spectra.spectral_tangent import tangent_candidate


def test_mode_descent_norm_angle_and_support():
    base=torch.diag(torch.tensor([3.,1.,0.]));o=torch.zeros_like(base)
    angle=.01;s=base.diag().double();h=torch.tensor([-1.,3.,0.],dtype=torch.float64);h/=h.norm()
    ref=torch.diag((s*math.cos(angle)+s.norm()*h*math.sin(angle)).float())
    g=torch.diag(torch.tensor([1.,-2.,9.]))
    out,r=tangent_candidate(o,base,ref,g)
    assert r['predicted_delta']<0 and r['used_angle']<=r['reference_angle']+1e-12
    assert r['norm_relative_error']<1e-10 and r['actual_FP32_norm_relative_error']<5e-4
    assert out[2,2]==0 and r['retained_rank']==2 and min(r['singular_coefficients_after'])>=0
    assert abs(float(out.norm()-base.norm()))<1e-6


def test_analytic_positivity_cap():
    base=torch.diag(torch.tensor([1.,.001]));o=torch.zeros_like(base)
    ref=torch.diag(torch.tensor([.8,.6000008]));g=torch.diag(torch.tensor([0.,1.]))
    out,r=tangent_candidate(o,base,ref,g)
    assert r['angle_capped'] and min(r['singular_coefficients_after'])>=0
    assert r['predicted_delta']<0 and out[1,1].abs()<1e-6


@pytest.mark.parametrize('repeated',[False,True])
def test_coordinate_equivariance_and_repeated_basis(repeated):
    gen=torch.Generator().manual_seed(41)
    u=torch.linalg.qr(torch.randn(6,3,generator=gen,dtype=torch.float64)).Q
    s=torch.tensor([2.,2.,1.] if repeated else [3.,2.,1.],dtype=torch.float64)
    base=u@torch.diag(s);o=torch.zeros_like(base);g=torch.randn(6,3,generator=gen,dtype=torch.float64)
    h=torch.tensor([-.2,.1,.3],dtype=torch.float64);h-=h.dot(s)/s.dot(s)*s;h/=h.norm()
    ref=u@torch.diag(s*math.cos(.01)+s.norm()*h*math.sin(.01))
    q=torch.linalg.qr(torch.randn(3,3,generator=gen,dtype=torch.float64)).Q
    a,r=tangent_candidate(o,base,ref,g);b,rr=tangent_candidate(o,base@q,ref@q,g@q)
    torch.testing.assert_close(b.double(),a.double()@q,rtol=2e-6,atol=2e-7)
    assert abs(r['used_angle']-rr['used_angle'])<1e-10


def test_identity_invalid_and_rng():
    b=torch.eye(3);o=torch.zeros_like(b);before=torch.get_rng_state().clone()
    out,r=tangent_candidate(o,b,b,b);assert torch.equal(out,b) and r['identity_reason']=='zero_reference_angle'
    out,r=tangent_candidate(o,o,o,b);assert torch.equal(out,o)
    ref=torch.diag(torch.tensor([1.1,1.,.9]));out,r=tangent_candidate(o,b,ref,b)
    assert torch.equal(out,b) and r['identity_reason']=='zero_spectral_tangent'
    assert torch.equal(before,torch.get_rng_state())
    with pytest.raises(ArithmeticError):tangent_candidate(o,b,ref,b*float('nan'))
    with pytest.raises(ValueError):tangent_candidate(o,b,ref,b[:2])
