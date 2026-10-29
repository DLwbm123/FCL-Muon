import pytest
import torch
from research.fed_spectra.direction_diagnostic import confirmation_probes,reflected_proposal,choose_direction
from research.fed_spectra.alignment_diagnostic import select_probes


def test_reflection_preserves_norm_angle_and_opposes_tangent():
    o=[torch.zeros(2,2)];b=[torch.tensor([[1.,0.],[0.,0.]])]
    p=[torch.tensor([[.8,.6],[0.,0.]])]
    m,r=reflected_proposal(o,b,p)
    assert torch.allclose(m[0],torch.tensor([[.8,-.6],[0.,0.]]))
    assert abs(float(m[0].norm()-p[0].norm()))<1e-6
    assert r[0]['correction_length_relative_error']<1e-6
    g=[torch.tensor([[0.,1.],[0.,0.]])]
    sign,d=choose_direction(g,b,p,m)
    assert sign=='negative' and d['negative']<d['positive']
    assert choose_direction([-g[0]],b,p,m)[0]=='positive'
    assert choose_direction([torch.zeros_like(g[0])],b,p,m)[0]=='positive'


def test_disjoint_reproducible_probes_preserve_rng():
    groups=[list(range(70)),list(range(70,90)),[]]
    selected=select_probes(groups,2);before=torch.get_rng_state().clone()
    confirm=confirmation_probes(groups,2,selected)
    assert confirm==confirmation_probes(groups,2,selected) and torch.equal(before,torch.get_rng_state())
    assert [len(x) for x in confirm]==[32,0,0]
    assert all(not set(a)&set(b) and set(b)<=set(ids) for a,b,ids in zip(selected,confirm,groups))


def test_zero_and_invalid_endpoints():
    z=[torch.zeros(2,2)];out,_=reflected_proposal(z,z,z);assert torch.equal(out[0],z[0])
    with pytest.raises(ValueError):reflected_proposal(z,z,[torch.ones(2,2)])
    with pytest.raises(ArithmeticError):reflected_proposal(z,z,[torch.full((2,2),float('nan'))])
