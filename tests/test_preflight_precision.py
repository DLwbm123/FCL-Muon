import pytest
import torch
from research.fed_spectra.preflight_precision import gradient_error


def test_small_cancellation_is_bounded_but_wrong_gradient_rejected():
    reference=torch.tensor([1.,0.])
    assert gradient_error(torch.tensor([1.,5e-7]),reference)['relative_L2_error']<1e-4
    with pytest.raises(AssertionError):gradient_error(reference*1.001,reference)
    with pytest.raises(ArithmeticError):gradient_error(reference*float('nan'),reference)
    assert gradient_error(torch.zeros(2),torch.zeros(2))['relative_L2_error']==0
    with pytest.raises(AssertionError):gradient_error(torch.tensor([1e-8,0.]),torch.zeros(2))
