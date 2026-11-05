import pytest
import torch
from research.route_compare_20261102.frame import reference_frame, graph_basis, projector_distance, repair_candidate


def test_fixed_ordered_chart_and_gradient():
    torch.manual_seed(8)
    pool = torch.linalg.qr(torch.randn(9, 6, dtype=torch.float64)).Q
    ref = pool[:, :3].T.float()
    e, z0, n = reference_frame(pool, ref)
    x = torch.randn(3, 3, dtype=torch.float64, requires_grad=True) * .05
    actual = graph_basis(e, z0, n, x)
    torch.testing.assert_close(actual @ actual.T, torch.eye(3, dtype=torch.float64), atol=1e-12, rtol=0)
    base = graph_basis(e, z0, n, torch.zeros_like(x))
    torch.testing.assert_close(base, ref.double(), atol=1e-7, rtol=0)
    assert float(projector_distance(base, ref)) < 2e-7
    assert torch.autograd.gradcheck(lambda q: graph_basis(e, z0, n, q), (x,))


def test_spectral_repair_respects_fixed_deployed_budget():
    e = torch.eye(4, dtype=torch.float64)
    ref = e[:, :2].T.float()
    e, z0, n = reference_frame(e, ref)
    h = torch.diag(torch.tensor([2., 2., 1., 5.], dtype=torch.float64))
    x = .2 * torch.eye(2, dtype=torch.float64)
    out, record = repair_candidate(e, z0, n, x, h, ref, [0., 1., 10.], 2., 1e-5)
    assert out is not None and record['accepted']
    candidate = graph_basis(e, z0, n, out).float().double()
    assert float(torch.trace(candidate @ h @ candidate.T)) <= record['budget']
    assert record['attempts'][0]['excess'] > 0
    assert record['budget_slack'] == 1.
    # If the reference globally minimizes history, finite spectral repair may
    # be unable to find a distinct strictly feasible point; keep the reference.
    min_h = torch.diag(torch.tensor([1., 1., 9., 9.], dtype=torch.float64))
    none, minimum = repair_candidate(e, z0, n, x, min_h, ref, [0., 1., 10.], 2., 1e-5)
    assert none is None and not minimum['accepted'] and minimum['budget_slack'] == 0.
    with pytest.raises(ValueError):
        repair_candidate(e, z0, n, x, h, ref, [], 1., 1e-5)
