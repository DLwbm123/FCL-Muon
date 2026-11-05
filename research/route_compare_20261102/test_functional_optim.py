import torch
from fcl.math import ClientMuon
from research.route_compare_20261102.functional_optim import muon_step, adamw_step


def test_muon_matches_production_and_preserves_state_across_rounds():
    torch.manual_seed(4)
    p = torch.nn.Parameter(torch.randn(5, 3))
    state = [torch.zeros_like(p)]
    func, mom = [p.detach().clone()], [torch.zeros_like(p)]
    for _round in range(2):
        for _ in range(4):
            gradient = torch.randn_like(p)
            p.grad = gradient.clone()
            ClientMuon([p], state, [], [], False).step(.003)
            func, mom = muon_step(func, mom, [gradient], .003)
            torch.testing.assert_close(func[0], p, rtol=2e-6, atol=2e-7)
            torch.testing.assert_close(mom[0], state[0], rtol=2e-6, atol=2e-7)


def test_muon_is_equivariant_under_basis_rotation():
    torch.manual_seed(8)
    q = torch.linalg.qr(torch.randn(3, 3)).Q
    p, m, g = (torch.randn(5, 3) for _ in range(3))
    left, ml = muon_step([p], [m], [g], .01)
    right, mr = muon_step([p @ q], [m @ q], [g @ q], .01)
    torch.testing.assert_close(right[0], left[0] @ q, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(mr[0], ml[0] @ q, rtol=2e-5, atol=2e-6)


def test_adam_matches_torch_with_fresh_state_each_round():
    torch.manual_seed(7)
    p = torch.nn.Parameter(torch.randn(4, 3))
    functional = [p.detach().clone()]
    for _round in range(2):
        opt = torch.optim.AdamW([p], lr=.004, weight_decay=0.)
        state = None
        for _ in range(4):
            gradient = torch.randn_like(p)
            p.grad = gradient.clone()
            opt.step()
            functional, state = adamw_step(functional, state, [gradient], [.004])
            torch.testing.assert_close(functional[0], p, rtol=2e-6, atol=2e-7)
