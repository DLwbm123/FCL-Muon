import unittest
import torch
from research.adamw_combined.multistep_subspace_draft import propose_subspace


class SubspaceProposalTest(unittest.TestCase):
    def test_weighted_real_endpoint_energy_and_history_fallback(self):
        p = torch.eye(3); base = p[:1].clone(); h = torch.diag(torch.tensor([2., 1., 3.]))
        endpoints = torch.zeros(2, 1, 1, 3)
        endpoints[0, 0, 0, 1] = 3
        endpoints[1, 0, 0, 0] = 1
        out, reason = propose_subspace(p, base, h, endpoints, [2, 1])
        self.assertEqual(reason, 'accepted')
        torch.testing.assert_close(out.abs(), p[1:2])
        out, reason = propose_subspace(p, base, torch.diag(torch.tensor([1., 2., 3.])), endpoints, [2, 1])
        self.assertEqual(reason, 'history_budget_exceeded')
        self.assertTrue(torch.equal(out, base))

    def test_cancelled_clients_still_have_energy_and_tie_falls_back(self):
        p = torch.eye(2); base = p[:1].clone(); h = torch.eye(2)
        endpoints = torch.tensor([[[[0., 1.]]], [[[0., -1.]]]])
        out, reason = propose_subspace(p, base, h, endpoints, [1, 1])
        self.assertEqual(reason, 'accepted')
        torch.testing.assert_close(out.abs(), p[1:2])
        zero = torch.zeros_like(endpoints)
        out, reason = propose_subspace(p, base, h, zero, [1, 1])
        self.assertEqual(reason, 'spectral_boundary_tie')
        self.assertTrue(torch.equal(out, base))
        with self.assertRaises(ValueError): propose_subspace(p, base, h, zero, [0, 0])


if __name__ == '__main__': unittest.main()
