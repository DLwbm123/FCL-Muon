import unittest
import torch
from research.continual_muon.server_candidate import candidate


class ServerCandidateTest(unittest.TestCase):
    def test_equal_momentum_or_no_update_falls_back_exactly(self):
        u = torch.tensor([[1., 2.], [0., 1.]])
        m = torch.tensor([[1., 0.], [0., 2.]])
        out, info = candidate(u, [m, m], [.5, .5], torch.eye(2))
        self.assertEqual(info['reason'], 'zero_tangent')
        self.assertTrue(torch.equal(out, u))
        out, info = candidate(u, [m, -m], [.5, .5], torch.eye(2))
        self.assertEqual(info['reason'], 'zero_tangent')
        self.assertTrue(torch.equal(out, u))
        z, info = candidate(torch.zeros_like(u), [m, -m], [.5, .5], torch.eye(2))
        self.assertEqual(info['reason'], 'zero_update_or_angle')
        self.assertTrue(torch.equal(z, torch.zeros_like(u)))

    def test_norm_history_budget_and_invalid_input(self):
        u = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float64)
        m = torch.tensor([[2., 0.], [0., 1.]])
        n = torch.tensor([[0., 1.], [1., 0.]])
        out, info = candidate(u, [m, n], [.5, .5], torch.eye(2))
        self.assertTrue(info['accepted'])
        self.assertLess(abs(float(out.norm() / u.norm()) - 1), 1e-6)
        self.assertLessEqual(info['proposal_history'], info['baseline_history'] + 1e-9)
        deployed32, receipt32 = candidate(u.float(), [m, n], [.5, .5], torch.eye(2))
        if receipt32['accepted']:
            self.assertLessEqual(receipt32['deployed_history'], receipt32['baseline_history'] + 1e-10 * max(receipt32['baseline_history'], 1.))
        else:
            self.assertTrue(torch.equal(deployed32, u.float()))
        rejected, info = candidate(u, [m, n], [.5, .5], torch.diag(torch.tensor([1., 9.])))
        self.assertEqual(info['reason'], 'history_budget')
        self.assertTrue(torch.equal(rejected, u))
        with self.assertRaises(ValueError):
            candidate(u, [m, n], [.5, .5], -torch.eye(2))
        with self.assertRaises(ValueError):
            candidate(u, [m, n], [.6, .6], torch.eye(2))


if __name__ == '__main__':
    unittest.main()
