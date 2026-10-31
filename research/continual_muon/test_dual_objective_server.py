import unittest
import torch
from research.continual_muon.dual_objective_server import project_cone, guarded_candidate


class DualServerTest(unittest.TestCase):
    def test_two_active_kkt_and_collinear_constraints(self):
        q = torch.tensor([1., 2., 3.], dtype=torch.float64)
        g1 = torch.tensor([1., 0., 0.], dtype=torch.float64)
        g2 = torch.tensor([0., 1., 0.], dtype=torch.float64)
        projected, info = project_cone(q, [g1, g2])
        self.assertTrue(info['solved'])
        torch.testing.assert_close(projected, torch.tensor([0., 0., 3.], dtype=torch.float64))
        torch.testing.assert_close(q-projected, g1+2*g2)
        projected, info = project_cone(q, [g1, 2*g1])
        self.assertTrue(info['solved'])
        torch.testing.assert_close(projected, torch.tensor([0., 2., 3.], dtype=torch.float64))

    def test_actual_deployment_constraints_and_exact_fallback(self):
        u = torch.tensor([[1., 0.], [0., 0.]], dtype=torch.float64)
        direction = torch.tensor([[0., -1.], [-1., 0.]], dtype=torch.float64)
        g_all = torch.tensor([[0., 1.], [0., 0.]], dtype=torch.float64)
        g_current = torch.tensor([[0., 0.], [1., 0.]], dtype=torch.float64)
        out, info = guarded_candidate([u], [direction], [g_all], [g_current], [torch.eye(2, dtype=torch.float64)])
        self.assertTrue(info['accepted'])
        self.assertLess(info['all_directional'], 0)
        self.assertLessEqual(info['current_directional'], 0)
        self.assertLess(abs(float(out[0].norm()/u.norm())-1), 1e-12)
        zero, info = guarded_candidate([torch.zeros_like(u)], [direction], [g_all], [g_current], [torch.eye(2)])
        self.assertEqual(info['reason'], 'zero_update')
        self.assertTrue(torch.equal(zero[0], torch.zeros_like(u)))
        with self.assertRaises(ValueError):
            guarded_candidate([u], [direction], [g_all], [g_current], [-torch.eye(2)])
        with self.assertRaises(ValueError):
            guarded_candidate([u], [direction], [g_all], [g_current], [torch.eye(2)], history_tolerance=float('nan'))
        out, info = guarded_candidate([u.float()], [direction.float()], [g_all.float()],
                                      [g_current.float()], [torch.eye(2)])
        self.assertEqual(out[0].dtype, torch.float32)
        if info['accepted']:
            self.assertLessEqual(info['deployed_relative_norm_error'], 1e-6)
            self.assertLess(info['all_directional'], -1e-12)
            self.assertLessEqual(info['current_directional'], 1e-12)
        else:
            self.assertEqual(info['reason'], 'deployed_budget')
            self.assertTrue(torch.equal(out[0], u.float()))


if __name__ == '__main__':
    unittest.main()
