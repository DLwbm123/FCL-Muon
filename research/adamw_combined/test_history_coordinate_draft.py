import unittest
import torch
from research.adamw_combined.history_coordinate_draft import history_coordinates


class HistoryCoordinateTest(unittest.TestCase):
    def test_same_subspace_diagonalizes_history(self):
        a = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
        h = torch.tensor([[2., 1., 0.], [1., 3., 0.], [0., 0., 4.]])
        out, reason = history_coordinates(a, h)
        self.assertEqual(reason, 'accepted')
        torch.testing.assert_close(out.T @ out, a.T @ a, atol=1e-6, rtol=1e-6)
        projected = out @ h @ out.T
        self.assertLess(abs(float(projected[0, 1])), 1e-6)
        again, _ = history_coordinates(a, h)
        self.assertTrue(torch.equal(out, again))

    def test_tie_and_invalid_history_fallback(self):
        a = torch.eye(2)
        out, reason = history_coordinates(a, torch.eye(2))
        self.assertEqual(reason, 'eigenvalue_tie')
        self.assertTrue(torch.equal(a, out))
        out, reason = history_coordinates(a, torch.diag(torch.tensor([-1., 2.])))
        self.assertEqual(reason, 'history_not_psd')
        self.assertTrue(torch.equal(a, out))
        with self.assertRaises(ValueError): history_coordinates(a, torch.full((2, 2), float('nan')))


if __name__ == '__main__': unittest.main()
