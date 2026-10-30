import unittest
import numpy as np
from research.continual_muon.real_boundary_geometry import measure, last_momentum_proxy


class BoundaryGeometryTest(unittest.TestCase):
    def test_projection_and_zero_states(self):
        a = np.array([[1., 0., 0.], [0., 1., 0.]])
        b = np.array([[1., 0., 0.], [0., 0., 1.]])
        m = np.array([[1., 2.], [0., 0.]])
        result = measure(m, a, b, lambda x: x)
        self.assertAlmostEqual(result['retained_energy'], .2)
        self.assertAlmostEqual(result['relative_residual'], np.sqrt(.8))
        self.assertLess(result['pythagorean_relative_error'], 1e-12)
        self.assertLess(result['residual_orthogonality_error'], 1e-12)
        zero = measure(np.zeros_like(m), a, b, lambda x: x)
        self.assertIsNone(zero['retained_energy'])
        self.assertIsNone(zero['ssb_weight_cosine'])

    def test_rejects_invalid_inputs(self):
        a = np.eye(2)
        with self.assertRaises(ValueError):
            measure(np.eye(2), a, a * 2, lambda x: x)
        with self.assertRaises(ValueError):
            measure(np.full((2, 2), np.nan), a, a, lambda x: x)
        with self.assertRaises(ValueError):
            measure(np.eye(2), a, a, lambda x: x * np.nan)

    def test_actual_ssb_and_exact_cancellation(self):
        import torch
        from fcl.math import mulora_ssb
        generator = torch.Generator().manual_seed(17)
        m = torch.randn(12, 4, generator=generator).numpy()
        a = torch.linalg.qr(torch.randn(9, 4, generator=generator)).Q.T.numpy()
        b = torch.linalg.qr(torch.randn(9, 4, generator=generator)).Q.T.numpy()
        ssb = lambda x: mulora_ssb(torch.from_numpy(x)).numpy()
        r = measure(m, a, b, ssb)
        old = mulora_ssb(torch.from_numpy(m)) @ torch.from_numpy(a)
        m_new = torch.from_numpy(m.astype(np.float64) @ a.astype(np.float64) @ b.astype(np.float64).T).float()
        new = mulora_ssb(m_new) @ torch.from_numpy(b)
        cosine = float((old * new).sum() / (old.norm() * new.norm()))
        self.assertAlmostEqual(r['ssb_weight_cosine'], cosine, places=5)
        result = last_momentum_proxy([m, -m], [.5, .5], ssb)
        self.assertEqual(result['difference_norm'], 0.)
        self.assertIsNone(result['relative_difference'])


if __name__ == '__main__':
    unittest.main()
