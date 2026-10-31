import unittest
import torch
from research.continual_muon.momentum_transport import candidate, select


class TransportTest(unittest.TestCase):
    def test_projection_and_guard(self):
        old = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
        new = torch.tensor([[1., 0., 0.], [0., 0., 1.]])
        m = torch.tensor([[1., 2.], [0., 0.]])
        candidates, retained = candidate([m, m], [old], [new])
        self.assertAlmostEqual(retained, .2)
        torch.testing.assert_close(candidates[0], torch.tensor([[1., 0.], [0., 0.]]))
        chosen, info = select(candidates, candidates, retained)
        self.assertTrue(info['accepted'])
        torch.testing.assert_close(chosen[0], candidates[0])
        rejected, info = select(candidates, [-x for x in candidates], retained)
        self.assertFalse(info['accepted'])
        self.assertTrue(all(torch.count_nonzero(x) == 0 for x in rejected))

    def test_zero_and_invalid_inputs(self):
        a = torch.eye(2)
        z = torch.zeros(2, 2)
        values, retained = candidate([z, z], [a], [a])
        chosen, info = select(values, [z, z], retained)
        self.assertIsNone(retained)
        self.assertIsNone(info['probe_cosine'])
        self.assertFalse(info['accepted'])
        self.assertTrue(all(torch.count_nonzero(x) == 0 for x in chosen))
        with self.assertRaises(ValueError):
            candidate([z, z], [a], [a * 2])
        with self.assertRaises(ValueError):
            select([z], [torch.full_like(z, float('nan'))], .5)


if __name__ == '__main__':
    unittest.main()
