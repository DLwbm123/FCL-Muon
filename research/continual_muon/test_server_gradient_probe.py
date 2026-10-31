import unittest
from types import SimpleNamespace
import torch
from research.continual_muon.server_gradient_probe import directional


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.b = torch.nn.Parameter(torch.zeros(1, 1))
        self.register_buffer('counter', torch.tensor(0))

    def b_parameters(self):return [self.b]

    def forward(self, x):
        self.counter.add_(1)
        torch.rand(1)
        z = x @ self.b
        return torch.cat((torch.zeros_like(z), torch.zeros_like(z), z), dim=1)


class ServerGradientTest(unittest.TestCase):
    def test_direction_reversal_and_state_restoration(self):
        model = Model().train();model.b.grad = torch.full_like(model.b, 7.)
        def loader(raw, ids, batch):
            return [(torch.tensor([[1. if i % 2 == 0 else -1.] for i in ids]),
                     torch.full((len(ids),), 2, dtype=torch.long))] if ids else []
        pilot = SimpleNamespace(model=model, manifest={'partitions': {'train': [[], [], [[0, 1], [2, 3]]] }},
            raw=[object(), None], loader=loader, cfg={'batch': 16}, device=torch.device('cpu'),
            budget=lambda: None, check_labels=lambda y: None, test_tf=lambda x: x)
        splits = [([0], [1]), ([2], [3])]
        before = torch.get_rng_state().clone()
        selection = directional(pilot, splits, [torch.ones(1, 1)], 0)
        confirmation = directional(pilot, splits, [torch.ones(1, 1)], 1)
        self.assertAlmostEqual(selection['directional_derivative'], -2/3, places=6)
        self.assertAlmostEqual(confirmation['directional_derivative'], 2/3, places=6)
        self.assertEqual(selection['samples'], 2)
        self.assertEqual(confirmation['weight_coverage'], 1.)
        self.assertTrue(torch.equal(torch.get_rng_state(), before))
        self.assertTrue(model.training);self.assertEqual(int(model.counter), 0)
        torch.testing.assert_close(model.b.grad, torch.full_like(model.b, 7.))


if __name__ == '__main__':
    unittest.main()
