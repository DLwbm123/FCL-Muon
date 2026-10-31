import unittest
from types import SimpleNamespace
import torch
from research.continual_muon.current_gradient_probe import probe


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.b = torch.nn.Parameter(torch.eye(2))
        self.register_buffer('counter', torch.tensor(0))

    def b_parameters(self):
        return [self.b]

    def forward(self, x):
        self.counter.add_(1)
        torch.rand(1)
        return torch.cat([torch.zeros(len(x), 4), x @ self.b], dim=1)


class ProbeTest(unittest.TestCase):
    def test_probe_restores_rng_modes_buffers_and_gradients(self):
        model = Model().train()
        model.b.grad = torch.full_like(model.b, 7.)
        x = torch.tensor([[1., 0.], [0., 1.]])
        y = torch.tensor([4, 5])
        def loader(raw, ids, batch):
            return [(x[ids], y[ids])]
        p = SimpleNamespace(model=model, s={'task': 2, 'round': 0, 'phase': 'round_start'},
                            ids=lambda split, task, client: [1, 0],
                            raw=[object(), None], loader=loader, cfg={'batch': 2, 'cpt': 2},
                            budget=lambda: None, check_labels=lambda labels: None,
                            device=torch.device('cpu'), test_tf=lambda value: value)
        before = torch.get_rng_state().clone()
        values, cost = probe(p, 0, count=2)
        self.assertEqual(cost['samples'], 2)
        self.assertGreater(float(values[0].norm()), 0.)
        self.assertTrue(torch.equal(torch.get_rng_state(), before))
        self.assertTrue(model.training)
        self.assertEqual(int(model.counter), 0)
        torch.testing.assert_close(model.b.grad, torch.full_like(model.b, 7.))


if __name__ == '__main__':
    unittest.main()
