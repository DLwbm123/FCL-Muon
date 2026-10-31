import unittest
from types import SimpleNamespace
import torch
from research.continual_muon.dual_fit_gradients import gradients


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.b = torch.nn.Parameter(torch.zeros(1, 2));self.register_buffer('counter', torch.tensor(0))
    def b_parameters(self):return [self.b]
    def forward(self, x):
        self.counter.add_(1);torch.rand(1)
        return torch.cat((torch.zeros(len(x), 4), x @ self.b), dim=1)


class DualFitGradientTest(unittest.TestCase):
    def test_distinct_objectives_and_state_restoration(self):
        model = Model().train();model.b.grad = torch.full_like(model.b, 7.)
        def loader(raw, ids, batch):
            return [(torch.ones(len(ids), 1), torch.full((len(ids),), 4, dtype=torch.long))] if ids else []
        p = SimpleNamespace(model=model, manifest={'partitions': {'train': [[], [], [[0, 1], [2, 3]]] }},
            raw=[object(), None], loader=loader, cfg={'batch': 16, 'cpt': 2}, device=torch.device('cpu'),
            budget=lambda: None, check_labels=lambda y: None, test_tf=lambda x: x)
        rng = torch.get_rng_state().clone()
        ga, gc, cost = gradients(p, [([0], [1]), ([2], [3])], 0)
        torch.testing.assert_close(ga[0], torch.tensor([[-5/6, 1/6]], dtype=torch.float64), rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(gc[0], torch.tensor([[-.5, .5]], dtype=torch.float64))
        self.assertEqual(cost['student_forward_images'], 2);self.assertEqual(cost['student_backward_images'], 4)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()));self.assertTrue(model.training)
        self.assertEqual(int(model.counter), 0);torch.testing.assert_close(model.b.grad, torch.full_like(model.b, 7.))


if __name__ == '__main__':
    unittest.main()
