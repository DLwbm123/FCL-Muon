import random
import unittest
import numpy as np
import torch
from research.adamw_combined.multistep_probe_draft import paired_probes


class PairedProbeTest(unittest.TestCase):
    def test_paired_rng_and_complete_restore(self):
        model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.BatchNorm1d(2))
        model[0].weight.grad = torch.ones_like(model[0].weight)
        model[0].bias.requires_grad_(False)
        model[1].eval()
        saved = {k: v.clone() for k, v in model.state_dict().items()}
        torch.manual_seed(5); random.seed(5); np.random.seed(5)
        before = torch.get_rng_state().clone()
        def probe(_):
            optimizer = torch.optim.AdamW([model[0].weight], lr=.1)
            model.train(); optimizer.zero_grad()
            loss = model(torch.randn(4, 3)).square().mean()
            loss.backward(); optimizer.step()
            return (float(loss), random.random(), float(np.random.rand()))
        a, b = paired_probes(model, [0, 1], probe)
        self.assertEqual(a, b)
        self.assertTrue(torch.equal(torch.get_rng_state(), before))
        for k, v in saved.items(): self.assertTrue(torch.equal(v, model.state_dict()[k]))
        self.assertFalse(model[1].training)
        self.assertFalse(model[0].bias.requires_grad)
        self.assertTrue(torch.equal(model[0].weight.grad, torch.ones_like(model[0].weight)))

    def test_exception_restores(self):
        model = torch.nn.Linear(1, 1)
        before = model.weight.detach().clone()
        def fail(_):
            with torch.no_grad(): model.weight.add_(10)
            raise ValueError('probe failed')
        with self.assertRaises(ValueError): paired_probes(model, [1], fail)
        self.assertTrue(torch.equal(model.weight, before))

class CandidateDecisionTest(unittest.TestCase):
    def test_weighted_improvement_cannot_hide_client_regression(self):
        from research.adamw_combined.multistep_probe_draft import select_from_client_losses
        result = select_from_client_losses([99, 1, 0], [1., 1., 0.], [.9, 1.01, 0.], 1., .9)
        self.assertLess(result['weighted_loss_change'], 0)
        self.assertFalse(result['accepted'])
        result = select_from_client_losses([99, 1], [1., 1.], [.9, 1.], 1., .9)
        self.assertTrue(result['accepted'])
        self.assertFalse(select_from_client_losses([1], [1.], [.9], 1., 1.1)['accepted'])
        self.assertFalse(select_from_client_losses([1], [1.], [1.], 1., 1.)['accepted'])

    def test_invalid_measurements_are_not_accepted(self):
        from research.adamw_combined.multistep_probe_draft import select_from_client_losses
        for counts, baseline, candidate in [([0], [1], [1]), ([1], [1], [float('nan')]), ([1.5], [1], [1])]:
            with self.assertRaises(ValueError):
                select_from_client_losses(counts, baseline, candidate, 1., 1.)

class ClientSimulationTest(unittest.TestCase):
    def test_paired_multistep_adam_and_current_label_guard(self):
        from research.adamw_combined.multistep_probe_draft import simulate_client
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.b = torch.nn.Parameter(torch.zeros(2, 2))
                self.rows = torch.nn.ParameterList([torch.nn.Parameter(torch.eye(2))])
            def b_parameters(self): return [self.b]
            def forward(self, x): return (x + x @ self.b) @ self.rows[-1].T
        model = Toy()
        batches = [(torch.tensor([[1., 0.], [0., 1.]]), torch.tensor([0, 1]))]
        saved = {k: v.clone() for k, v in model.state_dict().items()}
        def trial(_): return simulate_client(model, batches, 2, 0, 2, .01, 5, 1)
        a, b = paired_probes(model, [0, 1], trial)
        self.assertEqual((a['train_visits'], a['steps']), (4, 2))
        self.assertEqual(a['after'], b['after'])
        self.assertTrue(torch.equal(a['B'][0], b['B'][0]))
        self.assertGreater(float(a['B'][0].norm()), 0.)
        for k, v in saved.items(): self.assertTrue(torch.equal(v, model.state_dict()[k]))
        wrong = [(batches[0][0], torch.tensor([0, 2]))]
        with self.assertRaises(ValueError): simulate_client(model, wrong, 2, 0, 2, .01, 5, 1)

class FederatedSimulationTest(unittest.TestCase):
    def test_clients_candidates_are_paired_and_model_restored(self):
        from research.adamw_combined.multistep_probe_draft import simulate_candidates
        class Adapter(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer('A', torch.eye(2))
                self.B = torch.nn.Parameter(torch.zeros(2, 2))
            def forward(self, x): return x + (x @ self.A.T) @ self.B.T
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.adapter = Adapter(); self.adapters = {'0': self.adapter}
                self.rows = torch.nn.ParameterList([torch.nn.Parameter(torch.eye(2))])
            def b_parameters(self): return [self.adapter.B]
            def forward(self, x): return self.adapter(x) @ self.rows[-1].T
        model = Toy()
        client_a = [(torch.tensor([[1., 0.], [0., 1.]]), torch.tensor([0, 1]))]
        client_b = [(torch.tensor([[2., 0.], [0., 2.]]), torch.tensor([0, 1]))]
        saved = {k: v.clone() for k, v in model.state_dict().items()}
        base = {'0': torch.eye(2)}
        got = simulate_candidates(model, [base, base], [client_a, client_b], [1, 1], 2, 0, 2, .01, 5, 42)
        self.assertEqual(len(got), 2)
        for first, second in zip(*got):
            self.assertEqual(first['after'], second['after'])
            self.assertEqual(first['train_visits'], 4)
            self.assertTrue(torch.equal(first['B'][0], second['B'][0]))
        for k, v in saved.items(): self.assertTrue(torch.equal(v, model.state_dict()[k]))
        with self.assertRaises(ValueError):
            simulate_candidates(model, [base], [client_a, []], [1, 1], 2, 0, 2, .01, 5, 42)


if __name__ == '__main__': unittest.main()
