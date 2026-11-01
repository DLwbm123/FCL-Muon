import unittest
import torch
from fcl.model import QVAdapter
from research.adamw_combined.wide_model_draft import wide_model


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = QVAdapter(torch.nn.Linear(4, 12), 2)
        self.adapter.A.copy_(torch.eye(4)[:2])
        self.adapter.merged = False
        self.adapters = {'0': self.adapter}
        self.rows = torch.nn.ParameterList([torch.nn.Parameter(torch.eye(4))])
    def b_parameters(self): return [self.adapter.Bq, self.adapter.Bv]
    def forward(self, x): return self.adapter(x)[..., :4] @ self.rows[-1].T


class WideCopyTest(unittest.TestCase):
    def test_wide_initial_output_and_independence(self):
        torch.manual_seed(10); model = Model(); x = torch.randn(3, 4)
        before = model(x).detach().clone()
        wider = wide_model(model, {'0': torch.eye(4)[:, :3]})
        self.assertIs(wider.adapters['0'], wider.adapter)
        self.assertEqual(wider.adapter.Bq.shape, (4, 3))
        torch.testing.assert_close(wider(x), before, rtol=0, atol=0)
        with torch.no_grad(): wider.adapter.Bq.add_(1)
        self.assertTrue(torch.equal(model(x), before))
        self.assertEqual(model.adapter.Bq.shape, (4, 2))

    def test_wide_trials_generate_correct_endpoint_shapes(self):
        from research.adamw_combined.multistep_probe_draft import simulate_candidates
        model = Model(); pool = torch.eye(4)[:, :3]; wider = wide_model(model, {'0': pool})
        clients = [[(torch.randn(3, 4), torch.tensor([0, 1, 2]))],
                   [(torch.randn(2, 4), torch.tensor([0, 1]))]]
        trials = simulate_candidates(wider, [{'0': pool.T}], clients, [1, 1], 2, 0, 4, .01, 5, 30)
        self.assertEqual(len(trials[0]), 2)
        endpoints = torch.stack([torch.stack(client['B']) for client in trials[0]])
        self.assertEqual(tuple(endpoints.shape), (2, 2, 4, 3))
        from research.adamw_combined.multistep_subspace_draft import propose_from_trials
        candidates, reasons = propose_from_trials(model, {'0': pool}, {'0': torch.eye(4)}, trials[0])
        self.assertEqual(tuple(candidates['0'].shape), (2, 4))
        self.assertIn(reasons['0'], ['accepted', 'spectral_boundary_tie', 'selected_eigenvalue_tie', 'history_budget_exceeded'])
        self.assertEqual(sum(client['train_visits'] for client in trials[0]), 10)
        self.assertTrue(all(torch.count_nonzero(b) == 0 for b in wider.b_parameters()))
        self.assertTrue(all(torch.count_nonzero(b) == 0 for b in model.b_parameters()))

    def test_rejects_incompatible_source_and_pool(self):
        model = Model()
        with torch.no_grad(): model.adapter.Bq[0, 0] = 1
        with self.assertRaises(ValueError): wide_model(model, {'0': torch.eye(4)[:, :3]})
        with torch.no_grad(): model.adapter.Bq.zero_()
        with self.assertRaises(ValueError): wide_model(model, {'0': torch.eye(4)[:, 1:]})


if __name__ == '__main__': unittest.main()
