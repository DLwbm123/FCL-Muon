import copy
import unittest
import torch
from research.adamw_combined.differentiable_trial_draft import differentiable_trial
from research.adamw_combined.multistep_probe_draft import simulate_client
from research.adamw_combined.test_aggregated_response_draft import Toy


class TrialTest(unittest.TestCase):
    def test_matches_adamw_multistep_and_preserves_source(self):
        model = Toy().eval()
        direct = copy.deepcopy(model)
        basis = {'0': model.adapters['0'].A.clone().requires_grad_(True)}
        batches = [(torch.tensor([[1., .3], [.2, 1.]]), torch.tensor([0, 1])),
                   (torch.tensor([[.7, 1.], [-.4, 1.]]), torch.tensor([1, 0]))]
        before = {k: v.clone() for k, v in model.state_dict().items()}
        result = differentiable_trial(model, basis, batches, 1, 0, 2, .01, 5, 2, 1)
        reference = simulate_client(direct, batches, 1, 0, 2, .01, 5, 2, 1)
        for actual, expected in zip(result['B'] + [result['head']], reference['B'] + [reference['head']]):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
        self.assertEqual(result['train_visits'], 4)
        self.assertTrue(all(torch.equal(v, model.state_dict()[k]) for k, v in before.items()))
        direction = torch.autograd.grad(result['head'].sum(), basis['0'])[0]
        self.assertTrue(torch.isfinite(direction).all())

    def test_rejects_wrong_labels(self):
        model = Toy().eval()
        with self.assertRaisesRegex(ValueError, 'non-current'):
            differentiable_trial(model, {'0': model.adapters['0'].A},
                                 [(torch.ones(1, 2), torch.tensor([3]))], 1, 0, 2, .01, 5, 1, 1)
