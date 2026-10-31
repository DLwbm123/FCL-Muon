import unittest
from types import SimpleNamespace
import torch
from scripts.server_response_diagnostic import evaluate


class Model(torch.nn.Module):
    def forward(self, x):
        # Current label is always 2; odd records are correctly classified.
        z = x[:, 0]
        return torch.stack((1-z, torch.zeros_like(z), z), dim=1)


class ServerResponseEvaluationTest(unittest.TestCase):
    def test_weighting_predictions_and_rng(self):
        def loader(raw, ids, batch):
            x = torch.tensor([[i % 2] for i in ids], dtype=torch.float32)
            return [(x, torch.full((len(ids),), 2, dtype=torch.long))] if ids else []
        pilot = SimpleNamespace(model=Model(), manifest={'partitions': {'train': [[], [], [[0, 1], [2, 3]]] }},
            raw=[object(), None], cfg={'batch': 16, 'cpt': 1}, loader=loader, device=torch.device('cpu'),
            budget=lambda: None, check_labels=lambda y: None, test_tf=lambda x: x)
        before = torch.get_rng_state().clone()
        totals, predictions = evaluate(pilot, [([0], [1]), ([2], [3])], 1)
        self.assertEqual(totals['samples'], 2)
        self.assertEqual(totals['weight_coverage'], 1.)
        self.assertEqual(totals['accuracy'], 1.)
        self.assertEqual(predictions, [(2, 2), (2, 2)])
        self.assertTrue(torch.equal(before, torch.get_rng_state()))


if __name__ == '__main__':
    unittest.main()
