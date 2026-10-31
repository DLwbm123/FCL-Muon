import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F
from scripts.paired_boundary_worker import predictions


class Model(torch.nn.Module):
    def forward(self, x):
        return F.one_hot(x[:, 0].long(), 6).float()


class PairedEvaluationTest(unittest.TestCase):
    def test_seen_class_argmax_uses_one_model_output(self):
        calls = []
        def fixed_loader(raw, ids, batch):
            calls.append(len(ids))
            labels = torch.tensor([i // 500 for i in ids])
            return [(labels[i:i+batch, None].float(), labels[i:i+batch]) for i in range(0, len(labels), batch)]
        pilot = SimpleNamespace(model=Model(), raw=[None, object()], cfg={'batch': 64, 'cpt': 2},
            ids=lambda split, task: list(range(task * 1000, (task + 1) * 1000)),
            device=torch.device('cpu'), test_tf=lambda x: x, budget=lambda: None)
        with patch('scripts.paired_boundary_worker.loader', fixed_loader):
            labels, predictions_, metrics = predictions(pilot, 2)
        self.assertEqual(calls, [1000, 1000, 1000])
        self.assertEqual(metrics, [1., 1., 1.])
        np.testing.assert_array_equal(labels, predictions_)


if __name__ == '__main__':
    unittest.main()
