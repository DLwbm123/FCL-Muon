import unittest
import torch
from research.adamw_combined.bilevel_step_draft import propose_step


class StepTest(unittest.TestCase):
    def test_proposal_and_exact_fallback(self):
        a = torch.tensor([[1., 0.]], dtype=torch.float64)
        h = torch.zeros(2, 2, dtype=torch.float64)
        def loss(bases):
            b = bases['x']
            return {'loss': -b[0, 1], 'current': b.sum()*0}
        proposal, record = propose_step({'x': a}, {'x': h}, loss, .1)
        self.assertTrue(record['accepted'])
        torch.testing.assert_close(proposal['x'] @ proposal['x'].T, torch.eye(1, dtype=a.dtype))
        blocked, record = propose_step({'x': a}, {'x': torch.diag(torch.tensor([0., 1.], dtype=a.dtype))}, loss, .1)
        self.assertEqual(record['reason'], 'history_budget')
        self.assertTrue(torch.equal(blocked['x'], a))

    def test_response_rejection(self):
        a = torch.tensor([[1., 0.]], dtype=torch.float64)
        def loss(bases):
            b = bases['x']
            return {'loss': -b[0, 1], 'current': b[0, 1]}
        result, record = propose_step({'x': a}, {'x': torch.zeros(2, 2, dtype=a.dtype)}, loss, .1)
        self.assertEqual(record['reason'], 'response_gate')
        self.assertTrue(torch.equal(result['x'], a))
