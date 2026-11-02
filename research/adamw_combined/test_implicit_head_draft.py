import unittest
import torch
from research.adamw_combined.implicit_head_draft import sphere_implicit, certified_head_surrogate
from research.head_regmean.solver import sphere_solve, aggregate_heads


class ImplicitHeadTests(unittest.TestCase):
    def test_nondegenerate_matches_solver_and_finite_difference(self):
        h = torch.tensor([[2.0, .2, .1], [.2, 1.0, .05], [.1, .05, .5]], dtype=torch.float64, requires_grad=True)
        q = torch.tensor([[.5, .4, .3], [.1, -.3, .7]], dtype=torch.float64, requires_grad=True)
        expected = sphere_solve(h.detach(), q.detach())[0]
        torch.testing.assert_close(sphere_implicit(h, q), expected)
        self.assertTrue(torch.autograd.gradcheck(sphere_implicit, (h, q), eps=1e-5, atol=2e-5, rtol=2e-4))

    def test_hard_case_refuses_gradient(self):
        h = torch.eye(2, dtype=torch.float64)
        q = torch.zeros((1, 2), dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, 'hard case'):
            sphere_implicit(h, q)

    def test_certified_forward_and_surrogate_direction(self):
        heads = [torch.tensor([[.4, .2, .5]], requires_grad=True),
                 torch.tensor([[.2, -.3, .6]], requires_grad=True)]
        grams = [torch.tensor([[2., .1, .2], [.1, 1., .1], [.2, .1, 1.5]], dtype=torch.float64, requires_grad=True),
                 torch.tensor([[1., .05, .1], [.05, 1.2, .1], [.1, .1, .7]], dtype=torch.float64, requires_grad=True)]
        result, cert = certified_head_surrogate(heads, grams, [5, 3])
        actual = aggregate_heads([h.detach() for h in heads], [g.detach() for g in grams], [5, 3], True)[0]
        self.assertTrue(cert['deployed_fp32']['passed'])
        self.assertTrue(torch.equal(result, actual))
        direction = torch.tensor([[.2, -.1, .3]])
        vector = torch.tensor([[.3, .7, -.2]])
        grad = torch.autograd.grad((result * vector).sum(), heads[0])[0]
        eps = 1e-3
        def score(step):
            local = [heads[0].detach() + step * direction, heads[1].detach()]
            return float((aggregate_heads(local, [g.detach() for g in grams], [5, 3], True)[0] * vector).sum())
        finite = (score(eps) - score(-eps)) / (2 * eps)
        self.assertAlmostEqual(float((grad * direction).sum()), finite, delta=2e-4)
