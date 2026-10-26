"""Scientific invariants of the new evidence mechanism (CPU synthetic data)."""
import json
import unittest
from pathlib import Path

import torch

from research.boundary_memory.core import (
    compose_scores, evidence_scores, lda_classifier, merge_moments, moments,
    objective_gradient, pair_weights, project, solve_evidence,
)


class BoundaryMemoryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        torch.set_num_threads(2)
        self.cfg = json.loads((Path(__file__).parents[1] / 'configs/boundary_full_three_task_seed0.json').read_text())['solver']

    def test_unequal_client_scatter_includes_between_client_mean(self):
        x = torch.randn(91, 7, dtype=torch.float64)
        x[1:13] += 3
        x[13:] -= 2
        pooled = moments(x)
        merged = None
        for part in (x[:1], x[1:13], x[13:]):
            merged = merge_moments(merged, moments(part))
        self.assertEqual(merged['n'], 91)
        torch.testing.assert_close(merged['mean'], pooled['mean'], atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(merged['scatter'], pooled['scatter'], atol=1e-10, rtol=1e-12)
        naive = sum(moments(v)['scatter'] for v in (x[:1], x[1:13], x[13:]))
        self.assertGreater(float((naive - pooled['scatter']).norm()), 10)

    def test_analytic_gradient_against_autograd(self):
        means = torch.randn(6, 5, dtype=torch.float64)
        roots = torch.randn(6, 5, 5, dtype=torch.float64)
        cov = roots @ roots.transpose(1, 2)
        cfg = dict(self.cfg, cpt=2)
        w = pair_weights(6, 2, 'bidirectional', 'cpu')
        x = torch.randn(6, 6, dtype=torch.float64, requires_grad=True)
        value, analytic = objective_gradient(x, means, cov, w, cfg)
        numeric = torch.autograd.grad(value, x)[0]
        torch.testing.assert_close(analytic, numeric, atol=1e-10, rtol=1e-10)

    def test_only_old_to_new_terms_are_removed(self):
        full = pair_weights(30, 10, 'bidirectional', 'cpu')
        one = pair_weights(30, 10, 'no_old_to_new', 'cpu')
        self.assertEqual(int((full > 0).sum()), 600)
        self.assertEqual(int((one > 0).sum()), 400)
        torch.testing.assert_close(full[20:, :20], one[20:, :20])
        torch.testing.assert_close(full[:20, :20], one[:20, :20])
        self.assertEqual(int(torch.count_nonzero(one[:20, 20:])), 0)

    def test_compose_is_label_free_and_preserves_every_block(self):
        z = torch.randn(43, 30, dtype=torch.float64)
        e = torch.randn_like(z) * 13
        for value in (compose_scores(z, e), compose_scores(z)):
            torch.testing.assert_close(value.reshape(-1, 3, 10).argmax(2), z.reshape(-1, 3, 10).argmax(2))
        q = compose_scores(z)
        torch.testing.assert_close(q.reshape(-1, 3, 10).exp().sum(2), torch.ones(43, 3, dtype=torch.float64))
        shifted = z.reshape(-1, 3, 10) + torch.randn(43, 3, 1) * 10
        torch.testing.assert_close(compose_scores(shifted.reshape_as(z), e), compose_scores(z, e))

    def test_solver_learns_missing_task_direction(self):
        # The shared within-task scores (u,-u) cannot distinguish task v=+/-1.
        means = torch.tensor([[1., 1.], [-1., 1.], [1., -1.], [-1., -1.]], dtype=torch.float64)
        cov = torch.eye(2, dtype=torch.float64).repeat(4, 1, 1) * .001
        cfg = dict(self.cfg, cpt=2, margin=2., max_iterations=1500, stationarity_tolerance=1e-5)
        theta, report = solve_evidence(means, cov, cfg)
        self.assertLess(report['objective'], report['initial_objective'] * .01)
        self.assertTrue(report['converged'])
        logits = torch.stack([means[:, 0], -means[:, 0]] * 2, dim=1) * 5
        fused = compose_scores(logits, evidence_scores(means, theta), cpt=2)
        self.assertEqual(fused.argmax(1).tolist(), [0, 1, 2, 3])
        self.assertLessEqual(report['max_row_norm'], cfg['radius'] + 1e-8)
        lda = lda_classifier(means, cov, torch.ones(4, dtype=torch.float64), .1)
        self.assertEqual(evidence_scores(means, lda).argmax(1).tolist(), [0, 1, 2, 3])

    def test_one_task_no_boundary_and_ball_projection(self):
        means = torch.randn(10, 5, dtype=torch.float64)
        cov = torch.eye(5, dtype=torch.float64).repeat(10, 1, 1)
        theta, report = solve_evidence(means, cov, self.cfg)
        self.assertEqual(int(theta.count_nonzero()), 0)
        self.assertEqual(report['pair_count'], 0)
        projected = project(torch.ones(3, 6, dtype=torch.float64) * 200, 3)
        torch.testing.assert_close(projected[:, :-1].norm(dim=1), torch.ones(3, dtype=torch.float64) * 3)
        torch.testing.assert_close(projected[:, -1], torch.ones(3, dtype=torch.float64) * 200)


if __name__ == '__main__':
    unittest.main()
