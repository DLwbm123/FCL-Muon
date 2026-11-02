import unittest
import torch
from unittest.mock import patch
from fcl.model import QVAdapter
from research.adamw_combined.test_aggregated_response_draft import Toy
from research.adamw_combined.differentiable_response_draft import differentiable_response
from research.adamw_combined.bilevel_step_draft import propose_step
from research.adamw_combined.aggregated_response_draft import paired_aggregates
from research.adamw_combined.implicit_head_draft import UnsupportedHeadDerivative


class ResponseTest(unittest.TestCase):
    def test_completed_work_survives_head_rejection(self):
        model = Toy().eval()
        basis = {'0': model.adapters['0'].A.clone().requires_grad_(True)}
        clients = [[(torch.tensor([[1., .3], [.2, 1.]]), torch.tensor([0, 1]))]] * 2
        counts = {'trial_visits': 0, 'gram_visits': 0, 'evaluation_visits': 0}
        def progress(kind, count):
            counts[kind] += count
        with patch('research.adamw_combined.differentiable_response_draft.certified_head_surrogate',
                   side_effect=UnsupportedHeadDerivative('synthetic hard case')):
            with self.assertRaises(UnsupportedHeadDerivative):
                differentiable_response(model, basis, clients, [2, 2], [1, 1], clients,
                                        1, 0, 2, .01, 5, 1, progress=progress)
        self.assertEqual(counts, {'trial_visits': 4, 'gram_visits': 4, 'evaluation_visits': 0})

    def test_two_clients_backprop_and_no_model_mutation(self):
        model = Toy().eval()
        basis = {'0': model.adapters['0'].A.clone().requires_grad_(True)}
        clients = [[(torch.tensor([[1., .3], [.2, 1.]]), torch.tensor([0, 1]))],
                   [(torch.tensor([[.7, 1.], [-.4, 1.]]), torch.tensor([1, 0]))]]
        initial = {k: v.clone() for k, v in model.state_dict().items()}
        output = differentiable_response(model, basis, clients, [2, 5], [1, 1], clients,
                                         1, 0, 2, .01, 5, 1)
        confirmation = differentiable_response(model, basis, clients, [2, 5], [1, 1], clients,
                                                1, 0, 2, .01, 5, 1, track_response=False)
        torch.testing.assert_close(confirmation['loss'], output['loss'].detach(), atol=0, rtol=0)
        torch.testing.assert_close(confirmation['current'], output['current'].detach(), atol=0, rtol=0)
        self.assertFalse(confirmation['loss'].requires_grad)
        self.assertEqual(confirmation['head_certificate'], output['head_certificate'])
        def measured(m):
            losses = [torch.nn.functional.cross_entropy(m(x), y) for batches in clients for x, y in batches]
            return {'loss': float((2*losses[0]+5*losses[1])/7)}
        reference = paired_aggregates(model, [{'0': basis['0'].detach()}], clients, [1, 1], [2, 5],
                                      1, 0, 2, .01, 5, 42, measured, schedule_epochs=1)[0]
        self.assertAlmostEqual(float(output['loss'].detach()), reference['measurement']['loss'], delta=1e-6)
        grad = torch.autograd.grad(output['loss'], basis['0'])[0]
        self.assertTrue(torch.isfinite(grad).all())
        direction = torch.tensor([[.3, -.2], [.4, .1]])
        eps = .01
        def evaluate(b):
            return differentiable_response(model, {'0': b}, clients, [2, 5], [1, 1], clients,
                                           1, 0, 2, .01, 5, 1)['loss'].detach()
        finite = (evaluate(basis['0'].detach()+eps*direction)-evaluate(basis['0'].detach()-eps*direction))/(2*eps)
        self.assertAlmostEqual(float((grad*direction).sum()), float(finite), delta=2e-5)
        self.assertEqual(output['trial_visits'], 4)
        self.assertEqual(output['gram_visits'], 4)
        self.assertEqual(output['evaluation_visits'], 4)
        self.assertTrue(output['head_certificate']['deployed_fp32']['passed'])
        self.assertTrue(all(torch.equal(v, model.state_dict()[k]) for k, v in initial.items()))
        def response(b):
            return differentiable_response(model, b, clients, [2, 5], [1, 1], clients,
                                           1, 0, 2, .01, 5, 1)
        proposed, rec = propose_step({'0': basis['0'].detach()}, {'0': torch.zeros(2, 2)}, response, .01)
        self.assertIn('accepted', rec)
        torch.testing.assert_close(proposed['0'] @ proposed['0'].T, torch.eye(2), atol=1e-5, rtol=1e-5)

    def test_qv_adapter_two_heads_current_task(self):
        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.adapter = QVAdapter(torch.nn.Linear(4, 12), 2)
                self.adapter.start(torch.eye(4)[:2])
            def forward_features(self, x):
                q, _, v = self.adapter(x).chunk(3, -1)
                return (q+v).unsqueeze(1)
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = Backbone()
                self.adapters = {'0': self.backbone.adapter}
                self.rows = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(2, 4), requires_grad=False),
                                                    torch.nn.Parameter(torch.randn(2, 4))])
            def b_parameters(self):
                a = self.adapters['0']; return [a.Bq, a.Bv]
            def forward(self, x):
                z = torch.nn.functional.normalize(self.backbone.forward_features(x)[:, 0], dim=1)
                return z @ torch.nn.functional.normalize(torch.cat(list(self.rows)), dim=1).T
        torch.manual_seed(11)
        model = Net().eval()
        clients = [[(torch.randn(4, 4), torch.tensor([2, 3, 2, 3]))] for _ in range(2)]
        fit = [[(torch.randn(3, 4), torch.tensor([2, 3, 2]))] for _ in range(2)]
        basis = {'0': model.adapters['0'].A.clone().requires_grad_(True)}
        before = {k: v.clone() for k, v in model.state_dict().items()}
        out = differentiable_response(model, basis, clients, [4, 7], [1, 1], fit, 1, 1, 2, .01, 5, 1)
        grad = torch.autograd.grad(out['loss'], basis['0'])[0]
        self.assertTrue(torch.isfinite(grad).all())
        self.assertEqual((out['trial_visits'], out['gram_visits'], out['evaluation_visits']), (8, 8, 6))
        self.assertTrue(all(torch.equal(v, model.state_dict()[k]) for k, v in before.items()))
