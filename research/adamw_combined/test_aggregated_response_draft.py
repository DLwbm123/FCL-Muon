import unittest
import torch
from unittest.mock import patch
from research.adamw_combined.aggregated_response_draft import paired_aggregates


class Adapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('A', torch.eye(2))
        self.B = torch.nn.Parameter(torch.zeros(2, 2))

    def forward(self, x):
        return x + (x @ self.A.T) @ self.B.T


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = Adapter()

    def forward_features(self, x):
        return self.adapter(x).unsqueeze(1)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.adapters = {'0': self.backbone.adapter}
        self.rows = torch.nn.ParameterList([torch.nn.Parameter(torch.eye(2))])

    def b_parameters(self):
        return [self.backbone.adapter.B]

    def forward(self, x):
        z = torch.nn.functional.normalize(self.backbone.forward_features(x)[:, 0], dim=1)
        return z @ torch.nn.functional.normalize(self.rows[-1], dim=1).T


class AggregateTest(unittest.TestCase):
    def test_four_way_tiny_client_is_disjoint(self):
        from research.adamw_combined.aggregated_response_pilot_draft import split_four
        split = split_four(list(range(8)), 123, 32)
        self.assertEqual([len(x) for x in split], [2, 2, 2, 2])
        self.assertEqual(len(set(sum(split, []))), 8)
        self.assertEqual(split, split_four(list(range(8)), 123, 32))
        with self.assertRaises(ValueError):
            split_four([1, 2, 3], 123, 32)

    def test_paired_actual_head_aggregation_and_restore_on_failure(self):
        model = Toy()
        clients = [[(torch.tensor([[1., 0.], [0., 1.]]), torch.tensor([0, 1]))],
                   [(torch.tensor([[1., 1.], [-1., 1.]]), torch.tensor([1, 0]))]]
        basis = {'0': torch.eye(2)}
        saved = {k: v.clone() for k, v in model.state_dict().items()}
        def measure(m):
            return {'loss': float(torch.nn.functional.cross_entropy(m(clients[0][0][0]), clients[0][0][1]))}
        args = (model, [basis, basis], clients, [1, 1], [2, 5], 1, 0, 2, .01, 5, 42)
        out = paired_aggregates(*args, measure)
        self.assertEqual(out[0]['measurement'], out[1]['measurement'])
        self.assertTrue(torch.equal(out[0]['B'][0], out[1]['B'][0]))
        self.assertTrue(torch.equal(out[0]['head'], out[1]['head']))
        self.assertEqual(out[0]['train_visits'], 4)
        self.assertEqual(out[0]['gram_visits'], 4)
        self.assertTrue(all(torch.equal(saved[k], v) for k, v in model.state_dict().items()))
        def fail(_):
            raise RuntimeError('test failure')
        with self.assertRaises(RuntimeError):
            paired_aggregates(*args, fail)
        self.assertTrue(all(torch.equal(saved[k], v) for k, v in model.state_dict().items()))

    def test_task_boundary_lifecycle_on_toy_clients(self):
        from fcl.model import QVAdapter
        from research.adamw_combined.aggregated_response_pilot_draft import AggregatedResponsePilot
        class QBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.num_features = 4
                self.adapter = QVAdapter(torch.nn.Linear(4, 12), 2)
            def forward_features(self, x): return self.adapter(x)[..., :4].unsqueeze(1)
        class QModel(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.backbone = QBackbone()
                self.adapters = {'0': self.backbone.adapter}
                self.rows = torch.nn.ParameterList([torch.nn.Parameter(torch.eye(4))])
                self.backbone.adapter.start(torch.eye(4)[:2])
            def b_parameters(self): return [self.backbone.adapter.Bq, self.backbone.adapter.Bv]
            def forward(self, x):
                return self.backbone.forward_features(x)[:, 0] @ self.rows[-1].T
        torch.manual_seed(3)
        pilot = object.__new__(AggregatedResponsePilot); pilot.model = QModel(); pilot.device = torch.device('cpu')
        pilot.cfg = {'clients':2, 'batch':4, 'epochs':1, 'cpt':4, 'lr':.01, 'rounds':5,
                     'multistep':{'split_seed':3, 'client_cap':32, 'pool_width':3, 'trial_epochs':1,
                                  'trial_seed':12, 'response_mode':'selected'}}
        pilot.s = {'task':0, 'pending':{'0':torch.eye(4)}, 'history':type('H',(),{'raw':{'0':torch.eye(4)}})(),
                   'communication':{}}
        pilot.raw = [None]; xs = torch.randn(16,4); ys = torch.arange(16)%4
        pilot.ids = lambda _, __, k: list(range(8*k,8*k+8))
        pilot.loader = lambda _, ids, batch: [(xs[ids],ys[ids])]
        pilot.test_tf = lambda x:x; pilot.check_labels = lambda _:None; pilot.budget = lambda:None
        pilot.event = lambda *args, **kwargs:None
        def mock_pool(*_): return torch.eye(4)[:, :3], None, None
        def proposal(model, pools, histories, trials, mode):
            return {'0':model.adapters['0'].A.clone()}, {'0':{'accepted':False}}
        def fake_probe(p, ids, gradients):
            x = xs[sum(ids, [])]; y = ys[sum(ids, [])]
            with torch.no_grad(): loss = float(torch.nn.functional.cross_entropy(p.model(x), y))
            return {'all':loss, 'current':loss, 'coverage':1.}, None
        before = {k:v.clone() for k,v in pilot.model.state_dict().items()}
        with patch('research.adamw_combined.aggregated_response_pilot_draft.candidate_pool', mock_pool), \
             patch('research.adamw_combined.aggregated_response_pilot_draft.propose_from_trials', proposal), \
             patch('research.adamw_combined.aggregated_response_pilot_draft.probe', fake_probe):
            pilot._propose_multistep()
        self.assertEqual(pilot.s['selection_records'][0]['chosen'], 0)
        self.assertEqual(pilot.s['selection_records'][0]['trial_train_visits'], 16)
        self.assertTrue(all(torch.equal(before[k],v) for k,v in pilot.model.state_dict().items()))


if __name__ == '__main__':
    unittest.main()
