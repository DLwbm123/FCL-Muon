import unittest
import copy
import torch
import tempfile
from unittest.mock import patch
from research.adamw_combined.bilevel_pilot_draft import BilevelPilot, GuardedBilevelPilot
from research.adamw_combined.implicit_head_draft import UnsupportedHeadDerivative
from research.adamw_combined.test_aggregated_response_draft import Toy


class PilotTest(unittest.TestCase):
    def test_guarded_confirmation_full_synthetic_response(self):
        class TwoHeadToy(Toy):
            def __init__(self):
                super().__init__()
                self.rows.append(torch.nn.Parameter(torch.tensor([[.8, .2], [.1, .9]])))
            def forward(self, x):
                z = torch.nn.functional.normalize(self.backbone.forward_features(x)[:, 0], dim=1)
                return z @ torch.nn.functional.normalize(torch.cat(list(self.rows)), dim=1).T
        class Raw:
            targets = [0, 1] + [2, 3, 2, 3, 2, 3] * 2
            def __len__(self): return len(self.targets)
            def __getitem__(self, i):
                return torch.tensor([1. + i*.03, .5 + (i%3)*.1]), self.targets[i]
        manifest = {'cpt': 2, 'labels': {'train': Raw.targets},
                    'partitions': {'train': [[[0, 1], []], [list(range(2, 8)), list(range(8, 14))]],
                                   'test': [[[0]], [[1]]]}}
        cfg = {'clients': 2, 'batch': 4, 'epochs': 1, 'cpt': 2, 'lr': .01, 'rounds': 5,
               'bilevel': {'split_seed': 42, 'client_cap': 4, 'trial_epochs': 1, 'step': .01, 'mode': 'optimized'}}
        def loader(data, ids, batch):
            samples = [data[i] for i in ids]
            return [(torch.stack([x for x, _ in samples]), torch.tensor([y for _, y in samples]))]
        with tempfile.TemporaryDirectory() as root:
            pilot = GuardedBilevelPilot(TwoHeadToy(), (Raw(), object()), manifest, loader,
                lambda x: x, lambda x: x, cfg, root, {}, {}, holdout_tasks=[1],
                holdout_seed=7, holdout_fraction=.2)
            pilot.s['task'] = 1
            pilot.s['history'] = type('H', (), {'raw': {'0': torch.zeros(2, 2)}})()
            pilot.budget = lambda: None
            pilot.check_labels = lambda y: None
            def identical_proposal(bases, histories, evaluate, step, **kwargs):
                return {k: v.clone() for k, v in bases.items()}, {'accepted': True, 'reason': 'synthetic'}
            with patch('research.adamw_combined.bilevel_pilot_draft.propose_step', identical_proposal):
                pilot._propose_multistep()
            record = pilot.s['selection_records'][-1]
            self.assertEqual(record['confirmation'], 'accepted')
            self.assertEqual(pilot.raw[0].holdout_reads, 2)
            self.assertEqual(len(record['calls']), 2)
            self.assertEqual(record['calls'][0], record['calls'][1])
            self.assertEqual(record['calls'][0]['evaluation_visits'], 2)
            self.assertTrue(record['calls'][0]['completed'])

    def test_guarded_instance_routes_training_indices_and_raw(self):
        class Raw:
            targets = [0, 0, 1, 1, 2, 2, 3, 3]
            def __len__(self): return len(self.targets)
            def __getitem__(self, i): return torch.tensor([float(i)]), self.targets[i]
        manifest = {'cpt': 2, 'labels': {'train': Raw.targets},
                    'partitions': {'train': [[[0, 1, 2, 3]], [[4, 5, 6, 7]]],
                                   'test': [[[0]], [[1]]]}}
        source = copy.deepcopy(manifest)
        raw = Raw()
        with tempfile.TemporaryDirectory() as root:
            pilot = GuardedBilevelPilot(Toy(), (raw, object()), manifest, None, None, None,
                                        {}, root, {}, {}, holdout_tasks=[1], holdout_seed=7,
                                        holdout_fraction=.25)
            pilot.budget = lambda: None
            held = pilot.reserved[1][0][0]
            self.assertEqual(manifest, source)
            self.assertNotIn(held, pilot.ids('train', 1, 0))
            self.assertEqual(len(pilot.ids('train', 0, 0)), 4)
            with self.assertRaises(PermissionError):
                pilot.raw[0][held]
            self.assertEqual(pilot.raw[0][pilot.ids('train', 1, 0)[0]][1],
                             raw.targets[pilot.ids('train', 1, 0)[0]])
            pilot.cfg['batch'] = 2
            pilot.loader = lambda data, ids, batch: [(torch.stack([data[i][0] for i in ids]),
                                                      torch.tensor([data[i][1] for i in ids]))]
            pilot.check_labels = lambda y: None
            pilot.budget = lambda: None
            pilot.test_tf = lambda x: x
            seen = []
            def measure(basis, groups):
                seen.append(len(groups[0][0][1]))
                return {'loss': basis['0'].sum(), 'current': basis['0'].sum()}
            baseline = {'0': torch.eye(2)}
            candidate = {'0': torch.eye(2) * .5}
            chosen, verdict = pilot.confirm_proposal(baseline, candidate, 1, measure)
            self.assertEqual(verdict, 'accepted')
            self.assertTrue(torch.equal(chosen['0'], candidate['0']))
            self.assertEqual(seen, [1, 1])
            self.assertEqual(pilot.raw[0].holdout_reads, 2)  # loader fetches image and label separately
            with self.assertRaises(PermissionError): pilot.raw[0][held]
            restarted = GuardedBilevelPilot(Toy(), (raw, object()), manifest, None, None, None,
                                             {}, root, {}, {}, holdout_tasks=[1], holdout_seed=7,
                                             holdout_fraction=.25)
            self.assertEqual(restarted.identity, pilot.identity)
            changed = GuardedBilevelPilot(Toy(), (raw, object()), manifest, None, None, None,
                                           {}, root, {}, {}, holdout_tasks=[1], holdout_seed=7,
                                           holdout_fraction=.5)
            self.assertNotEqual(changed.identity, pilot.identity)
            with self.assertRaises(FileExistsError):
                restarted.confirm_proposal(baseline, candidate, 1, measure)
            self.assertEqual(restarted.raw[0].holdout_reads, 0)

    def test_two_client_task_boundary(self):
        pilot = object.__new__(BilevelPilot)
        pilot.model = Toy().eval()
        pilot.device = torch.device('cpu')
        pilot.cfg = {'clients': 2, 'batch': 4, 'epochs': 1, 'cpt': 2, 'lr': .01, 'rounds': 5,
                     'bilevel': {'split_seed': 42, 'client_cap': 4, 'trial_epochs': 1, 'step': .01, 'mode': 'optimized'}}
        pilot.s = {'task': 0, 'history': type('H', (), {'raw': {'0': torch.zeros(2, 2)}})()}
        pilot.raw = [None]
        x = torch.tensor([[1., .3], [.2, 1.], [.7, 1.], [-.4, 1.],
                          [.8, .1], [.1, .8], [.9, .4], [-.3, .7]])
        y = torch.tensor([0, 1, 1, 0, 0, 1, 1, 0])
        pilot.ids = lambda _, __, k: list(range(4*k, 4*k+4))
        pilot.loader = lambda _, ids, batch: [(x[ids], y[ids])]
        pilot.test_tf = lambda value: value
        pilot.budget = lambda: None
        pilot.check_labels = lambda value: None
        control = copy.copy(pilot)
        control.model = copy.deepcopy(pilot.model)
        control.cfg = copy.deepcopy(pilot.cfg)
        control.cfg['bilevel']['mode'] = 'fixed_rrs'
        control.s = dict(pilot.s)
        before = {k: v.clone() for k, v in pilot.model.state_dict().items()}
        anchor = {'0': pilot.model.adapters['0'].A.clone()}
        rejected, valid = pilot.deployment_candidate(anchor, {'0': anchor['0'] * 2},
                                                       {'0': torch.zeros(2, 2)})
        self.assertFalse(valid)
        self.assertTrue(torch.equal(rejected['0'], anchor['0']))
        rng_before = torch.get_rng_state().clone()
        pilot._propose_multistep()
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        control._propose_multistep()
        record = pilot.s['selection_records'][-1]
        other = control.s['selection_records'][-1]
        self.assertEqual(record['calls'], other['calls'])
        self.assertEqual(record['outer_backward_attempts'], 1)
        self.assertEqual(other['outer_backward_attempts'], 1)
        self.assertEqual(record['proposal_accepted'], other['proposal_accepted'])
        self.assertEqual(other['accepted_layers'], 0)
        self.assertTrue(torch.equal(before['backbone.adapter.A'], control.model.adapters['0'].A))
        self.assertIn(record['reason'], {'response_improved', 'response_gate', 'history_budget', 'zero_or_nonfinite_direction'})
        self.assertEqual(record['calls'][0]['trial_visits'], 4)
        self.assertEqual(record['calls'][0]['evaluation_visits'], 4)
        for k, v in pilot.model.state_dict().items():
            if not k.endswith('.A'):
                self.assertTrue(torch.equal(before[k], v))
        def fail_after_random(*args, **kwargs):
            torch.rand(1)
            raise RuntimeError('synthetic solver failure')
        rng_before = torch.get_rng_state().clone()
        with patch('research.adamw_combined.bilevel_pilot_draft.propose_step', fail_after_random):
            with self.assertRaisesRegex(RuntimeError, 'synthetic solver failure'):
                pilot._propose_multistep()
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        baseline = pilot.model.adapters['0'].A.clone()
        with patch('research.adamw_combined.bilevel_pilot_draft.propose_step',
                   side_effect=UnsupportedHeadDerivative('synthetic hard case')):
            pilot._propose_multistep()
        self.assertEqual(pilot.s['selection_records'][-1]['reason'], 'unsupported_head_derivative')
        self.assertEqual(pilot.s['selection_records'][-1]['accepted_layers'], 0)
        self.assertTrue(torch.equal(baseline, pilot.model.adapters['0'].A))
