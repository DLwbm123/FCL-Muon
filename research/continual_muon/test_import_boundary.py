import copy
import unittest
from types import SimpleNamespace
import torch
from fcl.checkpoint import rng_state
from fcl.math import History
from research.continual_muon.import_boundary import import_t1


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Bq = torch.nn.Parameter(torch.zeros(2, 2))
        self.Bv = torch.nn.Parameter(torch.zeros(2, 2))
        self.adapters = {'0': SimpleNamespace(Bq=self.Bq, Bv=self.Bv, merged=False)}
        self.rows = torch.nn.ParameterList()

    def add_head(self, cpt):
        self.rows.append(torch.nn.Parameter(torch.rand(cpt, 2)))


class ImportTest(unittest.TestCase):
    def setUp(self):
        model = Model()
        model.add_head(2)
        history = History()
        history.committed.add(0)
        self.cfg = {'selection_mode': 'prefix', 'output': 'source', 'cpt': 2}
        self.x = {'schema': 'cosine-head-regmean-pilot-1', 'config': self.cfg,
                  'identity': {'manifest': 'fixed'}, 'clock': {'started': 1, 'deadline': 2},
                  'state': {'task': 1, 'round': 0, 'phase': 'task_start', 'history': history,
                            'client_states': {}, 'merges': [0], 'training_records': [{'visited': 7}]},
                  'head_count': 1, 'model': model.state_dict(), 'merged_flags': {'0': True}, 'rng': rng_state()}

    def pilot(self):
        return SimpleNamespace(model=Model(), cfg={**self.cfg, 'output': 'new'},
                               clock={'started': 3, 'deadline': 4}, s={})

    def test_isolated_import_and_rng(self):
        pilot = self.pilot()
        info = import_t1(pilot, self.x, self.x['identity'])
        self.assertEqual(info['shared_training_visits'], 7)
        self.assertTrue(torch.equal(torch.get_rng_state(), self.x['rng']['torch']))
        pilot.s['merges'].append(1)
        self.assertEqual(self.x['state']['merges'], [0])
        self.assertTrue(pilot.model.adapters['0'].merged)

    def test_rejects_changed_identity_clock_and_weights(self):
        with self.assertRaises(ValueError):
            import_t1(self.pilot(), self.x, {'manifest': 'other'})
        pilot = self.pilot()
        pilot.clock = self.x['clock']
        with self.assertRaises(ValueError):
            import_t1(pilot, self.x, self.x['identity'])
        altered = copy.deepcopy(self.x)
        altered['model']['Bq'][0, 0] = 1
        with self.assertRaises(ValueError):
            import_t1(self.pilot(), altered, self.x['identity'])


if __name__ == '__main__':
    unittest.main()
