import copy
import unittest
from types import SimpleNamespace
import torch
from fcl.checkpoint import rng_state
from fcl.math import History
from research.continual_muon.import_t2_round import import_boundary
from research.continual_muon.paired_boundary import capture
from research.continual_muon.test_paired_lifecycle import Model


class ImportT2Test(unittest.TestCase):
    def setUp(self):
        cfg = {'clients': 2, 'rounds': 5, 'cpt': 2, 'output': 'source'}
        model = Model();model.add_head(2);model.add_head(2)
        history = History();history.committed = {0}
        m = torch.ones(2, 2)
        state = {'task': 1, 'round': 5, 'phase': 'ready_evaluation', 'history': history,
                 'pending': {'0': torch.eye(3)}, 'client_states': {k: {'m': [m.clone(), m.clone()]} for k in range(2)},
                 'training_records': [{'visited': 7}]}
        identity = {'manifest': 'fixed'};clock = {'started': 1, 'deadline': 2}
        source = SimpleNamespace(s=state, model=model, cfg=cfg, identity=identity, clock=clock)
        snapshot = capture(source)
        self.x = {'schema': 'cosine-head-regmean-pilot-1', 'head_count': 2, 'state': state,
                  'identity': identity, 'clock': clock, 'config': cfg, 'model': model.state_dict(),
                  'merged_flags': {'0': False}, 'rng': rng_state()}
        self.snapshot = snapshot
        self.cfg = {**cfg, 'output': 'isolated'}

    def target(self):
        return SimpleNamespace(model=Model(), cfg=self.cfg, clock={'started': 3, 'deadline': 4}, s={})

    def test_actual_state_and_rng_independent(self):
        pilot = self.target()
        info = import_boundary(pilot, self.x, self.snapshot, self.x['identity'])
        self.assertEqual(info['source_training_visits'], 7)
        self.assertTrue(torch.equal(torch.get_rng_state(), self.x['rng']['torch']))
        pilot.s['client_states'][0]['m'][0].zero_()
        self.assertTrue(torch.all(self.x['state']['client_states'][0]['m'][0] == 1))

    def test_mismatch_rejected_before_loading(self):
        snapshot = copy.deepcopy(self.snapshot)
        snapshot['momenta'][0][0][0, 0] = 2
        target = self.target()
        with self.assertRaises(ValueError):
            import_boundary(target, self.x, snapshot, self.x['identity'])
        self.assertEqual(len(target.model.rows), 0)


if __name__ == '__main__':
    unittest.main()
