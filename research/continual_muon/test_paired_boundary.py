import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from fcl.checkpoint import atomic_save
from research.continual_muon.paired_boundary import capture, install


class PairedBoundaryTest(unittest.TestCase):
    def pilot(self, task, phase, basis, values):
        adapter = SimpleNamespace(A=basis, merged=False)
        states = {k: {'m': [m.clone() for m in pair]} for k, pair in values.items()}
        return SimpleNamespace(s={'task': task, 'round': 5 if task == 1 else 0, 'phase': phase,
                                  'client_states': states},
                               model=SimpleNamespace(adapters={'0': adapter}),
                               cfg={'rounds': 5, 'clients': len(values)},
                               identity={'manifest': 'fixed'}, clock={'started': 1, 'deadline': 100})

    def test_private_roundtrip_and_matched_arms(self):
        old = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
        new = torch.tensor([[1., 0., 0.], [0., 0., 1.]])
        m = torch.tensor([[1., 2.], [0., 0.]])
        old_pilot = self.pilot(1, 'ready_evaluation', old, {0: [m, m], 1: [m, m]})
        before = torch.get_rng_state().clone()
        snapshot = capture(old_pilot)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'private.pt'
            atomic_save(snapshot, path)
            snapshot = torch.load(path, map_location='cpu', weights_only=False)
        zeros = {0: [torch.zeros_like(m), torch.zeros_like(m)], 1: [torch.zeros_like(m), torch.zeros_like(m)]}
        zero, projected, gated = [self.pilot(2, 'round_start', new, zeros) for _ in range(3)]
        install(zero, snapshot, 'zero')
        install(projected, snapshot, 'projected')
        projections = {k: projected.s['client_states'][k]['m'] for k in zeros}
        probes = {0: projections[0], 1: [-x for x in projections[1]]}
        receipts = install(gated, snapshot, 'guarded', probes)
        self.assertTrue(receipts[0]['accepted'])
        self.assertFalse(receipts[1]['accepted'])
        for a, b in zip(gated.s['client_states'][0]['m'], projections[0]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for a, b in zip(gated.s['client_states'][1]['m'], zero.s['client_states'][1]['m']):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_wrong_identity_and_invalid_probe_are_atomic(self):
        a = torch.eye(2)
        m = torch.ones(2, 2)
        snapshot = capture(self.pilot(1, 'ready_evaluation', a, {0: [m, m], 1: [m, m]}))
        zeros = {0: [torch.zeros_like(m), torch.zeros_like(m)], 1: [torch.zeros_like(m), torch.zeros_like(m)]}
        pilot = self.pilot(2, 'round_start', a, zeros)
        wrong = copy.deepcopy(snapshot)
        wrong['identity'] = {'manifest': 'different'}
        with self.assertRaises(ValueError):
            install(pilot, wrong, 'projected')
        probes = {0: [m, m], 1: [m, torch.full_like(m, float('nan'))]}
        with self.assertRaises(ValueError):
            install(pilot, snapshot, 'guarded', probes)
        self.assertTrue(all(not torch.count_nonzero(v) for c in pilot.s['client_states'].values() for v in c['m']))


if __name__ == '__main__':
    unittest.main()
