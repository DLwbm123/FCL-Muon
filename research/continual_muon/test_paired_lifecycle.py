import shutil
import tempfile
import unittest
from pathlib import Path
import torch
from research.head_regmean.pilot import Pilot
from research.continual_muon.paired_boundary import capture, install


class Adapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('A', torch.tensor([[1., 0., 0.], [0., 1., 0.]]))
        self.Bq = torch.nn.Parameter(torch.zeros(2, 2))
        self.Bv = torch.nn.Parameter(torch.zeros(2, 2))
        self.merged = False


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = Adapter()
        self.adapters = {'0': self.adapter}
        self.rows = torch.nn.ParameterList()

    def add_head(self, cpt):
        self.rows.append(torch.nn.Parameter(torch.randn(cpt, 2)))


class PairedLifecycleTest(unittest.TestCase):
    def test_real_pilot_checkpoint_reloads_matched_rng_and_state(self):
        config = {'clients': 2, 'rounds': 5, 'tasks': 3, 'cpt': 2}
        clock = {'started': 1, 'deadline': 100}
        identity = {'source': 'fixed'}
        m = torch.tensor([[1., 2.], [0., 0.]])
        zero = torch.zeros_like(m)
        with tempfile.TemporaryDirectory() as d:
            shared = Path(d) / 'shared'
            p = Pilot(Model(), [], {}, None, None, None, config, shared, identity, clock)
            p.s.update(task=1, round=5, phase='ready_evaluation',
                       client_states={k: {'m': [m.clone(), m.clone()]} for k in range(2)})
            snapshot = capture(p)
            p.model.adapter.A.copy_(torch.tensor([[1., 0., 0.], [0., 0., 1.]]))
            for _ in range(3):
                p.model.add_head(2)
            p.s.update(task=2, round=0, phase='round_start',
                       client_states={k: {'m': [zero.clone(), zero.clone()]} for k in range(2)})
            p.save()
            random_draws, states = [], []
            for mode in ('zero', 'projected'):
                out = Path(d) / mode;out.mkdir()
                shutil.copyfile(shared / 'latest.pt', out / 'latest.pt')
                branch = Pilot(Model(), [], {}, None, None, None, config, out, identity, clock)
                branch.restore()
                before = torch.get_rng_state().clone()
                install(branch, snapshot, mode)
                self.assertTrue(torch.equal(before, torch.get_rng_state()))
                random_draws.append(torch.rand(5))
                states.append(branch.s['client_states'][0]['m'][0])
            torch.testing.assert_close(random_draws[0], random_draws[1], rtol=0, atol=0)
            self.assertFalse(torch.count_nonzero(states[0]))
            self.assertGreater(int(torch.count_nonzero(states[1])), 0)


if __name__ == '__main__':
    unittest.main()
