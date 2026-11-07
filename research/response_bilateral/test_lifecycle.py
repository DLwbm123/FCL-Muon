"""Actual trainer and response integration on small Q/V models."""
import copy
import tempfile
import unittest
from pathlib import Path
import torch
from torch.nn import functional as F
from fcl.model import QVAdapter
from fcl.checkpoint import rng_state, restore_rng
from research.route_compare_20261102.pilot import RoutePilot
from research.route_compare_20261102.response import federated_response
from .pilot import BilateralPilot


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.num_features = 4
        self.adapter = QVAdapter(torch.nn.Linear(4, 12), 2)
    def forward_features(self, x):
        q, _, v = self.adapter(x).chunk(3, dim=-1)
        return q+v


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.backbone = Backbone()
        self.adapters = {'0': self.backbone.adapter}
        self.rows = torch.nn.ParameterList()
        self.backbone.requires_grad_(False)
        for b in self.b_parameters(): b.requires_grad_(True)
    def b_parameters(self): return [self.backbone.adapter.Bq, self.backbone.adapter.Bv]
    def add_head(self, cpt):
        for row in self.rows: row.requires_grad_(False)
        self.rows.append(torch.nn.Parameter(torch.randn(cpt, 4)*.02))
    def forward(self, x):
        z = F.normalize(self.backbone.forward_features(x)[:, 0], dim=-1)
        return 4*F.linear(z, F.normalize(torch.cat(list(self.rows)), dim=-1))


def load(raw, ids, batch, shuffle=False):
    order = torch.randperm(len(ids)).tolist() if shuffle else list(range(len(ids)))
    ordered = [ids[i] for i in order]
    return [tuple(torch.stack(v) for v in zip(*(raw[j] for j in ordered[i:i+batch])))
            for i in range(0, len(ids), batch)]


def config(arm='E3'):
    return {'rank': 2, 'clients': 2, 'cpt': 2, 'tasks': 2, 'rounds': 2, 'epochs': 2, 'batch': 2,
            'lr': .002, 'retain_round_checkpoints': True, 'retain_task_checkpoints': False,
            'bilateral_probe': False, 'route': {'optimizer': 'adamw', 'mode': 'rrs',
            'pool_width': 3, 'split_seed': 76103}, 'bilateral': {'arm': arm, 'gamma': .1, 'epsilon': .01}}


class LifecycleTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2); torch.manual_seed(37)
        self.raw = [(torch.randn(3, 4), torch.tensor(t*2+i % 2)) for t in range(2) for i in range(12)]
        self.manifest = {'labels': {'train': [int(y) for _, y in self.raw],
                                   'test': [int(y) for _, y in self.raw]},
            'partitions': {s: [[list(range(t*12, t*12+6)), list(range(t*12+6, (t+1)*12))]
                              for t in range(2)] for s in ('train', 'test')}}

    def build(self, directory, arm='E3', cls=BilateralPilot):
        return cls(Model().eval(), [self.raw, self.raw], copy.deepcopy(self.manifest), load,
            lambda x: x, lambda x: x, config(arm), directory, {'test': 'synthetic'},
            {'started': 1., 'deadline': 1e12})

    def test_lifecycle_checkpoint_restore_and_E0_exact_regression(self):
        with tempfile.TemporaryDirectory() as d:
            state = rng_state()
            plain = self.build(Path(d)/'plain', 'E0', RoutePilot)
            initial = copy.deepcopy(plain.model.state_dict()); run_rng = rng_state()
            plain.run(); plain_rng = rng_state()
            restore_rng(state)
            study = self.build(Path(d)/'study', 'E0')
            study.model.load_state_dict(initial); restore_rng(run_rng)
            study.run(); study_rng = rng_state()
            for k, v in plain.model.state_dict().items(): self.assertTrue(torch.equal(v, study.model.state_dict()[k]))
            self.assertEqual(plain.s['communication'], study.s['communication'])
            self.assertEqual(plain.s['matrix'], study.s['matrix'])
            self.assertTrue(torch.equal(plain_rng['torch'], study_rng['torch']))
            self.assertEqual(study.s['merges'], [0, 1]); self.assertEqual(study.s['history'].committed, {0, 1})
            self.assertEqual(study.s['bilateral']['applied'], [0, 1])
            # Nonidentity path: checkpoint at a corrected T2 round boundary.
            active = self.build(Path(d)/'active')
            while (active.s['task'], active.s['round'], active.s['phase']) != (1, 1, 'round_start'):
                self.assertTrue(active.step())
            counts = copy.deepcopy(active.s['bilateral']['applied'])
            model = copy.deepcopy(active.model.state_dict())
            round_file = torch.load(active.out/'round_t1_r1.pt', weights_only=False)
            for v, p in zip(round_file['B'], active.model.b_parameters()): self.assertTrue(torch.equal(v, p))
            saved = copy.deepcopy(torch.load(active.out/'latest.pt', weights_only=False))
            active.run(); final = copy.deepcopy(active.model.state_dict())
            resumed = self.build(Path(d)/'active')
            torch.save(saved, resumed.out/'latest.pt'); resumed.restore()
            self.assertEqual(resumed.s['bilateral']['applied'], counts)
            for k, v in model.items(): self.assertTrue(torch.equal(v, resumed.model.state_dict()[k]))
            resumed.run()
            for k, v in final.items(): self.assertTrue(torch.equal(v, resumed.model.state_dict()[k]))
            self.assertEqual(resumed.s['merges'], [0, 1])

    def test_two_round_functional_real_trainer_match_and_outer(self):
        from research.head_regmean.solver import aggregate_heads
        from .prox import server_map
        model = Model().eval(); model.add_head(2); model.add_head(2)
        model.backbone.adapter.start(torch.eye(4)[:2])
        x = torch.randn(2, 3, 4); y = torch.tensor([2, 3])
        rounds = [[[(x, y)]*4 for _ in range(2)] for _ in range(2)]
        gram = [[(x, y)]]*2
        a = model.adapters['0'].A.clone().requires_grad_()
        mem = {'U': torch.eye(4, dtype=torch.float64)[:, :2],
               'V': torch.eye(4, dtype=torch.float64)[:, :2], 'alpha': torch.tensor([1., .3], dtype=torch.float64)}
        c = torch.diag(torch.tensor([1., 3., 2., 4.], dtype=torch.float64))
        saved = copy.deepcopy(model.state_dict()); rng = rng_state()
        for arm in ('E0', 'E1', 'E2', 'E3'):
            def operator(bs, bases): return [server_map(b, bases['0'], mem, c, arm)[0] for b in bs]
            args = (model, {'0': a}, rounds, gram, gram, [4, 4], [2, 2], 'adamw', .002, 5, 2, 1, 2)
            out = federated_response(*args, use_checkpoint=False, track_response=False, server_operator=operator)
            # Independent production optimizer/aggregation loop, same short schedule.
            local = copy.deepcopy(model)
            bglobal = [p.detach().clone() for p in model.b_parameters()]; hglobal = model.rows[-1].detach().clone()
            import math
            for r in range(2):
                ends, heads, grams = [], [], []
                for k in range(2):
                    with torch.no_grad():
                        for p, v in zip(local.b_parameters(), bglobal): p.copy_(v)
                        local.rows[-1].copy_(hglobal)
                    opt = torch.optim.AdamW(local.b_parameters(), lr=0, weight_decay=0)
                    hopt = torch.optim.AdamW([local.rows[-1]], lr=.001, weight_decay=0)
                    for step in range(4):
                        local.zero_grad(set_to_none=True)
                        F.cross_entropy(local(x)[:, 2:], y-2).backward()
                        for group in opt.param_groups: group['lr'] = .002*.5*(1+math.cos(math.pi*(r*4+step)/19))
                        opt.step(); hopt.step()
                    ends.append([p.detach().clone() for p in local.b_parameters()]); heads.append(local.rows[-1].detach().clone())
                    with torch.no_grad():
                        z = F.normalize(local.backbone.forward_features(x)[:, 0], dim=-1).double()
                    grams.append(z.T@z*2)
                bglobal = operator([sum(e[j]*.5 for e in ends) for j in range(2)], {'0': a.detach()})
                hglobal = aggregate_heads(heads, grams, [4, 4], True)[0]
            for b, expected in zip(out['B'], bglobal): torch.testing.assert_close(b, expected, rtol=2e-5, atol=2e-7)
            torch.testing.assert_close(out['head'], hglobal, rtol=2e-5, atol=2e-7)
            tracked = federated_response(*args, use_checkpoint=True, server_operator=operator)
            torch.testing.assert_close(tracked['loss'].detach(), out['loss'], rtol=0, atol=1e-7)
            grad = torch.autograd.grad(tracked['loss'], a)[0]
            self.assertTrue(torch.isfinite(grad).all())
            for key, value in saved.items(): self.assertTrue(torch.equal(value, model.state_dict()[key]))
            self.assertTrue(torch.equal(rng['torch'], rng_state()['torch']))


if __name__ == '__main__': unittest.main()
