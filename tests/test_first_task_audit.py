"""Synthetic arithmetic tests only; no images, training loop or hyperparameter search."""
import ast
import copy
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F
from fcl.math import ClientMuon, mulora_ssb, muon_operator, rrs
from fcl.model import ContinualViT


def official_operator():
    root = Path(os.environ.get('EXPERIMENT_ROOT', Path(__file__).resolve().parents[1]))
    path = root / 'sources/fedmuon/code/optimizer/fedmuon_optimizer.py'
    if not path.is_file():
        pytest.skip('Pinned third-party supplement is required for the official-operator audit')
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('zeropower_via_newtonschulz5', 'muon_update')]
    assert len(nodes) == 2
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['muon_update']


def actual_head_average(heads, probs):
    # Execute the actual deployed trainer's aggregation statement, without
    # importing/running its main function or constructing an optimizer.
    root = Path(os.environ.get('EXPERIMENT_ROOT', Path(__file__).resolve().parents[1]))
    path = root / 'fcl/train.py'
    tree = ast.parse(path.read_text())
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
             and isinstance(n.value.func, ast.Attribute) and isinstance(n.value.func.value, ast.Name)
             and n.value.func.value.id == 'agghead' and n.value.func.attr == 'add_']
    assert len(nodes) == 1
    code = compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec')
    env = {'agghead': torch.zeros_like(heads[0]), 'probs': probs}
    for k, head in enumerate(heads):
        env.update(k=k, model=SimpleNamespace(rows=[head]))
        exec(code, env)
    return env['agghead']


def cosine(x, w):
    return 16 * F.normalize(x, dim=-1) @ F.normalize(w, dim=-1).T


def test_raw_head_fedavg_matches_protocol_but_not_local_function_equivalence():
    x = torch.tensor([[1., 0.]], dtype=torch.float64)
    heads = [torch.tensor([[1., 0.], [.8, .6]], dtype=torch.float64),
             torch.tensor([[0., 1.], [.8, .6]], dtype=torch.float64)]
    rescaled = [heads[0] * torch.tensor([[9.], [1.]]), heads[1].clone()]
    for a, b in zip(heads, rescaled):
        torch.testing.assert_close(cosine(x, a), cosine(x, b), atol=0, rtol=0)
    plain = actual_head_average(heads, [.5, .5])
    changed = actual_head_average(rescaled, [.5, .5])
    torch.testing.assert_close(plain, .5*heads[0]+.5*heads[1])
    assert cosine(x, plain).argmax(1).item() == 1
    assert cosine(x, changed).argmax(1).item() == 0
    # Normalizing the aggregate AFTER averaging cannot undo this dependence.
    torch.testing.assert_close(cosine(x, changed), cosine(x, F.normalize(changed, dim=-1)))
    # A mathematical reference only, NOT a proposed/deployed algorithm change.
    canonical = lambda hs: actual_head_average([F.normalize(h, dim=-1) for h in hs], [.5, .5])
    torch.testing.assert_close(canonical(heads), canonical(rescaled))


def test_sample_weighting_and_cosine_row_norm_do_not_directly_bias_logits():
    torch.manual_seed(71)
    heads = [torch.randn(10, 7, dtype=torch.float64) for _ in range(3)]
    p = [.7, .2, .1]
    result = actual_head_average(heads, p)
    torch.testing.assert_close(result, sum(w*h for w,h in zip(p, heads)))
    x = torch.randn(9, 7, dtype=torch.float64)
    scales = torch.arange(1, 11, dtype=torch.float64)[:, None]
    torch.testing.assert_close(cosine(x, result), cosine(x, result*scales))


def test_frozen_task1_rows_untouched_by_later_head_steps_and_aggregation():
    class Backbone(torch.nn.Module):
        num_features = 7
        blocks = []
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))
        def forward_features(self, x):
            return x
    model = ContinualViT(Backbone(), 30)
    model.add_head(10)
    first = model.rows[0].detach().clone()
    for _ in range(2):
        model.add_head(10)
        assert not model.rows[0].requires_grad
        # Exercise the optimizer parameter boundary with an artificial gradient;
        # this is a unit test, not model/data fitting.
        optimizer = torch.optim.AdamW([model.rows[-1]], lr=.001, weight_decay=0)
        model.rows[-1].grad = torch.ones_like(model.rows[-1])
        optimizer.step()
        with torch.no_grad():
            model.rows[-1].copy_(actual_head_average([model.rows[-1].detach(), model.rows[-1].detach()+.1], [.2, .8]))
        torch.testing.assert_close(model.rows[0], first, atol=0, rtol=0)


@pytest.mark.parametrize('shape', [(768, 32), (32, 768), (9, 3)])
def test_takezawa_operator_matches_actual_official_function_and_scale(shape):
    torch.manual_seed(72)
    g = torch.randn(shape)
    expected = official_operator()(g)
    torch.testing.assert_close(muon_operator(g), expected, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(muon_operator(g), mulora_ssb(g)*max(shape)**.5, atol=0, rtol=0)
    torch.testing.assert_close(muon_operator(torch.zeros_like(g)), torch.zeros_like(g), atol=0, rtol=0)


def test_weighted_controls_variable_local_steps_and_each_parameter_updated_once():
    torch.manual_seed(73)
    official = official_operator()
    shapes, weights, steps = [(7, 3), (3, 5)], [.8, .2], [3, 1]
    server = [torch.zeros(s) for s in shapes]
    m = [[torch.zeros(s) for s in shapes] for _ in weights]
    local_c = copy.deepcopy(m)
    global_c = copy.deepcopy(m[0])
    for _ in range(3):
        ends = []
        for k in range(2):
            parameters = [torch.nn.Parameter(s.clone()) for s in server]
            reference, ref_m = copy.deepcopy(server), copy.deepcopy(m[k])
            frozen = copy.deepcopy(local_c[k])
            opt = ClientMuon(parameters, m[k], local_c[k], global_c, True)
            for _ in range(steps[k]):
                for j, param in enumerate(parameters):
                    param.grad = torch.randn_like(param)
                    ref_m[j] = .9*ref_m[j] + .1*param.grad
                    reference[j] -= .0002*official(ref_m[j]-frozen[j]+global_c[j])
                opt.step(.0002)
                for j, param in enumerate(parameters):
                    torch.testing.assert_close(param, reference[j], atol=1e-7, rtol=1e-5)
                    torch.testing.assert_close(m[k][j], ref_m[j])
                    torch.testing.assert_close(local_c[k][j], frozen[j], atol=0, rtol=0)
            ends.append([p.detach().clone() for p in parameters])
        local_c = copy.deepcopy(m)
        global_c = [sum(weights[k]*local_c[k][j] for k in range(2)) for j in range(2)]
        server = [sum(weights[k]*ends[k][j] for k in range(2)) for j in range(2)]


def test_zero_history_rrs_is_current_principal_subspace_and_ba_has_no_hidden_scale():
    torch.manual_seed(74)
    x = torch.randn(20, 9, dtype=torch.float64)
    s = x.T@x
    a, _ = rrs(s, torch.zeros_like(s), 3)
    _, eigvec = torch.linalg.eigh(s)
    torch.testing.assert_close(a.T.double()@a.double(), eigvec[:, -3:]@eigvec[:, -3:].T, atol=2e-7, rtol=2e-6)
    b = torch.randn(11, 3)
    torch.testing.assert_close(torch.linalg.svdvals(b@a)[:3], torch.linalg.svdvals(b), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close((b@a).norm(), b.norm(), atol=1e-6, rtol=1e-6)
