"""Shared explicit operators. No automatic parameter dispatch by shape."""
import math
import torch


def mulora_ssb(g, steps=5):
    assert g.ndim == 2
    x = g.float()
    transpose = x.shape[0] > x.shape[1]
    if transpose:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        gram = x @ x.T
        x = 3.4445*x + (-4.7750*gram + 2.0315*(gram@gram))@x
    return x.T if transpose else x


def muon_operator(g, steps=5):
    """Takezawa operator: includes dimension scaling, unlike MuLoRA Eq.3."""
    return mulora_ssb(g, steps) * math.sqrt(max(g.shape))


def compact_polar(g):
    u, s, vh = torch.linalg.svd(g, full_matrices=False)
    tol = max(g.shape)*torch.finfo(g.dtype).eps*s.max()
    keep = s > tol
    return u[:, keep] @ vh[keep]


def rrs(s, history, rank, epsilon_relative=1e-6):
    s, history = s.double(), history.double()
    # Training-only scale; explicit, deterministic first-task regularization.
    scale = max(float(history.diag().mean()), float(s.diag().mean()), 1e-12)
    epsilon = epsilon_relative * scale
    eye = torch.eye(s.shape[0], dtype=s.dtype, device=s.device)
    for attempt in range(8):
        jitter = epsilon * 10**attempt
        l, info = torch.linalg.cholesky_ex((history+history.T)/2 + jitter*eye)
        if int(info) == 0:
            break
    else:
        raise RuntimeError('Historical metric failed Cholesky')
    tmp = torch.linalg.solve_triangular(l, s, upper=False)
    white = torch.linalg.solve_triangular(l, tmp.T, upper=False).T
    values, vectors = torch.linalg.eigh((white+white.T)/2)
    z = torch.linalg.solve_triangular(l.T, vectors[:, -rank:], upper=True)
    q, _ = torch.linalg.qr(z, mode='reduced')
    return q.T.float(), {'epsilon': epsilon, 'jitter': jitter, 'eigenvalues': values[-rank:].tolist()}


def aggregate(bs, bases, weights):
    if not all(torch.equal(bases[0], a) for a in bases[1:]):
        raise ValueError('Client A mismatch')
    if abs(sum(weights)-1) > 1e-6:
        raise ValueError('Weights must sum to one')
    return sum(p*b for p,b in zip(weights,bs))


class History:
    def __init__(self):
        self.raw = {}
        self.committed = set()
    def commit(self, task, stats):
        if task in self.committed:
            raise ValueError('Historical statistics already committed')
        for name, s in stats.items():
            self.raw[name] = self.raw.get(name, torch.zeros_like(s)) + s
        self.committed.add(task)


class ClientMuon:
    def __init__(self, parameters, state, local_c, global_c, corrected):
        self.parameters = list(parameters)
        self.state = state
        self.local_c = local_c
        self.global_c = global_c
        self.corrected = corrected
    @torch.no_grad()
    def step(self, lr):
        updates = []
        for i,p in enumerate(self.parameters):
            if p.grad is None:
                raise RuntimeError('Missing B gradient')
            self.state[i].mul_(0.9).add_(p.grad.float(), alpha=0.1)
            h = self.state[i]
            if self.corrected:
                h = h - self.local_c[i] + self.global_c[i]
            operator = muon_operator if self.corrected else mulora_ssb
            update = -lr*operator(h)
            p.add_(update)
            updates.append(update)
        return updates


def spectrum(x):
    s = torch.linalg.svdvals(x.float())
    energy = s.square()
    p = s / s.sum().clamp_min(1e-30)
    legacy = energy / energy.sum().clamp_min(1e-30)
    return {'fro': float(x.norm()), 'stable_rank': float(energy.sum()/energy.max().clamp_min(1e-30)),
            'effective_rank': float(torch.exp(-(p*p.clamp_min(1e-30).log()).sum())) if float(energy.sum()) else 0.0,
            'effective_rank_definition':'singular_value_amplitude_entropy',
            'legacy_energy_entropy_rank':float(torch.exp(-(legacy*legacy.clamp_min(1e-30).log()).sum())) if float(energy.sum()) else 0.0}
