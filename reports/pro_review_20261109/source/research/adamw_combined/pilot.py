"""Teacher-free AdamW with the existing federated SSB-utility basis as a proxy."""
import copy
import time
import torch
from torch.nn import functional as F
from fcl.checkpoint import rng_state, restore_rng
from research.aggregation_utility.pilot import UtilityPilot, split_indices
from research.fed_spectra.actual_update import global_shape_update
from research.fed_spectra.matrix_tangent import tangent_candidate


class AdamAdapter:
    def __init__(self, parameters):
        self.optimizer = torch.optim.AdamW(parameters, lr=0., betas=(.9, .999), eps=1e-8, weight_decay=0.)
    def step(self, lr):
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        self.optimizer.step()


def probe(pilot, groups, gradients):
    model = pilot.model
    params = model.b_parameters()
    all_params = list(model.parameters())
    saved_grads = [None if p.grad is None else p.grad.detach().clone() for p in all_params]
    modes = {m: m.training for m in model.modules()}
    buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
    rng = rng_state()
    totals = [torch.zeros_like(v, dtype=torch.float64) for v in params]
    client_weights = [len(g) for g in pilot.manifest['partitions']['train'][pilot.s['task']]]
    denom = sum(w for w, group in zip(client_weights, groups) if group)
    if denom <= 0:
        raise ValueError('empty probe')
    total_all = total_current = 0.
    samples = 0
    try:
        model.eval()
        for ids, weight in zip(groups, client_weights):
            if not ids:
                continue
            observed = 0
            for x, y in pilot.loader(pilot.raw[0], ids, pilot.cfg['batch']):
                pilot.budget(); pilot.check_labels(y)
                y = y.to(pilot.device)
                with torch.set_grad_enabled(gradients):
                    logits = model(pilot.test_tf(x.to(pilot.device)))
                    c = pilot.cfg['cpt']; task = pilot.s['task']
                    la = F.cross_entropy(logits, y, reduction='sum')
                    lc = F.cross_entropy(logits[:,task*c:(task+1)*c], y-task*c, reduction='sum')
                    if gradients:
                        gs = torch.autograd.grad(la, params)
                        for target, grad in zip(totals, gs):
                            target.add_(grad.detach().double(), alpha=weight/(denom*len(ids)))
                total_all += float(la.detach())*weight/(denom*len(ids))
                total_current += float(lc.detach())*weight/(denom*len(ids))
                observed += len(y); samples += len(y)
            if observed != len(ids):
                raise ValueError('probe count mismatch')
        if not all(torch.isfinite(x).all() for x in totals):
            raise ArithmeticError('nonfinite probe gradient')
        return {'all': total_all, 'current': total_current, 'samples': samples,
                'coverage': denom/sum(client_weights)}, [x.cpu() for x in totals] if gradients else None
    finally:
        with torch.no_grad():
            for k, v in model.named_buffers(): v.copy_(buffers[k])
        for p, g in zip(all_params, saved_grads): p.grad = g
        for m, mode in modes.items(): m.training = mode
        restore_rng(rng)


def accept_candidate(before, after):
    return (before["coverage"] >= .95 and after["all"] < before["all"]-1e-6
            and after["current"] <= before["current"]+1e-6)


class CombinedPilot(UtilityPilot):
    def adapter_optimizer(self, parameters, state):
        return AdamAdapter(parameters)

    def local_gram(self, client):
        if self.cfg['server_correction']:
            self.s.setdefault('combined_endpoints', []).append(
                {'client': client, 'B': [p.detach().cpu().clone() for p in self.model.b_parameters()]})
        return super().local_gram(client)

    def fuse(self):
        if not self.cfg['server_correction'] or self.s['task'] == 0:
            self.s.pop('combined_endpoints', None)
            return super().fuse()
        start_time = time.perf_counter()
        t = self.s['task']; round_no = self.s['round']
        origin = [p.detach().cpu().clone() for p in self.model.b_parameters()]
        endpoints = self.s.pop('combined_endpoints', [])
        weights = list(self.s['sizes'])
        expected = [k for k in range(self.cfg['clients']) if self.ids('train', t, k)]
        if [item['client'] for item in endpoints] != expected or len(weights) != len(endpoints):
            raise ValueError('missing local endpoints')
        super().fuse()
        baseline = [p.detach().cpu().clone() for p in self.model.b_parameters()]
        settings = self.cfg['selection']
        splits = [split_indices(self.ids('train', t, k), self.manifest['labels']['train'], t, k,
                  self.cfg['cpt'], settings['split_seed'], settings['check_fraction'])
                  for k in range(self.cfg['clients'])]
        fit = [a[:32] for a, _ in splits]
        before, grads = probe(self, fit, True)
        candidates, geometry = [], []
        for j, (o, b, g) in enumerate(zip(origin, baseline, grads)):
            ds = [e['B'][j].double()-o.double() for e in endpoints]
            d = b.double()-o.double()
            ref, _ = global_shape_update(ds, weights, d, eta=.1)
            candidate, rec = tangent_candidate(o, b, o.double()+ref.double(), g)
            candidates.append(candidate); geometry.append(rec)
        with torch.no_grad():
            for p, v in zip(self.model.b_parameters(), candidates): p.copy_(v)
        after, _ = probe(self, fit, False)
        accepted = accept_candidate(before, after)
        if not accepted:
            with torch.no_grad():
                for p, v in zip(self.model.b_parameters(), baseline): p.copy_(v)
        if not accepted and any(not torch.equal(p.detach().cpu(), b) for p,b in zip(self.model.b_parameters(), baseline)):
            raise AssertionError('inexact fallback')
        byte_size = sum(p.numel()*p.element_size() for p in self.model.b_parameters())
        self.s.setdefault('combined_server_records', []).append({
            'task': t, 'round': round_no, 'accepted': accepted, 'fit_before': before,
            'fit_candidate': after, 'geometry': geometry, 'forward_images': before['samples']+after['samples'],
            'backward_images': before['samples'], 'seconds': time.perf_counter()-start_time,
            'assumed_gradient_upload_bytes': len(expected)*byte_size,
            'assumed_probe_broadcast_bytes': 2*len(expected)*byte_size,
            'network_measured_bytes': None})
        self.event('combined_server', accepted=accepted, task_index=t, round_index=round_no)
