"""Unregistered four-arm route pilot; real launch requires frozen protocol."""
import math
import time
import contextlib
import torch
from fcl.checkpoint import rng_state, restore_rng
from fcl.math import ClientMuon
from research.head_regmean.pilot import Pilot
from research.aggregation_utility.pilot import split_indices
from research.aggregation_utility.selection import candidate_pool
from research.adamw_combined.pilot import AdamAdapter
from research.route_compare_20261102.frame import reference_frame, projector_distance
from research.route_compare_20261102.response import federated_response
from research.route_compare_20261102.selection import canonical_references, select_basis
from research.response_drift_20261103.energy import response_energy


def split_support_query(manifest, task, clients, cpt, seed):
    groups = manifest['partitions']['train'][task]
    if len(groups) != clients:
        raise ValueError('client count differs from manifest')
    support, query = [], []
    for k, ids in enumerate(groups):
        if not ids:
            support.append([]); query.append([]); continue
        s, q = split_indices(ids, manifest['labels']['train'], task, k, cpt, seed, .2)
        if not s or set(s).intersection(q) or sorted(s+q) != sorted(ids):
            raise ValueError('invalid support/query partition')
        support.append(s); query.append(q)
    return support, query


def _subset(ids, limit, seed):
    if len(ids) <= limit:
        return list(ids)
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(ids), generator=g)[:limit].tolist()
    return [ids[i] for i in order]


class RoutePilot(Pilot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rrs_support = None

    def adapter_optimizer(self, parameters, state):
        mode = self.cfg['route']['optimizer']
        if mode == 'adamw':
            return AdamAdapter(parameters)
        if mode == 'muon':
            return ClientMuon(parameters, state, [], [], False)
        raise ValueError('unknown local optimizer')

    def ids(self, split, t, k=None):
        if split == 'train' and self._rrs_support is not None and t == self.s['task']:
            return self._rrs_support[k] if k is not None else [v for group in self._rrs_support for v in group]
        return super().ids(split, t, k)

    def initialize_task(self):
        task = self.s['task']
        if task == 0:
            return super().initialize_task()
        cfg = self.cfg['route']
        support, query = split_support_query(self.manifest, task, self.cfg['clients'],
                                             self.cfg['cpt'], cfg['split_seed'])
        self._rrs_support = support
        try:
            super().initialize_task()
        finally:
            self._rrs_support = None
        # Freeze the task's historic metric and identical canonical RRS gauge
        # for selection and control. Only current-task S is used for the pool.
        started = time.monotonic()
        anchors = {name: a.A.detach().clone() for name, a in self.model.adapters.items()}
        histories = {name: self.s['history'].raw[name].to(self.device) for name in anchors}
        frames, pool_failures = {}, {}
        for name in anchors:
            self.budget()
            try:
                pool, _, _ = candidate_pool(self.s['pending'][name].to(self.device), histories[name],
                                             anchors[name], cfg['pool_width'])
                frames[name] = reference_frame(pool, anchors[name])
            except (ArithmeticError, RuntimeError) as error:
                # A failed spectral pool is frozen at its reference, while
                # other layers continue through the same whole-model gate.
                old = anchors[name].double().T
                q = torch.linalg.qr(old, mode='complete').Q
                pool = torch.cat((old, q[:, old.shape[1]:cfg['pool_width']]), dim=1)
                frames[name] = reference_frame(pool, anchors[name])
                pool_failures[name] = str(error)
        references = canonical_references(frames)
        if any(not torch.isfinite(a).all() for a in references.values()):
            raise ArithmeticError('invalid canonical reference')
        ref_changes = {name: {'value_max_abs': float((references[name]-anchors[name]).abs().max()),
                              'projector_distance': float(projector_distance(references[name], anchors[name]))}
                       for name in anchors}
        # All arms use this exact reference gauge, including RRS control.
        with torch.no_grad():
            for name, adapter in self.model.adapters.items():
                adapter.A.copy_(references[name])
        full_sizes = [len(Pilot.ids(self, 'train', task, k)) for k in range(self.cfg['clients'])]
        query_weight = sum(n for n, q in zip(full_sizes, query) if q)
        labels = self.manifest['labels']['train']
        split_record = {'task': task, 'support_sizes': [len(v) for v in support],
                        'query_sizes': [len(v) for v in query],
                        'query_clients': sum(bool(v) for v in query),
                        'query_weight_coverage': query_weight/sum(full_sizes),
                        'query_classes': sorted({labels[i] for group in query for i in group})}
        record = {'task': task, 'route': cfg['mode'], 'optimizer': cfg['optimizer'],
                  'reference_changes': ref_changes, 'pool_failures': pool_failures,
                  'split': split_record, 'changed_layers': 0}
        if cfg['mode'] == 'selected' and query_weight:
            before = rng_state()
            try:
                record.update(self._select(frames, histories, support, query, full_sizes, pool_failures))
            finally:
                restore_rng(before)
        elif cfg['mode'] == 'selected':
            record['reason'] = 'no_query_coverage'
        elif cfg['mode'] != 'rrs':
            raise ValueError('unknown route mode')
        record['seconds'] = time.monotonic() - started
        self.s.setdefault('route_records', []).append(record)

    def _select(self, frames, histories, support, query, full_sizes, pool_failures):
        cfg, task = self.cfg['route'], self.s['task']
        clients = len(full_sizes)
        sids = [_subset(ids, cfg['sample_cap'], cfg['trial_seed']+10007*task+101*k)
                for k, ids in enumerate(support)]
        qids = [_subset(ids, cfg['sample_cap'], cfg['trial_seed']+20011*task+103*k)
                for k, ids in enumerate(query)]
        def load(ids, transform, batch):
            out = []
            for x, y in self.loader(self.raw[0], ids, batch):
                self.budget(); self.check_labels(y)
                out.append((transform(x.to(self.device)), y.to(self.device)))
            return out
        gram = [load(ids, self.test_tf, 4) for ids in sids]
        queries = [load(ids, self.test_tf, 4) for ids in qids]
        rounds = []
        for r in range(cfg['short_rounds']):
            row = []
            for k, ids in enumerate(sids):
                if not ids:
                    row.append([]); continue
                gen = torch.Generator().manual_seed(cfg['trial_seed']+30011*task+1009*r+101*k)
                perm = torch.randperm(len(ids), generator=gen).tolist()
                ordered = [ids[j] for j in perm]
                batches = []
                for step in range(4):
                    count = min(4, len(ordered))
                    indices = [ordered[(step*4+i) % len(ordered)] for i in range(count)]
                    batch = load(indices, self.train_tf, 4)
                    if len(batch) != 1:
                        raise ValueError('invalid trial batch')
                    batches.extend(batch)
                row.append(batches)
            rounds.append(row)
        paired = rng_state()
        calls = []
        def evaluate(bases, track):
            self.budget(); restore_rng(paired)
            offload = torch.autograd.graph.save_on_cpu(pin_memory=False) if track else contextlib.nullcontext()
            with offload:
                result = federated_response(self.model, bases, rounds, gram, queries, full_sizes,
                            [math.ceil(n/self.cfg['batch']) for n in full_sizes],
                            cfg['optimizer'], self.cfg['lr'], self.cfg['rounds'], self.cfg['epochs'],
                            task, self.cfg['cpt'], use_checkpoint=True, track_response=track,
                            server_operator=getattr(self, 'response_server_operator', None))
            calls.append({'track_outer': track, **result['counts'],
                          'query_clients': result['query_client_coverage'],
                          'query_weight_coverage': result['query_weight_coverage'],
                          'query_classes': result['query_classes']})
            measured = {'loss': result['loss'], 'current': result['current']}
            if not track and cfg.get('measure_response_energy', False):
                names = list(self.model.adapters)
                expected = [p for name in names for p in
                            (self.model.adapters[name].Bq, self.model.adapters[name].Bv)]
                if any(a is not b for a, b in zip(expected, self.model.b_parameters())) or len(result['B']) != len(expected):
                    raise ValueError('unexpected Q/V endpoint order')
                measured['energy'] = {name: response_energy(bases[name], result['B'][2*i],
                    result['B'][2*i+1], histories[name]) for i, name in enumerate(names)}
            return measured
        selected, record = select_basis(frames, histories, evaluate, steps=cfg['steps'],
                        lambdas=cfg['lambdas'], trust_radius=cfg['trust_radius'],
                        orthogonal_tolerance=cfg['orthogonal_tolerance'],
                        loss_tolerance=cfg['loss_tolerance'], space_tolerance=cfg['space_tolerance'],
                        outer_iterations=cfg['outer_iterations'], frozen_layers=pool_failures,
                        energy_gate=cfg.get('response_energy_gate', False))
        with torch.no_grad():
            originals = {name: a.A.clone() for name, a in self.model.adapters.items()}
            try:
                for name, adapter in self.model.adapters.items(): adapter.A.copy_(selected[name])
                if any(not torch.equal(a.A, selected[name]) for name, a in self.model.adapters.items()):
                    raise ArithmeticError('selected basis copy mismatch')
            except BaseException:
                for name, adapter in self.model.adapters.items(): adapter.A.copy_(originals[name])
                raise
        record['calls'] = calls
        record['support_sample_visits'] = sum(map(len, sids))
        record['query_sample_visits'] = sum(map(len, qids))
        return record
