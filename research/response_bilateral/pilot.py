"""Route policy preserved; the same B map is used in formal and short rounds."""
import copy
import time
import torch
from torch.func import functional_call
from torch.nn import functional as F
from fcl.checkpoint import rng_state, restore_rng
from research.head_regmean.pilot import observe
from research.route_compare_20261102.pilot import RoutePilot, split_support_query
from .prox import history_spectrum, server_map, diagnostics, symmetric, PSD_RTOL


class BilateralPilot(RoutePilot):
    def __init__(self, *args, pretrained_qv=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.w0 = pretrained_qv or {name: self._weights(a) for name, a in self.model.adapters.items()}
        self._geometry_cache = None

    def save(self):
        # Reserve a full atomic checkpoint temporary file before any large write.
        # No protected or historical files are removed to make room.
        limit = self.cfg.get('bilateral_storage_stop')
        if limit is not None:
            import os
            from pathlib import Path
            from research.block_response_diagnostic.boundary_budget import allocated
            root = Path(os.environ['PROJECT_ROOT']).resolve().parent
            if allocated(root)+750000000 >= limit:
                raise OSError('insufficient reserved space for atomic full checkpoint')
        return super().save()

    @staticmethod
    def _weights(a):
        d = a.Bq.shape[0]
        return {'q': a.base.weight[:d].detach().cpu().clone(),
                'v': a.base.weight[2*d:].detach().cpu().clone()}

    def _memory(self):
        t = self.s['task']
        state = self.s.get('bilateral')
        if state is not None and state['task'] == t:
            return state
        spectra = {}
        started = time.monotonic()
        for name, adapter in self.model.adapters.items():
            current = self._weights(adapter)
            spectra[name] = {p: history_spectrum(current[p].double()-self.w0[name][p].double(),
                                                2*self.cfg['rank']) for p in ('q', 'v')}
            if t == 0 and any(len(v['alpha']) for v in spectra[name].values()):
                raise ValueError('nonzero historical residual at first task')
        state = {'task': t, 'spectra': spectra, 'current': None, 'applied': [],
                 'svd_seconds': time.monotonic()-started}
        self.s['bilateral'] = state
        self._geometry_cache = None
        self.event('bilateral_history', svd_seconds=state['svd_seconds'],
                   layers={n: {p: m['info'] for p, m in pair.items()} for n, pair in spectra.items()})
        return state

    def _reference(self):
        state = self._memory()
        if state['current'] is None:
            if set(self.s['pending']) != set(self.model.adapters):
                raise ValueError('missing task-start raw patch Gram')
            current = {}
            for name, c in self.s['pending'].items():
                c = symmetric(c.detach().cpu().double())
                if not torch.isfinite(c).all(): raise ArithmeticError('nonfinite raw patch Gram')
                eig = torch.linalg.eigvalsh(c)
                if float(eig[0]) < -PSD_RTOL*max(float(eig.abs().max()), 1e-300):
                    raise ArithmeticError('negative raw patch Gram spectrum')
                current[name] = c
            state['current'] = current
        if self._geometry_cache is None or self._geometry_cache[0] != self.s['task']:
            self._geometry_cache = (self.s['task'],
                {n: {p: {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in mem.items()}
                     for p, mem in pair.items()} for n, pair in state['spectra'].items()},
                {n: c.to(self.device) for n, c in state['current'].items()})
        return self._geometry_cache[1:]

    def initialize_task(self):
        self._memory()
        super().initialize_task()
        self._reference()

    def _select(self, *args, **kwargs):
        self._reference()
        return super()._select(*args, **kwargs)

    def mapped(self, bs, bases, arm=None):
        spectra, current = self._reference()
        chosen = arm or self.cfg['bilateral']['arm']
        expected = [b for a in self.model.adapters.values() for b in (a.Bq, a.Bv)]
        if len(expected) != len(bs) or any(a is not b for a, b in zip(expected, self.model.b_parameters())):
            raise ValueError('Q/V endpoint order mismatch')
        out, coefficients = [], []
        for n, name in enumerate(self.model.adapters):
            for j, projection in enumerate(('q', 'v')):
                value, c = server_map(bs[2*n+j], bases[name], spectra[name][projection], current[name],
                                      chosen, self.cfg['bilateral']['gamma'], self.cfg['bilateral']['epsilon'])
                out.append(value); coefficients.append(float(c.detach()) if isinstance(c, torch.Tensor) else c)
        return out, coefficients

    def response_server_operator(self, bs, bases):
        # Inputs B and A remain connected. Fixed memory and C are detached.
        return self.mapped(bs, bases)[0]

    def _probe(self, bs, head, *, old=False):
        if not self.cfg.get('bilateral_probe', True): return None
        t = self.s['task']
        if old:
            # Prespecified fixed diagnostic: first ten test indices per old
            # class. All logits still use all seen classes; no task-id readout.
            ids = []
            labels = self.manifest['labels']['test']
            for label in range(t*self.cfg['cpt']):
                ids.extend([i for i, y in enumerate(labels) if y == label][:10])
            raw = self.raw[1]
        else:
            groups = (split_support_query(self.manifest, t, self.cfg['clients'], self.cfg['cpt'],
                                          self.cfg['route']['split_seed'])[0] if t else
                      self.manifest['partitions']['train'][t])
            ids = sorted(i for g in groups for i in g)[:8]
            raw = self.raw[0]
        params = dict(self.model.named_parameters())
        names = {id(p): n for n, p in params.items()}
        params.update({names[id(p)]: v.to(self.device) for p, v in zip(self.model.b_parameters(), bs)})
        params[names[id(self.model.rows[-1])]] = head.to(self.device)
        outputs, targets = [], []
        with observe(self.model):
            for x, y in self.loader(raw, ids, self.cfg['batch']):
                logits = functional_call(self.model, params, (self.test_tf(x.to(self.device)),))
                outputs.append(logits.cpu()); targets.append(y)
        if not outputs: return None
        logits, y = torch.cat(outputs), torch.cat(targets)
        return {'samples': len(y), 'all_seen_CE': float(F.cross_entropy(logits, y)),
                'all_seen_accuracy': float((logits.argmax(1) == y).double().mean()),
                'logits': logits, 'split': 'fixed_old_test_diagnostic' if old else 'current_training_support'}

    @staticmethod
    def probe_record(before, after):
        if before is None: return None
        return {'before': {k: v for k, v in before.items() if k != 'logits'},
                'after': {k: v for k, v in after.items() if k != 'logits'},
                'logit_rms_change': float((after['logits']-before['logits']).square().mean().sqrt())}

    def server_aggregate(self, proposal, head):
        state = self._memory()
        r = self.s['round']
        if r in state['applied']:
            raise ValueError('duplicate server operator in one formal round')
        bases = {n: a.A for n, a in self.model.adapters.items()}
        before = [b.to(self.device) for b in proposal]
        previous = [b.detach().clone() for b in self.model.b_parameters()]
        start = time.monotonic()
        out, coefficients = self.mapped(before, bases)
        spectra, current = self._reference()
        seconds = time.monotonic()-start
        arm = self.cfg['bilateral']['arm']
        rows = []
        for i, name in enumerate(self.model.adapters):
            for j, projection in enumerate(('q', 'v')):
                n = 2*i+j
                row = diagnostics(before[n], out[n], previous[n], bases[name], spectra[name][projection],
                                  current[name], arm, self.cfg['bilateral']['gamma'], self.cfg['bilateral']['epsilon'])
                row.update(layer=name, projection=projection, method=arm, c=coefficients[n],
                           solve_batch_seconds=seconds)
                rows.append(row)
                self.event('bilateral_operator', **row)
        if self.s['task'] == 1:
            self.s.setdefault('bilateral_T2_activity', []).extend(
                {k: row[k] for k in ('relative_correction', 'norm_before', 'layer', 'projection')} for row in rows)
        probe = self.probe_record(self._probe(before, head), self._probe(out, head))
        self.event('bilateral_probe', result=probe)
        if (arm == 'E0' and self.s['task'] == 1 and r == self.cfg['rounds']-1
                and self.cfg.get('common_intervention', False)):
            old_before = self._probe(before, head, old=True)
            for candidate in ('E1', 'E2', 'E3'):
                shadow, cs = self.mapped(before, bases, candidate)
                self.event('common_intervention', method=candidate,
                    current=self.probe_record(self._probe(before, head), self._probe(shadow, head)),
                    old=self.probe_record(old_before, self._probe(shadow, head, old=True)),
                    geometry=[diagnostics(before[2*i+j], shadow[2*i+j], previous[2*i+j], bases[name],
                        spectra[name][p], current[name], candidate) for i, name in enumerate(bases)
                        for j, p in enumerate(('q', 'v'))], c=cs)
        state['applied'].append(r)
        state['last_application'] = {'task': self.s['task'], 'round': r, 'arm': arm}
        return [v.detach().cpu().clone() for v in out]

    def restore(self):
        super().restore()
        self._geometry_cache = None
        state = self.s.get('bilateral')
        if state is not None and state['task'] == self.s['task']:
            applied = state['applied']
            if applied != list(range(self.s['round'])):
                raise ValueError('restored operator count/round mismatch')
