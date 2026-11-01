"""Unregistered task-boundary integration for paired aggregate-response selection."""
import math
import time
import torch
from research.adamw_combined.multistep_pilot_draft import MultistepPilot
from research.adamw_combined.multistep_probe_draft import paired_probes, simulate_candidates
from research.adamw_combined.aggregated_response_draft import paired_aggregates
from research.adamw_combined.constrained_multistep_draft import propose_from_trials
from research.adamw_combined.wide_model_draft import wide_model
from research.adamw_combined.pilot import probe
from research.aggregation_utility.selection import candidate_pool


def split_four(ids, seed, cap):
    if len(ids) < 4 or cap < 1 or len(set(ids)) != len(ids):
        raise ValueError('insufficient or duplicated client samples')
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(ids), generator=g).tolist()
    groups = [[ids[j] for j in order[i::4]][:cap] for i in range(4)]
    if any(not v for v in groups) or len(set(sum(groups, []))) != sum(map(len, groups)):
        raise ValueError('invalid client split')
    return groups


class AggregatedResponsePilot(MultistepPilot):
    def _propose_multistep(self):
        cfg = self.cfg['multistep']; task = self.s['task']; started = time.monotonic()
        sizes = [len(self.ids('train', task, k)) for k in range(self.cfg['clients'])]
        try:
            splits = [split_four(self.ids('train', task, k), cfg['split_seed']+10007*task+101*k,
                                 cfg['client_cap']) for k in range(self.cfg['clients'])]
        except ValueError:
            self.s.setdefault('selection_records', []).append({'task': task, 'accepted_layers': 0,
                'reason': 'insufficient_client_coverage', 'trial_train_visits': 0,
                'probe_forward_images': 0})
            return
        groups = [[v[i] for v in splits] for i in range(4)]
        def batches(ids):
            out = []
            for x, y in self.loader(self.raw[0], ids, self.cfg['batch']):
                self.budget(); self.check_labels(y)
                out.append((self.test_tf(x.to(self.device)), y.to(self.device)))
            return out
        generation, training = [[batches(ids) for ids in groups[i]] for i in (0, 1)]
        formal_batches = [math.ceil(n/self.cfg['batch']) for n in sizes]
        pools, histories = {}, {}
        for name, adapter in self.model.adapters.items():
            self.budget(); h = self.s['history'].raw[name].to(self.device)
            pools[name], _, _ = candidate_pool(self.s['pending'][name].to(self.device), h,
                                                adapter.A, cfg['pool_width'])
            histories[name] = h
        anchor = {name: a.A.detach().clone() for name, a in self.model.adapters.items()}
        wide = wide_model(self.model, pools)
        generation_trials = simulate_candidates(wide, [{n:p.T for n,p in pools.items()}], generation,
                formal_batches, cfg['trial_epochs'], task, self.cfg['cpt'], self.cfg['lr'],
                self.cfg['rounds'], cfg['trial_seed']+task*10000, schedule_epochs=self.cfg['epochs'])[0]
        mean, mean_rec = propose_from_trials(self.model, pools, histories, generation_trials, 'mean')
        individual, ind_rec = propose_from_trials(self.model, pools, histories, generation_trials, 'individual')
        candidates = [anchor, mean, individual]
        def measure(_):
            self.budget(); result, _ = probe(self, groups[2], False)
            return {'all': result['all'], 'current': result['current'], 'coverage': result['coverage']}
        scores = paired_aggregates(self.model, candidates, training, formal_batches, sizes,
                 cfg['trial_epochs'], task, self.cfg['cpt'], self.cfg['lr'], self.cfg['rounds'],
                 cfg['trial_seed']+task*10000+5000, measure, schedule_epochs=self.cfg['epochs'])
        base = scores[0]['measurement']; chosen = 0
        for j in (1, 2):
            m = scores[j]['measurement']
            if m['coverage'] == 1 and m['current'] <= base['current'] and m['all'] < scores[chosen]['measurement']['all']-1e-6:
                chosen = j
        # Confirmation is read after the fit decision; a veto makes it part of selection.
        def confirm(j):
            with torch.no_grad():
                for name, a in self.model.adapters.items(): a.A.copy_(candidates[j][name])
                for dst, src in zip(self.model.b_parameters(), scores[j]['B']): dst.copy_(src)
                self.model.rows[-1].copy_(scores[j]['head'])
            self.budget(); result, _ = probe(self, groups[3], False)
            return {'all': result['all'], 'current': result['current'], 'coverage': result['coverage']}
        confirm_values = paired_probes(self.model, [0, chosen], confirm)
        cb, cc = confirm_values
        if chosen and not (cc['coverage'] == 1 and cc['all'] < cb['all']-1e-6 and cc['current'] <= cb['current']):
            chosen = 0
        if cfg.get('response_mode') == 'individual_fixed':
            chosen = 2
        elif cfg.get('response_mode') != 'selected':
            raise ValueError('invalid response mode')
        with torch.no_grad():
            for name, a in self.model.adapters.items(): a.A.copy_(candidates[chosen][name])
        accepted = sum(not torch.equal(anchor[n], candidates[chosen][n]) for n in anchor)
        n_gen = sum(x['samples'] for x in generation_trials)
        self.charge('response_pool_broadcast', self.cfg['clients']*sum(p.numel()*p.element_size() for p in pools.values()))
        self.charge('response_wide_B_upload', sum(b.numel()*b.element_size() for t in generation_trials for b in t['B']))
        self.charge('response_wide_head_upload', sum(t['head'].numel()*t['head'].element_size() for t in generation_trials))
        basis_bytes = sum(a.A.numel()*a.A.element_size() for a in self.model.adapters.values())
        aggregate_bytes = (sum(p.numel()*p.element_size() for p in self.model.b_parameters())+
                           self.model.rows[-1].numel()*self.model.rows[-1].element_size())
        self.charge('response_candidate_basis_broadcast', 3*len(sizes)*basis_bytes)
        self.charge('response_aggregate_evaluation_broadcast', 5*len(sizes)*aggregate_bytes)
        self.charge('response_trial_B_upload', 3*sum(p.numel()*p.element_size() for p in self.model.b_parameters())*len(sizes))
        self.charge('response_trial_head_upload', 3*self.model.rows[-1].numel()*self.model.rows[-1].element_size()*len(sizes))
        d = self.model.backbone.num_features
        self.charge('response_trial_gram_upload', 3*len(sizes)*d*d*8)
        record = {'task': task, 'accepted_layers': accepted, 'chosen': chosen,
                  'reasons': {n: ('accepted' if not torch.equal(anchor[n], candidates[chosen][n])
                                   else 'original_rrs') for n in anchor},
                  'fit_measurements': [v['measurement'] for v in scores], 'confirmation': confirm_values,
                  'trial_train_visits': sum(x['train_visits'] for x in generation_trials)+sum(x['train_visits'] for x in scores),
                  'generation_train_visits': sum(x['train_visits'] for x in generation_trials),
                  'paired_train_visits': sum(x['train_visits'] for x in scores),
                  'generation_evaluation_visits': 2*n_gen,
                  'paired_local_evaluation_visits': sum(x['trial_evaluation_visits'] for x in scores),
                  'paired_gram_visits': sum(x['gram_visits'] for x in scores),
                  'fit_visits': 3*sum(map(len,groups[2])),
                  'confirmation_visits': 2*sum(map(len,groups[3])),
                  'preload_visits': sum(map(len,groups[0]))+sum(map(len,groups[1])),
                  'probe_forward_images': 2*n_gen+sum(x['trial_evaluation_visits']+x['gram_visits'] for x in scores)+
                      3*sum(map(len,groups[2]))+2*sum(map(len,groups[3])),
                  'mean_accepted_layers': sum(v['accepted'] for v in mean_rec.values()),
                  'individual_accepted_layers': sum(v['accepted'] for v in ind_rec.values()),
                  'seconds': time.monotonic()-started}
        self.s.setdefault('selection_records', []).append(record)
        self.event('aggregate_response_selection', accepted_layers=accepted, chosen=chosen,
                   trial_train_visits=record['trial_train_visits'])
