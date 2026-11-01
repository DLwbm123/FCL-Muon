"""Unregistered integration of wide-client probes into a task boundary."""
import math
import time
import torch
from fcl.checkpoint import rng_state, restore_rng
from research.head_regmean.pilot import Pilot
from research.aggregation_utility.selection import candidate_pool
from research.aggregation_utility.pilot import split_indices
from research.adamw_combined.pilot import CombinedPilot
from research.adamw_combined.wide_model_draft import wide_model
from research.adamw_combined.multistep_probe_draft import simulate_candidates
from research.adamw_combined.multistep_subspace_draft import propose_from_trials


class MultistepPilot(CombinedPilot):
    def initialize_task(self):
        Pilot.initialize_task(self)
        if self.s['task'] == 0:
            return
        state = rng_state()
        started = time.monotonic()
        try:
            self._propose_multistep()
        finally:
            restore_rng(state)
        self.s['selection_records'][-1]['seconds'] = time.monotonic()-started

    def _propose_multistep(self):
        cfg = self.cfg['multistep']
        task = self.s['task']
        pools, histories = {}, {}
        for name, adapter in self.model.adapters.items():
            self.budget()
            h = self.s['history'].raw[name].to(self.device)
            pools[name], _, _ = candidate_pool(self.s['pending'][name].to(self.device), h,
                                                adapter.A, cfg['pool_width'])
            histories[name] = h
        fit_groups, formal_batches = [], []
        for k in range(self.cfg['clients']):
            ids = self.ids('train', task, k)
            fit, _ = split_indices(ids, self.manifest['labels']['train'], task, k,
                                   self.cfg['cpt'], cfg['split_seed'], cfg['check_fraction'])
            fit_groups.append(fit[:cfg['client_cap']])
            formal_batches.append(math.ceil(len(ids)/self.cfg['batch']))
        if any(not fit for fit in fit_groups):
            self.s.setdefault('selection_records', []).append({'task':task, 'accepted_layers':0,
                'reason':'empty_client', 'trial_train_visits':0, 'probe_forward_images':0})
            return
        groups = []
        for fit in fit_groups:
            batches = []
            for x, y in self.loader(self.raw[0], fit, self.cfg['batch']):
                self.budget(); self.check_labels(y)
                batches.append((self.test_tf(x.to(self.device)), y.to(self.device)))
            groups.append(batches)
        self.budget()
        wide = wide_model(self.model, pools)
        trials = simulate_candidates(wide, [{name: p.T for name, p in pools.items()}], groups,
                    formal_batches, cfg['trial_epochs'], task, self.cfg['cpt'], self.cfg['lr'],
                    self.cfg['rounds'], cfg['trial_seed']+task*10000,
                    schedule_epochs=self.cfg['epochs'])[0]
        self.budget()
        self.charge('multistep_pool_broadcast', self.cfg['clients']*sum(p.numel()*p.element_size() for p in pools.values()))
        self.charge('multistep_wide_B_upload', sum(b.numel()*b.element_size() for t in trials for b in t['B']))
        self.charge('multistep_head_upload', sum(t['head'].numel()*t['head'].element_size() for t in trials))
        selected, reasons = propose_from_trials(self.model, pools, histories, trials)
        with torch.no_grad():
            for name, adapter in self.model.adapters.items():
                adapter.A.copy_(selected[name])
        samples = sum(x['samples'] for x in trials)
        visits = sum(x['train_visits'] for x in trials)
        self.s.setdefault('selection_records', []).append({'task':task,
            'accepted_layers':sum(v == 'accepted' for v in reasons.values()),
            'reasons':reasons, 'trial_train_visits':visits,
            'probe_forward_images':2*samples, 'fit_samples':samples})
        self.event('multistep_selection', accepted_layers=self.s['selection_records'][-1]['accepted_layers'],
                   trial_train_visits=visits, probe_forward_images=2*samples)
