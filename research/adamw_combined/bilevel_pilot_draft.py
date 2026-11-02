"""Unregistered task-boundary integration; not a production experiment runner.

No source validation, independent holdout, communication accounting, resource
supervision, or released protocol is provided here. Do not launch real training.
"""
import math
import os
import hashlib
import json
import torch
from fcl.checkpoint import rng_state, restore_rng
from research.adamw_combined.multistep_pilot_draft import MultistepPilot
from research.adamw_combined.bilevel_step_draft import propose_step
from research.adamw_combined.differentiable_response_draft import differentiable_response
from research.adamw_combined.implicit_head_draft import UnsupportedHeadDerivative
from research.adamw_combined.holdout_firewall_draft import build_firewalled_manifest, GuardedTrainingDataset


class BilevelPilot(MultistepPilot):
    def confirm_proposal(self, baseline, candidate, task, evaluate):
        return candidate, 'not_configured'

    def deployment_candidate(self, baseline, selected, histories):
        """Validate the values after conversion to each actual basis dtype."""
        deployed = {}
        for name, original in baseline.items():
            target = self.model.adapters[name].A
            value = selected[name].detach().to(device=target.device, dtype=target.dtype)
            if value.shape != target.shape or not torch.isfinite(value).all():
                return {n: a.clone() for n, a in baseline.items()}, False
            eye = torch.eye(value.shape[0], device=value.device, dtype=value.dtype)
            h = histories[name].double()
            old = torch.trace(original.double() @ h @ original.double().T)
            new = torch.trace(value.double() @ h @ value.double().T)
            if (not torch.allclose(value @ value.T, eye, atol=1e-5, rtol=1e-5)
                    or not torch.isfinite(new) or new > old):
                return {n: a.clone() for n, a in baseline.items()}, False
            deployed[name] = value
        return deployed, True

    def _propose_multistep(self):
        cfg = self.cfg['bilevel']
        if cfg.get('mode') not in ('optimized', 'fixed_rrs'):
            raise ValueError('unknown paired control mode')
        task = self.s['task']
        clients, fit, formal_sizes, formal_batches = [], [], [], []
        preload_visits = 0
        holdout_before = getattr(self.raw[0], 'holdout_reads', 0)
        for k in range(self.cfg['clients']):
            ids = self.ids('train', task, k)
            if len(ids) < 2:
                self.s.setdefault('selection_records', []).append(
                    {'task': task, 'accepted_layers': 0, 'reason': 'insufficient_client_samples'})
                return
            generator = torch.Generator().manual_seed(cfg['split_seed'] + 10007*task + 101*k)
            order = torch.randperm(len(ids), generator=generator).tolist()
            train_ids = [ids[j] for j in order[::2]][:cfg['client_cap']]
            fit_ids = [ids[j] for j in order[1::2]][:cfg['client_cap']]
            def batches(group):
                nonlocal preload_visits
                result = []
                for x, y in self.loader(self.raw[0], group, self.cfg['batch']):
                    self.budget(); self.check_labels(y)
                    result.append((self.test_tf(x.to(self.device)), y.to(self.device)))
                    preload_visits += len(y)
                return result
            clients.append(batches(train_ids)); fit.append(batches(fit_ids))
            formal_sizes.append(len(ids))
            formal_batches.append(math.ceil(len(ids)/self.cfg['batch']))
        baseline = {n: a.A.detach().clone() for n, a in self.model.adapters.items()}
        histories = {n: self.s['history'].raw[n].to(self.device) for n in baseline}
        paired_rng = rng_state()
        calls = []
        outer_backward_attempts = 0
        def proposal_progress(kind, count):
            nonlocal outer_backward_attempts
            if kind != 'outer_backward_attempts':
                raise ValueError('unknown proposal progress')
            outer_backward_attempts += count
        def evaluate(bases, evaluation=None):
            self.budget(); restore_rng(paired_rng)
            call = {'trial_visits': 0, 'gram_visits': 0, 'evaluation_visits': 0, 'completed': False}
            calls.append(call)
            def progress(kind, count):
                call[kind] += count
            result = differentiable_response(self.model, bases, clients, formal_sizes, formal_batches,
                        fit if evaluation is None else evaluation, cfg['trial_epochs'], task, self.cfg['cpt'], self.cfg['lr'],
                        self.cfg['rounds'], self.cfg['epochs'], progress=progress,
                        track_response=evaluation is None)
            call['completed'] = True
            return result
        try:
            try:
                selected, record = propose_step(baseline, histories, evaluate, cfg['step'], progress=proposal_progress)
                if record['accepted']:
                    selected, confirmation = self.confirm_proposal(baseline, selected, task, evaluate)
                    record['confirmation'] = confirmation
                    if confirmation == 'rejected':
                        record['accepted'] = False
                        record['reason'] = 'confirmation_gate'
            except UnsupportedHeadDerivative:
                selected = {n: a.clone() for n, a in baseline.items()}
                record = {'accepted': False, 'reason': 'unsupported_head_derivative'}
        finally:
            # Probe evaluations must not advance the subsequent training stream,
            # including when a solver or budget check raises.
            restore_rng(paired_rng)
        # The fixed control incurs exactly the same candidate computation and
        # evaluations, but deliberately discards the optimized basis.
        if cfg['mode'] == 'fixed_rrs':
            selected = {n: a.clone() for n, a in baseline.items()}
        selected, deployment_valid = self.deployment_candidate(baseline, selected, histories)
        if not deployment_valid:
            record['accepted'] = False
            record['reason'] = 'deployment_geometry'
        with torch.no_grad():
            try:
                for n, adapter in self.model.adapters.items():
                    adapter.A.copy_(selected[n])
                if any(not torch.equal(adapter.A, selected[n]) for n, adapter in self.model.adapters.items()):
                    raise ArithmeticError('basis deployment mismatch')
            except Exception:
                for n, adapter in self.model.adapters.items():
                    adapter.A.copy_(baseline[n])
                raise
        accepted = sum(not torch.equal(selected[n], baseline[n]) for n in baseline)
        self.s.setdefault('selection_records', []).append(
            {'task': task, 'accepted_layers': accepted, 'reason': record['reason'],
             'mode': cfg['mode'], 'proposal_accepted': record['accepted'],
             'confirmation': record.get('confirmation', 'not_run'), 'calls': calls,
             'outer_backward_attempts': outer_backward_attempts,
             'preload_visits': preload_visits,
             'confirmation_reads': getattr(self.raw[0], 'holdout_reads', 0) - holdout_before})


class GuardedBilevelPilot(BilevelPilot):
    """Synthetic integration only; confirmation recovery is not implemented.

    Every consumer must use this instance's raw dataset and manifest. Retaining
    the original raw object bypasses the guard; this is not a security boundary.
    """
    def __init__(self, model, raw, manifest, loader, train_tf, test_tf, config,
                 out, identity, clock, *, holdout_tasks, holdout_seed, holdout_fraction):
        if len(raw) != 2:
            raise ValueError('expected training and test datasets')
        filtered, reserved = build_firewalled_manifest(manifest, holdout_tasks, holdout_seed, holdout_fraction)
        guarded = GuardedTrainingDataset(raw[0], reserved)
        # Checkpoints must distinguish different actual partitions, even when
        # they originate from the same prior manifest identifier.
        partition = {'train': filtered['partitions']['train'], 'reserved': reserved}
        fingerprint = hashlib.sha256(json.dumps(partition, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        guarded_identity = dict(identity)
        guarded_identity['guarded_partition'] = fingerprint
        super().__init__(model, (guarded, raw[1]), filtered, loader, train_tf, test_tf,
                         config, out, guarded_identity, clock)
        guarded.budget_check = lambda: self.budget()
        self.reserved = reserved

    def confirm_proposal(self, baseline, candidate, task, evaluate):
        if task not in self.reserved or any(not ids for ids in self.reserved[task]):
            raise ValueError('missing confirmation coverage')
        # Freeze local copies before allowing any reserved pixel access.
        frozen = {n: a.detach().clone() for n, a in candidate.items()}
        guard = self.raw[0]
        # Fail closed after a process restart: once confirmation begins, a new
        # pilot must not silently read the same selection data again. This does
        # not implement resume of an interrupted confirmation.
        marker = self.out / f'confirmation_task_{task}.started'
        fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, b'confirmation started; automatic retry forbidden\n')
            os.fsync(fd)
        finally:
            os.close(fd)
        directory = os.open(self.out, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        guard.freeze_candidate(task)
        try:
            with guard.confirmation(task):
                groups = []
                for ids in self.reserved[task]:
                    batches = []
                    for x, y in self.loader(guard, ids, self.cfg['batch']):
                        self.budget(); self.check_labels(y)
                        batches.append((self.test_tf(x.to(self.device)), y.to(self.device)))
                    groups.append(batches)
            before = evaluate(baseline, groups)
            after = evaluate(frozen, groups)
            values = (before['loss'], before['current'], after['loss'], after['current'])
            if not all(bool(torch.isfinite(x)) for x in values):
                raise ValueError('nonfinite confirmation response')
            if after['loss'] <= before['loss'] and after['current'] <= before['current']:
                return frozen, 'accepted'
            return {n: a.clone() for n, a in baseline.items()}, 'rejected'
        finally:
            guard.finish_task(task)
