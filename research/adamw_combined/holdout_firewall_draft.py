"""Unregistered manifest firewall for future task-boundary candidate confirmation.

The returned manifest is for all training/statistics consumers; the private holdout
is passed only to the frozen candidate evaluator. This does not make test data blind.
"""
import copy
import math
import operator
import torch
from contextlib import contextmanager


def build_firewalled_manifest(manifest, tasks, seed, fraction):
    """Split current-task client indices without opening images or changing T1.

    Returns a new manifest and private holdout, leaving the source manifest intact.
    The caller must ensure *every* training/statistic/probe uses the new manifest.
    """
    if not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError('invalid holdout fraction')
    output = copy.deepcopy(manifest)
    source = manifest['partitions']['train']
    dest = output['partitions']['train']
    labels = manifest['labels']['train']
    cpt = int(manifest['cpt'])
    if cpt <= 0 or not tasks or len(set(tasks)) != len(tasks):
        raise ValueError('invalid tasks')
    all_source = [i for groups in source for group in groups for i in group]
    if (len(set(all_source)) != len(all_source) or
            any(not isinstance(i, int) or not 0 <= i < len(labels) for i in all_source)):
        raise ValueError('invalid or duplicated manifest indices')
    holdout = {}
    for task in tasks:
        if not isinstance(task, int) or not 0 < task < len(source):
            raise ValueError('only existing post-T1 tasks can be firewalled')
        holdout[task] = []
        for client, ids in enumerate(source[task]):
            if len(ids) < 2 or any(not task*cpt <= labels[i] < (task+1)*cpt for i in ids):
                raise ValueError('missing client coverage or incorrect task label')
            generator = torch.Generator().manual_seed(seed + 10007*task + 101*client)
            permutation = torch.randperm(len(ids), generator=generator).tolist()
            count = min(len(ids)-1, max(1, int(len(ids)*fraction)))
            reserved = sorted(ids[j] for j in permutation[:count])
            kept = sorted(ids[j] for j in permutation[count:])
            holdout[task].append(reserved)
            dest[task][client] = kept
    kept = {i for groups in dest for group in groups for i in group}
    held = [i for groups in holdout.values() for group in groups for i in group]
    if kept.intersection(held) or len(set(held)) != len(held) or kept.union(held) != set(all_source):
        raise AssertionError('firewall is not a disjoint partition')
    return output, holdout


class GuardedTrainingDataset:
    """Draft pixel-access guard; callers must not retain an unguarded dataset.

    Use one instance per worker (the current loader uses num_workers=0). This is
    an accidental-access check, not a security boundary against hostile code.
    """
    def __init__(self, raw, holdout):
        self._raw = raw
        self.budget_check = None
        self.targets = raw.targets
        self._held = frozenset(i for groups in holdout.values() for group in groups for i in group)
        self._frozen_task = None
        self._active = frozenset()
        self._completed = set()
        self._by_task = {t: frozenset(i for group in groups for i in group)
                         for t, groups in holdout.items()}
        self.holdout_reads = 0
        self.training_reads = 0

    def __len__(self):
        return len(self._raw)

    def __getitem__(self, index):
        index = operator.index(index)
        if index < 0 or index >= len(self):
            raise IndexError('training index outside dataset')
        # Check before accessing a pixel, including loader prefetch. This is
        # cooperative and does not replace an external process timeout.
        if self.budget_check is not None:
            self.budget_check()
        if index in self._held:
            if index not in self._active:
                raise PermissionError('reserved training image accessed outside confirmation')
        sample = self._raw[index]
        if index in self._held:
            self.holdout_reads += 1
        else:
            self.training_reads += 1
        return sample

    def freeze_candidate(self, task):
        if self._frozen_task is not None or task not in self._by_task or task in self._completed:
            raise ValueError('invalid freeze transition')
        self._frozen_task = task

    def finish_task(self, task):
        if self._frozen_task != task or self._active:
            raise ValueError('invalid finish transition')
        self._completed.add(task)
        self._frozen_task = None

    @contextmanager
    def confirmation(self, task):
        if task != self._frozen_task or self._active:
            raise ValueError('confirmation requires the frozen task')
        self._active = self._by_task[task]
        try:
            yield self
        finally:
            self._active = frozenset()
