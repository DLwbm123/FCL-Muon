"""Synthetic T1 lifecycle and real guarded checkpoint import; no datasets."""
import copy
import torch
from torch import nn
from fcl.model import MuLoRAFL
from research.adamw_combined.bilevel_pilot_draft import GuardedBilevelPilot
from research.adamw_combined.guarded_prefix_draft import import_guarded_prefix
from test_aggregation_utility import toy
from test_head_regmean_pilot import Backbone, Augment, loader


def test_guarded_prefix_real_model_import_and_restore(tmp_path):
    source = toy(tmp_path / 'source', kind='prefix')
    source.cfg['clients'] = 2
    source.manifest['cpt'] = 2
    source.manifest['labels']['test'] = list(source.manifest['labels']['train'])
    for split in ('train', 'test'):
        source.manifest['partitions'][split] = [groups[:2] for groups in source.manifest['partitions'][split]]
    source.raw[0].targets = source.manifest['labels']['train']
    while not (source.s['task'] == 1 and source.s['phase'] == 'task_start'):
        source.step()
    checkpoint = torch.load(source.out / 'latest.pt', map_location='cpu', weights_only=False)
    original_config = copy.deepcopy(checkpoint['config'])
    target_config = copy.deepcopy(checkpoint['config'])
    target_config['bilevel'] = {'mode': 'optimized', 'split_seed': 42,
                                'client_cap': 4, 'trial_epochs': 1, 'step': .01}
    clock = {'started': 3., 'deadline': 1e30}
    def make():
        return GuardedBilevelPilot(MuLoRAFL(Backbone(), 6, 2, 16), source.raw, source.manifest,
                loader, Augment(), nn.Identity(), target_config, tmp_path / 'target', checkpoint['identity'],
                clock, holdout_tasks=[1, 2], holdout_seed=7, holdout_fraction=.25)
    target = make()
    import_guarded_prefix(target, checkpoint, source.manifest, checkpoint['clock'])
    assert checkpoint['config'] == original_config
    assert target.s['prefix_provenance']['guarded_partition']
    assert (target.s['task'], target.s['phase']) == (1, 'task_start')
    for name, value in checkpoint['model'].items():
        assert torch.equal(target.model.state_dict()[name], value)
    reloaded = make()
    reloaded.restore()
    assert reloaded.identity == target.identity
    assert reloaded.s['prefix_provenance'] == target.s['prefix_provenance']
    for name, value in target.model.state_dict().items():
        assert torch.equal(reloaded.model.state_dict()[name], value)

    # Exercise the inherited RRS, real local training/Gram, and server probe
    # against synthetic data while recording the dataset and indices supplied
    # to the loader, including the actual task-boundary proposal.
    from research.adamw_combined.pilot import probe
    seen = []
    def audited_loader(raw, ids, *args):
        seen.append((raw, tuple(ids), bool(getattr(raw, '_active', False))))
        return loader(raw, ids, *args)
    target.loader = audited_loader
    target.cfg['server_correction'] = True
    target.step()  # T2 RRS
    assert target.s['selection_records'][-1]['calls']
    target.step()  # begin first round
    target.step()  # first client, including endpoint Gram
    target.step()  # second client, including endpoint Gram
    target.step()  # production head aggregation and server correction
    groups = [target.ids('train', 1, k) for k in range(2)]
    probe(target, groups, True)
    held = {i for groups in target.reserved.values() for group in groups for i in group}
    assert len(seen) >= 5
    assert all(raw is target.raw[0] and held.isdisjoint(ids) for raw, ids, active in seen)
    assert target.raw[0].holdout_reads == 0
    assert len(target.s['combined_server_records']) == 1
    while target.step():
        pass
    assert target.s['phase'] == 'complete'
    assert len(target.s['combined_server_records']) == 4
    for raw, ids, active in seen:
        if raw is target.raw[0]:
            assert active or held.isdisjoint(ids)
        else:
            assert raw is target.raw[1]

    control_cfg = copy.deepcopy(target_config)
    control_cfg['server_correction'] = True
    control_cfg['bilevel']['mode'] = 'fixed_rrs'
    control = GuardedBilevelPilot(MuLoRAFL(Backbone(), 6, 2, 16), source.raw, source.manifest,
            loader, Augment(), nn.Identity(), control_cfg, tmp_path / 'control', checkpoint['identity'],
            clock, holdout_tasks=[1, 2], holdout_seed=7, holdout_fraction=.25)
    import_guarded_prefix(control, checkpoint, source.manifest, checkpoint['clock'])
    while control.step():
        pass
    assert control.s['phase'] == 'complete'
    assert control.manifest == target.manifest
    assert all(r['accepted_layers'] == 0 for r in control.s['selection_records'])
    assert control.s['selection_records'][0]['calls'] == target.s['selection_records'][0]['calls']
    assert sum(r['visited'] for r in control.s['training_records']) == sum(
        r['visited'] for r in target.s['training_records'])
