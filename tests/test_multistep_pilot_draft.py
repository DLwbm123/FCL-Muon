import torch
from test_aggregation_utility import toy
from research.adamw_combined.multistep_pilot_draft import MultistepPilot


def test_multistep_pilot_runs_task_boundary_and_restores(tmp_path):
    pilot = toy(tmp_path/'trial', kind='prefix')
    pilot.__class__ = MultistepPilot
    pilot.cfg['clients'] = 2
    pilot.cfg['server_correction'] = False
    pilot.cfg['multistep'] = dict(pool_width=4, split_seed=76103, check_fraction=.2,
                                  client_cap=4, trial_epochs=1, trial_seed=123)
    pilot.run()
    assert pilot.s['phase'] == 'complete'
    assert len(pilot.s['selection_records']) == 2
    assert all(r['trial_train_visits'] > 0 for r in pilot.s['selection_records'])
    assert len(pilot.s['training_records']) == 12
    saved = {k: v.clone() for k, v in pilot.model.state_dict().items()}
    pilot.restore()
    for key, value in saved.items():
        assert torch.equal(value, pilot.model.state_dict()[key])


def test_constrained_pilot_runs_both_moments_without_lifecycle_errors(tmp_path):
    from research.adamw_combined.constrained_pilot_draft import ConstrainedPilot
    for mode in ('mean', 'individual'):
        pilot = toy(tmp_path/mode, kind='prefix')
        pilot.__class__ = ConstrainedPilot
        pilot.cfg['clients'] = 2
        pilot.cfg['server_correction'] = False
        pilot.cfg['multistep'] = dict(pool_width=4, split_seed=76103, check_fraction=.2,
                                      client_cap=4, trial_epochs=1, trial_seed=123, moment_mode=mode)
        pilot.run()
        assert pilot.s['phase'] == 'complete'
        assert len(pilot.s['selection_records']) == 2
        assert len(pilot.s['constraint_records']) == 2
        assert len(pilot.s['training_records']) == 12
        saved = {k: v.clone() for k, v in pilot.model.state_dict().items()}
        pilot.restore()
        for key, value in saved.items(): assert torch.equal(value, pilot.model.state_dict()[key])
