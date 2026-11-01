"""Unregistered task-boundary comparison of constrained endpoint moments."""
from research.adamw_combined.multistep_pilot_draft import MultistepPilot
from research.adamw_combined.constrained_multistep_draft import propose_from_trials


class ConstrainedPilot(MultistepPilot):
    def propose_layers(self, pools, histories, trials):
        candidates, records = propose_from_trials(self.model, pools, histories, trials,
                                                   self.cfg['multistep']['moment_mode'])
        self.s.setdefault('constraint_records', []).append({'task':self.s['task'], 'layers':records})
        return candidates, {k: 'accepted' if v['accepted'] else v['reason'] for k,v in records.items()}
