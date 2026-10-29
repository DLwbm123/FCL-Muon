"""Accept a fixed spectral proposal only after a training-objective improvement."""
import copy
import math
import time
import torch
from .actual_update import ActualUpdatePilot
from .alignment_diagnostic import measure,select_probes
from .direction_diagnostic import reflected_proposal,choose_direction,reflected_record


def gate_accept(baseline,candidate,minimum_improvement):
    if not math.isfinite(baseline) or not math.isfinite(candidate):raise ArithmeticError('Nonfinite gate loss')
    return baseline-candidate>minimum_improvement


class GatedActualUpdatePilot(ActualUpdatePilot):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        assert self.cfg['training_gate']=={'minimum_improvement':1e-6,'probe_per_client':32,'selection_seed_base':61029}
        if self.cfg.get('direction_selection'):
            assert self.cfg['direction_selection']=={'type':'gradient_reflection','tie_tolerance':1e-12}
        self.s.update(training_gate_records=[])

    def fuse(self):
        started=time.perf_counter();t=self.s['task'];r=self.s['round']
        baseline=[b.clone() for b in self.s['agg_B']]
        directed=bool(self.cfg.get('direction_selection'))
        origin=[b.detach().cpu().clone() for b in self.model.b_parameters()] if directed else None
        super().fuse()
        proposal=copy.deepcopy(self.s['actual_update_rounds'][-1])
        candidate=[b.detach().cpu().clone() for b in self.model.b_parameters()]
        probes=select_probes(self.manifest['partitions']['train'][t],self.cfg['seed'])
        def deploy(values):
            with torch.no_grad():
                for b,v in zip(self.model.b_parameters(),values):b.copy_(v)
        deploy(baseline);before,grads=measure(self,probes,gradient=directed)
        direction=None
        if directed:
            positive_proposal=copy.deepcopy(proposal)
            negative,reflection=reflected_proposal(origin,baseline,candidate)
            sign,dots=choose_direction(grads,baseline,candidate,negative)
            if sign=='negative':
                candidate=negative
                proposal=reflected_record(proposal,origin,baseline,candidate)
                self.s['actual_update_rounds'][-1]=copy.deepcopy(proposal)
            direction={'selected':sign,'directional_derivatives':dots,'reflection_check':reflection,
                       'positive_round':positive_proposal}
            del grads
        deploy(candidate);after,_=measure(self,probes)
        accepted=gate_accept(before['weighted']['loss'],after['weighted']['loss'],self.cfg['training_gate']['minimum_improvement'])
        if not accepted:
            deploy(baseline)
            for pair in self.s['actual_update_rounds'][-1]['layers'].values():
                for v in pair:
                    v.update(identity_reason='training_loss_gate_rejected',output_norm=v['input_norm'],norm_relative_error=0.,
                             relative_correction=0.,cosine_with_fedavg=1.,actual_FP32_norm=v['input_norm'],
                             actual_FP32_norm_relative_error=0.,actual_FP32_relative_correction=0.)
                    if directed:v['direction']='identity_gate_rejected'
                    if 'singular_values_before' in v:
                        v.update(singular_values_after=list(v['singular_values_before']),energy_after=list(v['energy_before']),
                                 effective_energy_rank_after=v['effective_energy_rank_before'],singular_scale_min=1.,singular_scale_max=1.)
        # Broadcast two B points (three with orientation) and the shared head;
        # gradients remain private, with directional derivatives returned as scalars.
        nonempty=sum(bool(ids) for ids in probes)
        self.charge('gate_B_candidates_broadcast',nonempty*(3 if directed else 2)*sum(b.numel()*4 for b in self.model.b_parameters()))
        self.charge('gate_shared_head_broadcast',nonempty*self.model.rows[-1].numel()*4)
        self.charge('gate_scalar_upload',nonempty*2*2*8)
        if directed:
            self.charge('direction_derivative_scalar_upload',nonempty*2*8)
            self.charge('direction_choice_broadcast',nonempty*8)
        self.s['training_gate_records'].append({'task':t+1,'round':r+1,'accepted':accepted,
            'baseline':before,'candidate':after,'candidate_round':proposal,
            'loss_difference':after['weighted']['loss']-before['weighted']['loss'],
            'student_forward_images':before['student_forward_images']+after['student_forward_images'],
            'teacher_forward_images':before['teacher_forward_images']+after['teacher_forward_images'],
            'forward_calls':sum(v['student_forward_calls']+v['teacher_forward_calls'] for v in [before,after]),
            'seconds':time.perf_counter()-started})
        if directed:
            self.s['training_gate_records'][-1].update(direction=direction,backward_images=before['backward_images'],
                backward_calls=before['student_forward_calls'])
        self.event('training_loss_gate',accepted=accepted,loss_difference=after['weighted']['loss']-before['weighted']['loss'])
