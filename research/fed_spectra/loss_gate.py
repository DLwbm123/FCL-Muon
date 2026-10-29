"""Accept a fixed spectral proposal only after a training-objective improvement."""
import copy
import math
import time
import torch
from .actual_update import ActualUpdatePilot
from .alignment_diagnostic import measure,select_probes


def gate_accept(baseline,candidate,minimum_improvement):
    if not math.isfinite(baseline) or not math.isfinite(candidate):raise ArithmeticError('Nonfinite gate loss')
    return baseline-candidate>minimum_improvement


class GatedActualUpdatePilot(ActualUpdatePilot):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        assert self.cfg['training_gate']=={'minimum_improvement':1e-6,'probe_per_client':32,'selection_seed_base':61029}
        self.s.update(training_gate_records=[])

    def fuse(self):
        started=time.perf_counter();t=self.s['task'];r=self.s['round']
        baseline=[b.clone() for b in self.s['agg_B']]
        super().fuse()
        proposal=copy.deepcopy(self.s['actual_update_rounds'][-1])
        candidate=[b.detach().cpu().clone() for b in self.model.b_parameters()]
        probes=select_probes(self.manifest['partitions']['train'][t],self.cfg['seed'])
        def deploy(values):
            with torch.no_grad():
                for b,v in zip(self.model.b_parameters(),values):b.copy_(v)
        deploy(baseline);before,_=measure(self,probes)
        deploy(candidate);after,_=measure(self,probes)
        accepted=gate_accept(before['weighted']['loss'],after['weighted']['loss'],self.cfg['training_gate']['minimum_improvement'])
        if not accepted:
            deploy(baseline)
            for pair in self.s['actual_update_rounds'][-1]['layers'].values():
                for v in pair:
                    v.update(identity_reason='training_loss_gate_rejected',output_norm=v['input_norm'],norm_relative_error=0.,
                             relative_correction=0.,cosine_with_fedavg=1.,actual_FP32_norm=v['input_norm'],
                             actual_FP32_norm_relative_error=0.,actual_FP32_relative_correction=0.)
                    if 'singular_values_before' in v:
                        v.update(singular_values_after=list(v['singular_values_before']),energy_after=list(v['energy_before']),
                                 effective_energy_rank_after=v['effective_energy_rank_before'],singular_scale_min=1.,singular_scale_max=1.)
        # A distributed implementation broadcasts both B candidates and the shared
        # head for client-local probes; aggregate only scalar losses, never images.
        nonempty=sum(bool(ids) for ids in probes)
        self.charge('gate_B_candidates_broadcast',nonempty*2*sum(b.numel()*4 for b in self.model.b_parameters()))
        self.charge('gate_shared_head_broadcast',nonempty*self.model.rows[-1].numel()*4)
        self.charge('gate_scalar_upload',nonempty*2*2*8)
        self.s['training_gate_records'].append({'task':t+1,'round':r+1,'accepted':accepted,
            'baseline':before,'candidate':after,'candidate_round':proposal,
            'loss_difference':after['weighted']['loss']-before['weighted']['loss'],
            'student_forward_images':before['student_forward_images']+after['student_forward_images'],
            'teacher_forward_images':before['teacher_forward_images']+after['teacher_forward_images'],
            'forward_calls':sum(v['student_forward_calls']+v['teacher_forward_calls'] for v in [before,after]),
            'seconds':time.perf_counter()-started})
        self.event('training_loss_gate',accepted=accepted,loss_difference=after['weighted']['loss']-before['weighted']['loss'])
