"""Unregistered teacher-response Muon aggregation; no real-data run."""
import time
import torch
from fcl.checkpoint import rng_state
from research.head_regmean.pilot import Pilot
from research.continual_muon.teacher_probe_draft import probe
from research.continual_muon.teacher_eval_draft import evaluate,choose
from research.continual_muon.teacher_muon_candidate_draft import propose
from scripts.real_boundary_worker import same_rng


class TeacherServerPilot(Pilot):
    def __init__(self,*args,teacher,heldout_groups,teacher_variant='full',**kwargs):
        if teacher_variant not in ('full','no_consensus','kl_only'):raise ValueError('unknown teacher variant')
        self.teacher=teacher;self.heldout_groups=heldout_groups;self.teacher_variant=teacher_variant
        super().__init__(*args,**kwargs)

    def fuse(self):
        if self.s.get('teacher_variant')!=self.teacher_variant:raise ValueError('checkpoint variant mismatch')
        if self.s['task']!=2 or self.s['phase']!='local_done':raise ValueError('unexpected fuse lifecycle')
        groups=self.manifest['partitions']['train'][2];kcount=self.cfg['clients'];round_no=self.s['round']
        if len(groups)!=kcount or len(self.heldout_groups)!=kcount or any(not g for g in groups):
            raise ValueError('invalid client groups')
        fit=[sorted(g)[:32] for g in groups];held=[sorted(g)[:32] for g in self.heldout_groups]
        if sum(map(len,fit))!=281 or sum(map(len,held))!=236:
            raise ValueError('frozen fit/confirmation counts mismatch')
        if self.loader.allow_validation:raise ValueError('validation enabled during training')
        origins=[v.detach().cpu().clone() for v in self.model.b_parameters()]
        super().fuse();baseline=[v.detach().cpu().clone() for v in self.model.b_parameters()]
        anchor_rng=rng_state();read_before=self.loader.validation_access_images;started=time.time()
        include_evidence=self.teacher_variant!='kl_only';use_consensus=self.teacher_variant!='no_consensus'
        if use_consensus:
            grad,local_polar,fit_before,grad_cost=probe(self,self.teacher,fit,include_local_polar=True,include_evidence=include_evidence)
        else:
            grad,fit_before,grad_cost=probe(self,self.teacher,fit,include_evidence=include_evidence);local_polar=None
        proposed,proposal=propose(origins,baseline,grad,fit_before['teacher_response'],local_polar=local_polar)
        matrices=self.model.b_parameters()
        with torch.no_grad():
            for target,value in zip(matrices,proposed):target.copy_(value)
        fit_after,fit_cost=evaluate(self,self.teacher,fit,include_evidence=include_evidence)
        with torch.no_grad():
            for target,value in zip(matrices,baseline):target.copy_(value)
        if self.loader.validation_access_images!=read_before or self.loader.allow_validation:
            raise ValueError('confirmation accessed before selection freeze')
        selected,decision=choose(baseline,proposed,proposal,fit_before,fit_after,grad_cost['weight_coverage'])
        self.event('teacher_selection_frozen',selection_round=round_no,accepted=decision['accepted'],
                   confirmation_reads_at_freeze=read_before)
        # Confirmation measures the frozen selection but cannot affect it.
        self.loader.allow_validation=True
        try:
            confirm_before,c1=evaluate(self,self.teacher,held,allowed_client_groups=self.heldout_groups,
                                       include_evidence=include_evidence)
            with torch.no_grad():
                for target,value in zip(matrices,selected):target.copy_(value)
            confirm_after,c2=evaluate(self,self.teacher,held,allowed_client_groups=self.heldout_groups,
                                      include_evidence=include_evidence)
        finally:
            self.loader.allow_validation=False
        if self.loader.validation_access_images-read_before!=472 or not same_rng(anchor_rng,rng_state()):
            raise ValueError('confirmation count or RNG restoration mismatch')
        if not decision['accepted'] and any(not torch.equal(v.detach().cpu(),b) for v,b in zip(matrices,baseline)):
            raise ValueError('fallback differs from FedAvg')
        self.s.setdefault('teacher_server_records',[]).append({'round':round_no,'teacher_variant':self.teacher_variant,
            'proposal':proposal,'decision':decision,'fit_baseline':fit_before,'fit_candidate':fit_after,
            'confirmation_baseline':confirm_before,'confirmation_selected':confirm_after,
            'gradient_cost':grad_cost,'fit_candidate_cost':fit_cost,'confirmation_baseline_cost':c1,'confirmation_selected_cost':c2,
            'confirmation_reads_before_selection':read_before,'confirmation_reads_after':self.loader.validation_access_images,
            'teacher_student_probe_seconds':time.time()-started,
            'teacher_model_logical_bytes':sum(v.numel()*v.element_size() for v in self.teacher.state_dict().values()),
            'network_measured_bytes':None})
