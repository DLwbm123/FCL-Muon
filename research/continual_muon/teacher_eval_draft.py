"""Current-task, two-model response scoring without optimizer or test access."""
import math
import torch
from torch.nn import functional as F
from fcl.checkpoint import restore_rng,rng_state
from research.continual_muon.teacher_response_draft import old_response_loss


def evaluate(pilot,teacher,client_ids,*,allowed_client_groups=None,include_evidence=True):
    model=pilot.model;t=pilot.s['task'];c=pilot.cfg['cpt'];groups=pilot.manifest['partitions']['train'][t]
    if teacher is model or t<=0 or len(groups)!=len(client_ids):raise ValueError('invalid teacher or split')
    allowed=groups if allowed_client_groups is None else allowed_client_groups
    if len(allowed)!=len(groups):raise ValueError('invalid allowed groups')
    if allowed_client_groups is not None and set(i for g in allowed for i in g)&set(i for g in groups for i in g):
        raise ValueError('confirmation groups overlap training')
    if any(not set(ids).issubset(group) or len(set(ids))!=len(ids) for ids,group in zip(client_ids,allowed)):
        raise ValueError('evaluation is not a subset of allowed client data')
    weights=[len(g) for g in groups];denom=sum(w for w,ids in zip(weights,client_ids) if ids)
    if denom==0:raise ValueError('no eligible client')
    models=(model,teacher);modes=[{m:m.training for m in mod.modules()} for mod in models]
    buffers=[{k:v.detach().clone() for k,v in mod.named_buffers()} for mod in models]
    state=rng_state();metrics={'teacher_response':0.,'relative_kl':0.,'evidence_shift_sq':0.,'ce_all':0.,'ce_current':0.};samples=0
    try:
        model.eval();teacher.eval()
        with torch.no_grad():
            for k,ids in enumerate(client_ids):
                count=0
                for x,y in pilot.loader(pilot.raw[0],ids,pilot.cfg['batch']):
                    pilot.budget();pilot.check_labels(y);features=pilot.test_tf(x.to(pilot.device));y=y.to(pilot.device)
                    teacher_old=teacher(features);student=model(features)
                    if teacher_old.shape!=(len(y),t*c) or student.shape!=(len(y),(t+1)*c):
                        raise ValueError('teacher/student head dimension mismatch')
                    loss,parts=old_response_loss(student[:,:t*c],teacher_old,include_evidence=include_evidence)
                    a=F.cross_entropy(student.double(),y);b=F.cross_entropy(student[:,t*c:(t+1)*c].double(),y-t*c)
                    values={'teacher_response':loss,'relative_kl':parts['relative_kl'],'evidence_shift_sq':parts['evidence_shift_sq'],
                            'ce_all':a,'ce_current':b}
                    if any(not torch.isfinite(v) for v in values.values()):raise ValueError('nonfinite evaluation')
                    factor=weights[k]/denom*len(y)/len(ids)
                    for name,value in values.items():metrics[name]+=factor*float(value)
                    samples+=len(y);count+=len(y)
                if count!=len(ids):raise ValueError('client sample count mismatch')
        return metrics,{'samples':samples,'weight_coverage':denom/sum(weights),
                        'student_forward_images':samples,'teacher_forward_images':samples,'student_backward_images':0}
    finally:
        with torch.no_grad():
            for mod,old in zip(models,buffers):
                for name,value in mod.named_buffers():value.copy_(old[name])
        for mode in modes:
            for module,value in mode.items():module.training=value
        restore_rng(state)


def choose(baselines,candidates,proposal,baseline_metrics,candidate_metrics,coverage):
    """Fixed training-only selection; real old-class safety is not guaranteed."""
    if not baselines or len(baselines)!=len(candidates):raise ValueError('invalid candidate count')
    if any(a.shape!=b.shape or a.dtype!=b.dtype or not torch.isfinite(a).all() or not torch.isfinite(b).all()
           for a,b in zip(baselines,candidates)):raise ValueError('invalid candidate')
    keys=('teacher_response','ce_all','ce_current')
    if any(k not in m or not math.isfinite(m[k]) for m in (baseline_metrics,candidate_metrics) for k in keys):
        raise ValueError('invalid evaluation')
    if not math.isfinite(coverage) or not 0<=coverage<=1:raise ValueError('invalid coverage')
    checks={'candidate_proposed':bool(proposal['proposed']), 'coverage':bool(coverage>=.95),
            'teacher_improved':bool(candidate_metrics['teacher_response']<baseline_metrics['teacher_response']-1e-8),
            'all_not_worse':bool(candidate_metrics['ce_all']<=baseline_metrics['ce_all']+1e-6),
            'current_not_worse':bool(candidate_metrics['ce_current']<=baseline_metrics['ce_current']+1e-6)}
    accepted=all(checks.values())
    return [v.clone() for v in (candidates if accepted else baselines)],{'accepted':accepted,'checks':checks}
