"""Unregistered old-model response proxy for current samples; no old-data guarantee."""
import torch
from torch.nn import functional as F


def old_response_loss(student_old,teacher_old,*,include_evidence=True):
    """KL of old-class distribution plus squared shift in old-class log evidence.

    Teacher scores are detached; both arrays must describe the SAME current inputs
    and exactly the same old class order. Absolute loss scale is a design choice.
    """
    if not isinstance(include_evidence,bool):raise ValueError('invalid evidence choice')
    if (student_old.ndim!=2 or teacher_old.shape!=student_old.shape or
            student_old.shape[0]==0 or student_old.shape[1]<2):
        raise ValueError('student/teacher old logits must match and be nonempty')
    if not torch.isfinite(student_old).all() or not torch.isfinite(teacher_old).all():
        raise ValueError('nonfinite old logits')
    teacher=teacher_old.detach().double();student=student_old.double()
    teacher_log=F.log_softmax(teacher,dim=-1);student_log=F.log_softmax(student,dim=-1)
    relative=(teacher_log.exp()*(teacher_log-student_log)).sum(dim=-1).mean()
    old_evidence_delta=torch.logsumexp(student,dim=-1)-torch.logsumexp(teacher,dim=-1)
    absolute=.5*old_evidence_delta.square().mean()
    return relative+(absolute if include_evidence else 0.), {'relative_kl':relative.detach(),'evidence_shift_sq':absolute.detach()}
