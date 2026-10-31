"""Unregistered current-input old-teacher probe; no real-data run or privacy claim."""
import torch
from torch.nn import functional as F
from fcl.checkpoint import restore_rng,rng_state
from fcl.math import mulora_ssb
from research.continual_muon.teacher_response_draft import old_response_loss


def probe(pilot,teacher,client_ids,*,include_local_polar=False,include_evidence=True):
    """Same-anchor old-response gradient, old proxy and current CE, client weighted.

    No stored targets or individual gradients leave this function. Restores RNG,
    parameters' existing grads, all module modes and all buffers on both models.
    """
    model=pilot.model
    if teacher is model:raise ValueError('teacher must be an independent snapshot')
    t=pilot.s['task'];c=pilot.cfg['cpt'];groups=pilot.manifest['partitions']['train'][t]
    if t<=0 or len(groups)!=len(client_ids):raise ValueError('invalid task or client count')
    if any(not set(ids).issubset(group) or len(set(ids))!=len(ids) for ids,group in zip(client_ids,groups)):
        raise ValueError('probe is not a unique subset of current training clients')
    weights=[len(g) for g in groups];denom=sum(w for w,ids in zip(weights,client_ids) if ids)
    if denom==0:raise ValueError('no eligible client')
    params=model.b_parameters();model_params=list(model.parameters());teacher_params=list(teacher.parameters())
    old_grads=[None if p.grad is None else p.grad.detach().clone() for p in model_params]
    teacher_grads=[None if p.grad is None else p.grad.detach().clone() for p in teacher_params]
    modes=[{m:m.training for m in mod.modules()} for mod in (model,teacher)]
    buffers=[{k:v.detach().clone() for k,v in mod.named_buffers()} for mod in (model,teacher)]
    random_state=rng_state();acc=[torch.zeros_like(p,dtype=torch.float64) for p in params]
    local_polar=[torch.zeros_like(p,dtype=torch.float64) for p in params] if include_local_polar else None
    totals={'teacher_response':0.,'relative_kl':0.,'evidence_shift_sq':0.,'ce_all':0.,'ce_current':0.}
    samples=0
    try:
        model.eval();teacher.eval()
        for k,ids in enumerate(client_ids):
            count=0;local=[torch.zeros_like(p,dtype=torch.float64) for p in params]
            for x,y in pilot.loader(pilot.raw[0],ids,pilot.cfg['batch']):
                pilot.budget();pilot.check_labels(y)
                features=pilot.test_tf(x.to(pilot.device));y=y.to(pilot.device)
                with torch.no_grad():teacher_old=teacher(features)
                student=model(features)
                if teacher_old.shape!=(len(y),t*c) or student.shape!=(len(y),(t+1)*c):
                    raise ValueError('teacher/student head dimension mismatch')
                loss,parts=old_response_loss(student[:,:t*c],teacher_old,include_evidence=include_evidence)
                a=F.cross_entropy(student.double(),y);b=F.cross_entropy(student[:,t*c:(t+1)*c].double(),y-t*c)
                if not all(torch.isfinite(v) for v in (loss,a,b)):raise ValueError('nonfinite probe loss')
                grad=torch.autograd.grad(loss,params)
                if any(not torch.isfinite(g).all() for g in grad):raise ValueError('nonfinite probe gradient')
                factor=weights[k]/denom*len(y)/len(ids)
                with torch.no_grad():
                    for dst,g in zip(local,grad):dst.add_(g.detach().double(),alpha=len(y)/len(ids))
                totals['teacher_response']+=factor*float(loss.detach())
                totals['relative_kl']+=factor*float(parts['relative_kl'])
                totals['evidence_shift_sq']+=factor*float(parts['evidence_shift_sq'])
                totals['ce_all']+=factor*float(a.detach());totals['ce_current']+=factor*float(b.detach())
                count+=len(y);samples+=len(y)
            if count!=len(ids):raise ValueError('client sample count mismatch')
            if count:
                with torch.no_grad():
                    for dst,g in zip(acc,local):dst.add_(g,alpha=weights[k]/denom)
                    if include_local_polar:
                        for dst,g in zip(local_polar,local):dst.add_(mulora_ssb(g.float()).double(),alpha=weights[k]/denom)
        receipt={'samples':samples,'weight_coverage':denom/sum(weights),
            'teacher_forward_images':samples,'student_forward_images':samples,'student_backward_images':samples,
            'local_ssb_calls':sum(bool(ids) for ids in client_ids)*len(params) if include_local_polar else 0}
        if include_local_polar:return [g.cpu() for g in acc],[g.cpu() for g in local_polar],totals,receipt
        return [g.cpu() for g in acc],totals,receipt
    finally:
        with torch.no_grad():
            for mod,old in zip((model,teacher),buffers):
                for name,value in mod.named_buffers():value.copy_(old[name])
        for p,g in zip(model_params,old_grads):p.grad=g
        for p,g in zip(teacher_params,teacher_grads):p.grad=g
        for mode in modes:
            for module,value in mode.items():module.training=value
        restore_rng(random_state)
