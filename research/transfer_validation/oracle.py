"""Evaluator-only diagnostics from the saved T10 shared endpoint."""

import os

import torch
from torch.nn import functional as F

from research.adamw_t10.storage import write
from research.stat_ablation.runtime import setup,build,validate
from .frozen_stage import restore_model


def main():
    setup()
    root,freeze=validate()
    pilot=build({k:freeze[k] for k in ('started','deadline')})
    state=restore_model(pilot,os.environ['INPUT_CHECKPOINT'])
    task=pilot.cfg['tasks']
    cpt=pilot.cfg['cpt']
    if state['task']!=task or state['phase']!='complete':
        raise ValueError('Expected a completed shared trajectory')
    pilot.s=state
    state['task']=task-1
    z,y=pilot.evaluation_features()
    old_limit=(task-1)*cpt
    old=y<old_limit
    z=z.cuda()
    y=y.cuda()
    old=old.cuda()
    stats=state['stat_branches']
    simple,low_rank='translation','mean_only'
    if not torch.equal(stats[simple]['scatter'],stats[low_rank]['scatter']):
        raise ValueError('M and S do not share covariance')
    cov=stats[simple]['scatter'].cuda()/stats[simple]['counts'].sum()
    chol=torch.linalg.cholesky(cov+(.1*cov.trace()/len(cov)+1e-6)*torch.eye(len(cov),device='cuda',dtype=cov.dtype))
    test_means=torch.stack([z[y==i].double().mean(0) for i in range(old_limit)])
    geometry={}
    for mode in (simple,low_rank):
        delta=stats[mode]['means'][:old_limit].cuda()-test_means
        weighted=torch.linalg.solve_triangular(chol,delta.T,upper=False).square().sum(0)
        l2=delta.norm(dim=1)
        geometry[mode]=[
            dict(first_task=i+1, age=task-i-1,l2=float(a.mean()),mahalanobis_squared=float(b.mean()))
            for i,(a,b) in enumerate(zip(l2.split(cpt),weighted.split(cpt)))]
    heads={m:tuple(t.float().cuda() for t in state['branch_heads'][m]) for m in (simple,low_rank)}
    scores={}
    pairs={}
    for weights in heads:
        for intercept in heads:
            W=heads[weights][0]
            b=heads[intercept][1]
            logits=F.linear(z,W,b)
            yhat=logits.argmax(1)
            pred=yhat==y
            rivals=logits.clone()
            rivals.scatter_(1,y[:,None],float('-inf'))
            margin=logits.gather(1,y[:,None]).squeeze(1)-rivals.max(1).values
            scores[weights+'__'+intercept]=dict(
                correct=int(pred.sum()), old_correct=int((pred&old).sum()),new_correct=int((pred&~old).sum()),
                old_to_new=int((old&(yhat>=old_limit)).sum()),
                old_to_other_old=int((old&~pred&(yhat<old_limit)).sum()),
                new_to_old=int((~old&(yhat<old_limit)).sum()),
                mean_old_margin=float(margin[old].mean()),mean_new_margin=float(margin[~old].mean()))
            if weights==intercept:
                pairs[weights]=(pred,yhat)
                expected=state['stages'][-1]['branch_readouts'][weights]
                c=task*cpt
                cm=torch.bincount(y*c+yhat,minlength=c*c).reshape(c,c).cpu().tolist()
                if cm!=expected['confusion']:
                    raise ValueError('Official T10 raw confusion did not reproduce')
    flips={}
    sm,sp=pairs[simple]
    mm,mp=pairs[low_rank]
    groups={'old':old,'new':~old}
    for name,mask in groups.items():
        correct=~sm&mm&mask
        harmed=sm&~mm&mask
        flips[name]=dict(
            samples=int(mask.sum()),simple_wrong_lowrank_right=int(correct.sum()),
            simple_right_lowrank_wrong=int(harmed.sum()),
            corrected_simple_prediction_new=int((correct&(sp>=old_limit)).sum()),
            harmed_lowrank_prediction_new=int((harmed&(mp>=old_limit)).sum()))
    write(root,root/'src/reports/COMPLETE.json',dict(
        status='completed',task=task,geometry_by_class_age=geometry,head_mix=scores,paired_flips=flips,
        old_prototypes='cumulative training statistics versus same-cohort test means; includes sampling difference',
        conditional_head_swap='diagnostic only; no test-based selection',
        same_covariance_exact=True,official_confusions_reproduced=True))
    print('Final endpoint diagnostic completed',flush=True)


if __name__=='__main__':
    main()
