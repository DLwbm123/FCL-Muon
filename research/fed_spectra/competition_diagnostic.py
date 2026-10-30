"""Separate within-task fitting from cross-task competition on training probes."""
import torch
from torch.nn import functional as F
from fcl.checkpoint import rng_state,restore_rng


def competition_terms(z,y,old,cpt):
    if old<=0 or z.shape[1]!=old+cpt:raise ValueError('Requires all seen heads and at least one old task')
    current=z[:,old:old+cpt]
    ce=F.cross_entropy(current,y-old)
    all_ce=F.cross_entropy(z,y)
    competition=(torch.logsumexp(z,dim=1)-torch.logsumexp(current,dim=1)).mean()
    return ce,all_ce,competition


def measure_competition(p,probes,gradient=False):
    state=rng_state();modes={m:m.training for m in p.model.modules()}
    p.model.eval();p.model.zero_grad(set_to_none=True)
    total=sum(len(p.ids('train',p.s['task'],k)) for k in range(p.cfg['clients']))
    rows=[];images=calls=0
    try:
        for k,ids in enumerate(probes):
            if not ids:continue
            row={'client':k,'images':len(ids),'weight':len(p.ids('train',p.s['task'],k))/total}
            keys=['current_ce','all_seen_ce','competition_ce','current_accuracy','all_seen_accuracy','old_winner_fraction','correct_current_stolen_fraction','true_minus_max_old']
            row.update({key:0. for key in keys})
            for x,y in p.loader(p.raw[0],ids,p.cfg['batch']):
                p.budget();p.check_labels(y);x=p.test_tf(x.to(p.device));y=y.to(p.device)
                old=p.s['task']*p.cfg['cpt'];fraction=len(y)/len(ids)
                with torch.set_grad_enabled(gradient):
                    z=p.model(x);ce,all_ce,competition=competition_terms(z,y,old,p.cfg['cpt'])
                    if not torch.isfinite(competition):raise ArithmeticError('Nonfinite competition')
                    if gradient:(competition*row['weight']*fraction).backward()
                with torch.no_grad():
                    current_correct=z[:,old:].argmax(-1)==y-old
                    predicted=z.argmax(-1);old_winner=predicted<old
                    values=[ce,all_ce,competition,current_correct.float().mean(),(predicted==y).float().mean(),old_winner.float().mean(),(current_correct&old_winner).float().mean(),(z.gather(1,y[:,None]).squeeze(1)-z[:,:old].max(-1).values).mean()]
                    for key,value in zip(keys,values):row[key]+=float(value)*fraction
                images+=len(y);calls+=1
            if abs(row['all_seen_ce']-row['current_ce']-row['competition_ce'])>2e-6:raise ArithmeticError('CE decomposition mismatch')
            rows.append(row)
        grads=[b.grad.detach().cpu().double().clone() for b in p.model.b_parameters()] if gradient else None
        return {'clients':rows,'weighted':{key:sum(r['weight']*r[key] for r in rows) for key in keys},
                'student_forward_images':images,'student_backward_images':images if gradient else 0,'student_forward_calls':calls},grads
    finally:
        p.model.zero_grad(set_to_none=True)
        for module,mode in modes.items():module.training=mode
        restore_rng(state)


def gradient_alignment(base_gradient,competition_gradient,baseline,candidate):
    gg=sum(float(g.square().sum()) for g in base_gradient)
    hh=sum(float(h.square().sum()) for h in competition_gradient)
    return {'objective_competition_gradient_cosine':sum(float((g*h).sum()) for g,h in zip(base_gradient,competition_gradient))/max((gg*hh)**.5,1e-60),
            'competition_predicted_delta':sum(float((h*(v.double()-b.double())).sum()) for h,b,v in zip(competition_gradient,baseline,candidate))}
