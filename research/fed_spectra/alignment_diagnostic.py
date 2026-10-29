"""Read-only endpoint probes on current training inputs, never test selection."""
import torch
from torch.nn import functional as F
from fcl.checkpoint import rng_state,restore_rng


def select_probes(groups,seed):
    chosen=[]
    for client,ids in enumerate(groups):
        ids=sorted(ids);g=torch.Generator().manual_seed(61029+1009*seed+31*client)
        order=torch.randperm(len(ids),generator=g).tolist()[:32]
        chosen.append([ids[i] for i in order])
    return chosen


def measure(p,probes,gradient=False,projectors=None):
    if projectors is not None and not gradient:raise ValueError('Projection requires gradients')
    projected=[torch.zeros(vh.shape[0],dtype=torch.float64) for u,vh in projectors] if projectors is not None else None
    state=rng_state();modes={m:m.training for m in p.model.modules()}
    p.model.eval();p.model.zero_grad(set_to_none=True)
    total=sum(len(p.ids('train',p.s['task'],k)) for k in range(p.cfg['clients']))
    rows=[];forward=teacher_forward=backward=0;calls=teacher_calls=0
    try:
        for k,ids in enumerate(probes):
            if not ids:continue
            if projectors is not None:p.model.zero_grad(set_to_none=True)
            weight=len(p.ids('train',p.s['task'],k))/total
            row={'client':k,'images':len(ids),'weight':weight,'ce':0.,'kl':0.,'accuracy':0.}
            for x,y in p.loader(p.raw[0],ids,p.cfg['batch']):
                p.budget();p.check_labels(y);x=p.test_tf(x.to(p.device));y=y.to(p.device)
                old=p.s['task']*p.cfg['cpt'];temperature=p.settings['anchor_temperature']
                with torch.no_grad():target=(p.get_teacher()(x)[:,:old]/temperature).softmax(-1) if old else None
                with torch.set_grad_enabled(gradient):
                    z=p.model(x);ce=F.cross_entropy(z[:,old:old+p.cfg['cpt']],y-old)
                    kl=F.kl_div((z[:,:old]/temperature).log_softmax(-1),target,reduction='batchmean')*temperature**2 if old else torch.zeros((),device=z.device)
                    loss=ce+p.settings['anchor_weight']*kl
                    if not torch.isfinite(loss):raise ArithmeticError('Nonfinite probe objective')
                    fraction=len(y)/len(ids)
                    if gradient:(loss*(fraction if projectors is not None else weight*fraction)).backward();backward+=len(y)
                row['ce']+=float(ce.detach())*fraction;row['kl']+=float(kl.detach())*fraction
                row['accuracy']+=float((z[:,old:old+p.cfg['cpt']].argmax(-1)==y-old).float().sum())/len(ids)
                forward+=len(y);teacher_forward+=len(y) if old else 0;calls+=1;teacher_calls+=int(old>0)
            if projectors is not None:
                for target,b,(u,vh) in zip(projected,p.model.b_parameters(),projectors):
                    target.add_((u.T@b.grad.detach().cpu().double()@vh.T).diag(),alpha=weight)
            row['loss']=row['ce']+p.settings['anchor_weight']*row['kl'];rows.append(row)
        totals={k:sum(v['weight']*v[k] for v in rows) for k in ['ce','kl','loss','accuracy']}
        grads=projected if projectors is not None else ([b.grad.detach().cpu().double().clone() for b in p.model.b_parameters()] if gradient else None)
        return {'clients':rows,'weighted':totals,'student_forward_images':forward,
                'teacher_forward_images':teacher_forward,'backward_images':backward,'student_forward_calls':calls,'teacher_forward_calls':teacher_calls},grads
    finally:
        p.model.zero_grad(set_to_none=True)
        for module,mode in modes.items():module.training=mode
        restore_rng(state)
