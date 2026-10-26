"""One real current-task batch, rolled back entirely before formal learning."""
import time
import torch
from torch import nn
from torch.nn import functional as F
from fcl.checkpoint import rng_state
from fcl.math import ClientMuon
from .math import metric


def real_batch(p):
    started=time.time();assert p.s['task']==1 and p.s['phase']=='task_start'
    original_rng=rng_state();c=p.cfg['cpt'];p.model.add_head(c)
    for a in p.model.adapters.values():a.start(a.A.detach().clone())
    saved={}
    for name,a in p.model.adapters.items():
        q,inv,_=metric(a.A,p.s['history'].raw[name].to(p.device))
        saved[name]={'Q':q.cpu(),'inverse':inv.cpu()}
    p.s['repair_metric']={'task':1,'layers':saved}
    x,y=next(iter(p.loader(p.raw[0],p.ids('train',1,0),p.cfg['batch'])))
    x=p.train_tf(x.to(p.device));y=y.to(p.device)
    old=[v.detach().clone() for v in p.model.rows[:-1]];bs=p.model.b_parameters();w=p.model.rows[-1]
    logits=p.model(x);ce=F.cross_entropy(logits[:,c:],y-c)
    grads=torch.autograd.grad(ce,bs+[w],retain_graph=True)
    old_path=torch.autograd.grad(logits[:,:c].square().mean(),bs)
    assert sum(float(v.abs().sum()) for v in old_path)>0, 'Old-head path detached'
    reference=nn.Parameter(w.detach().clone());reference.grad=grads[-1].clone()
    expected_head=torch.optim.AdamW([reference],lr=.001,weight_decay=0);expected_head.step()
    momentum=[torch.zeros_like(b) for b in bs];opt=ClientMuon(bs,momentum,[],[],False)
    headopt=torch.optim.AdamW([w],lr=.001,weight_decay=0)
    p._diagnostic={'steps':0,'images':0,'mask_images':0,'extra_backward_calls':0,'extra_backward_images':0,
                  'diagnostic_forward_images':0,'diagnostic_forward_calls':0,'momentum_help_ssb_hurt':0,
                  'triggers':0,'nonzero_repairs':0,'reasons':{},'sums':{},'maxima':{},'repair_step_wall_seconds':0.}
    p.repair_step(x,y,opt,headopt,p.cfg['lr'],True)
    assert torch.equal(w,reference), 'Head differs from CE-only AdamW'
    for m,g in zip(momentum,grads[:-1]):assert torch.equal(m,.1*g), 'EMA not updated once from CE'
    assert int(headopt.state[w]['step'])==1
    assert all(torch.equal(a,b) and not b.requires_grad for a,b in zip(old,p.model.rows[:-1]))
    result={'passed':True,'images':len(y),'forward_calls':4,'backward_calls':3+p._diagnostic['extra_backward_calls'],
            'old_logits_B_gradient_l1':sum(float(v.abs().sum()) for v in old_path),
            'head_CE_only_exact':True,'EMA_CE_only_once_exact':True,'old_heads_frozen':True,
            'diagnostics':p._diagnostic,'finite_step':p.s['finite_step_windows'][-1],
            'seconds':time.time()-started,'scope':'disposable real T2 batch with inherited T1 basis; no selected-A claim'}
    p.model.rows=nn.ParameterList();p.restore();p._metric_cache=None
    now=rng_state();assert torch.equal(now['torch'],original_rng['torch'])
    assert all(torch.equal(a,b) for a,b in zip(now['cuda'],original_rng['cuda']))
    result['full_checkpoint_and_RNG_restored']=True
    return result
