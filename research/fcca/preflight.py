"""Two disposable real train-batch forwards; restore the complete T1 prefix."""
import time
import torch
from torch import nn
from fcl.checkpoint import rng_state
from .math import directions
from .pilot import probe_context


def real_batch(p):
    start=time.time();assert p.s['task']==1 and p.s['phase']=='task_start'
    original_rng=rng_state();p.model.add_head(p.cfg['cpt'])
    for a in p.model.adapters.values():a.start(a.A.detach().clone())
    ids=p.ids('train',1,0)[:16]
    p.s['fcca_probe_ids']=[(ids,[])]+[([],[]) for _ in range(p.cfg['clients']-1)]
    bs=p.model.b_parameters();before={n:v.detach().clone() for n,v in p.model.state_dict().items()}
    endpoints=[[b.detach().cpu().clone() for b in bs] for _ in range(2)]
    # A deterministic test direction does not consume RNG and is never deployed.
    endpoints[0][0].fill_(.001);endpoints[1][0].fill_(-.001)
    base=[b.detach().cpu().clone() for b in bs];ds=directions(endpoints,base,base,[.5,.5])
    fit=p.measure(0,ds)
    assert fit['probe_images']==len(ids) and fit['backward_calls']>0
    assert all(torch.equal(v,p.model.state_dict()[n]) for n,v in before.items()),'Probe mutated weights/buffers'
    assert all(not w.requires_grad for w in p.model.rows[:-1])
    with probe_context(p.model):
        x,y=next(iter(p.loader(p.raw[0],ids,p.cfg['batch'])))
        z=p.model(p.test_tf(x.to(p.device)))
        old_gradient=torch.autograd.grad(z[:,:p.cfg['cpt']].square().mean(),bs)
        old_l1=sum(float(g.abs().sum()) for g in old_gradient)
        assert old_l1>0,'Old head path detached'
    result={'passed':True,'images_per_forward':len(ids),'forward_calls':fit['forward_calls']+1,
            'backward_calls':fit['backward_calls']+1,'old_logits_B_gradient_l1':old_l1,
            'fixed_heads_unchanged':True,'probe_state_restored':True,'scope':'disposable T2 training batch, inherited T1 basis; no optimizer step',
            'seconds':time.time()-start}
    p.model.rows=nn.ParameterList();p.restore()
    now=rng_state();assert torch.equal(now['torch'],original_rng['torch'])
    assert all(torch.equal(a,b) for a,b in zip(now['cuda'],original_rng['cuda']))
    result['full_checkpoint_and_RNG_restored']=True
    return result
