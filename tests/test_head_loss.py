import copy
import torch
from research.head_regmean.head_loss import HeadLossPilot,separated_backward
from test_head_regmean_pilot import make


def test_extra_head_loss_does_not_change_direct_B_gradient(tmp_path):
    p=make(tmp_path/'base')
    while p.s['task']==0:p.step()
    p.step();x=p.raw[0].tensors[0][p.ids('train',1,0)];y=p.raw[0].tensors[1][p.ids('train',1,0)]
    grads=[]
    for mode in ['current','all_seen']:
        p.model.zero_grad(set_to_none=True);separated_backward(p.model,x,y,1,2,mode)
        assert p.model.rows[0].grad is None
        grads.append(([v.grad.clone() for v in p.model.b_parameters()],p.model.rows[-1].grad.clone()))
    assert all(torch.equal(a,b) for a,b in zip(grads[0][0],grads[1][0]))
    assert not torch.equal(grads[0][1],grads[1][1])


def test_separated_current_loss_matches_original_gradients(tmp_path):
    p=make(tmp_path/'base');p.step()
    x=p.raw[0].tensors[0][p.ids('train',0,0)];y=p.raw[0].tensors[1][p.ids('train',0,0)]
    p.model.zero_grad(set_to_none=True);p.backward_loss(x,y)
    expected=[v.grad.clone() for v in p.model.b_parameters()+[p.model.rows[-1]]]
    p.model.zero_grad(set_to_none=True);separated_backward(p.model,x,y,0,2,'current')
    for p,g in zip(p.model.b_parameters()+[p.model.rows[-1]],expected):torch.testing.assert_close(p.grad,g,rtol=0,atol=0)


def test_head_loss_old_head_frozen_through_full_lifecycle(tmp_path):
    p=make(tmp_path/'full');p.__class__=HeadLossPilot;p.cfg.update(aggregation='regmean',head_loss_mode='all_seen')
    while p.s['task']==0:p.step()
    old=p.model.rows[0].clone()
    p.run()
    assert torch.equal(old,p.model.rows[0])
    assert p.s['merges']==[0,1] and p.s['history'].committed=={0,1}
    assert all(v['deployed_fp32']['passed'] for v in p.s['certificates'])
