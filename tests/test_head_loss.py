import copy
import torch
from research.head_regmean.head_loss import HeadLossPilot,separated_backward,center_blocks
from test_head_regmean_pilot import make


def test_extra_head_loss_does_not_change_direct_B_gradient(tmp_path):
    p=make(tmp_path/'base')
    while p.s['task']==0:p.step()
    p.step();x=p.raw[0].tensors[0][p.ids('train',1,0)];y=p.raw[0].tensors[1][p.ids('train',1,0)]
    grads=[]
    for mode in ['current','all_seen','centered_all_seen']:
        p.model.zero_grad(set_to_none=True);separated_backward(p.model,x,y,1,2,mode)
        assert p.model.rows[0].grad is None
        grads.append(([v.grad.clone() for v in p.model.b_parameters()],p.model.rows[-1].grad.clone()))
    assert all(torch.equal(a,b) for a,b in zip(grads[0][0],grads[1][0]))
    assert not torch.equal(grads[0][1],grads[1][1])
    assert all(torch.equal(a,b) for a,b in zip(grads[0][0],grads[2][0]))
    assert not torch.equal(grads[0][1],grads[2][1])


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


def test_centered_loss_ignores_block_offsets_and_has_no_common_mode_gradient():
    logits=torch.tensor([[1.,2.,4.,-3.,1.,0.],[2.,0.,-2.,1.,-1.,3.]],dtype=torch.float64,requires_grad=True)
    labels=torch.tensor([3,5])
    offsets=torch.tensor([[13.,13.,13.,-7.,-7.,-7.],[-4.,-4.,-4.,8.,8.,8.]],dtype=torch.float64)
    a=torch.nn.functional.cross_entropy(center_blocks(logits,3),labels)
    b=torch.nn.functional.cross_entropy(center_blocks(logits+offsets,3),labels)
    torch.testing.assert_close(a,b,rtol=0,atol=1e-14)
    a.backward()
    torch.testing.assert_close(logits.grad.reshape(2,2,3).sum(-1),torch.zeros(2,2,dtype=torch.float64),rtol=0,atol=1e-14)


def test_centered_all_seen_old_head_freeze_and_lifecycle(tmp_path):
    p=make(tmp_path/'centered');p.__class__=HeadLossPilot
    p.cfg.update(aggregation='regmean',head_loss_mode='centered_all_seen')
    while p.s['task']==0:p.step()
    old=p.model.rows[0].clone();p.run()
    assert torch.equal(old,p.model.rows[0])
    assert p.s['merges']==[0,1] and p.s['history'].committed=={0,1}
    assert all(v['deployed_fp32']['passed'] for v in p.s['certificates'])
