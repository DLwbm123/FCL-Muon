import copy
import torch
from research.head_regmean.head_loss import separated_backward
from research.head_regmean.feature_anchor import FeatureAnchorPilot,feature_anchor_mse
from research.head_regmean.old_response import old_response_mse
from test_head_regmean_pilot import make


def prepared(path):
    p=make(path)
    while p.s['task']==0:p.step()
    p.step();p.__class__=FeatureAnchorPilot
    p.cfg.update(aggregation='regmean',head_loss_mode='centered_all_seen',feature_anchor_coefficient=1.)
    p.teacher=copy.deepcopy(p.model).eval().requires_grad_(False)
    return p


def test_feature_anchor_zero_start_and_head_route_unchanged(tmp_path):
    p=prepared(tmp_path/'zero');x=p.raw[0].tensors[0][p.ids('train',1,0)];y=p.raw[0].tensors[1][p.ids('train',1,0)]
    p.model.zero_grad(set_to_none=True)
    separated_backward(p.model,x,y,1,2,'centered_all_seen');expected=[v.grad.clone() for v in p.model.b_parameters()+[p.model.rows[-1]]]
    p.model.zero_grad(set_to_none=True);p.backward_loss(x,y)
    assert p.s['preservation_training']['mse_sum']==0
    assert all(torch.equal(v.grad,g) for v,g in zip(p.model.b_parameters()+[p.model.rows[-1]],expected))
    with torch.no_grad():p.model.b_parameters()[1].add_(.01*torch.arange(10).reshape(5,2))
    p.model.zero_grad(set_to_none=True);separated_backward(p.model,x,y,1,2,'centered_all_seen')
    grads=[v.grad.clone() for v in p.model.b_parameters()];h=p.model.rows[-1].grad.clone()
    p.model.zero_grad(set_to_none=True);p.backward_loss(x,y)
    assert p.s['preservation_training']['mse_sum']>0
    assert any(not torch.equal(v.grad,g) for v,g in zip(p.model.b_parameters(),grads))
    assert torch.equal(p.model.rows[-1].grad,h)
    assert all(v.grad is None for v in p.teacher.parameters()) and p.model.rows[0].grad is None


def test_feature_loss_detects_old_response_nullspace_and_is_rotation_invariant():
    teacher=torch.tensor([[0.,1.,0.]],dtype=torch.float64,requires_grad=True)
    student=torch.tensor([[0.,0.,1.]],dtype=torch.float64,requires_grad=True)
    heads=torch.tensor([[1.,0.,0.],[-1.,0.,0.]],dtype=torch.float64)
    assert old_response_mse(student@heads.T,teacher@heads.T,2)==0
    a=feature_anchor_mse(student,teacher,16.)
    assert a>0
    q,_=torch.linalg.qr(torch.tensor([[1.,2.,0.],[0.,1.,2.],[2.,0.,1.]],dtype=torch.float64))
    assert torch.allclose(a,feature_anchor_mse(student@q,teacher@q,16.),atol=1e-12,rtol=1e-12)
    a.backward();assert teacher.grad is None and student.grad.abs().sum()>0


def test_teacher_old_head_and_full_lifecycle(tmp_path):
    p=prepared(tmp_path/'full');teacher=copy.deepcopy(p.teacher.state_dict());old=p.model.rows[0].clone()
    p.run()
    assert all(torch.equal(v,p.teacher.state_dict()[k]) for k,v in teacher.items())
    assert torch.equal(old,p.model.rows[0]);assert p.s['preservation_training']['visits']==16
    assert p.s['preservation_training']['positive_mse_batches']>0
    assert p.s['merges']==[0,1] and p.s['history'].committed=={0,1}
    assert all(v['deployed_fp32']['passed'] for v in p.s['certificates'])
