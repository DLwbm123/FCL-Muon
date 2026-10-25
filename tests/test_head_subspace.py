import copy
import torch
from torch.nn import functional as F
from research.head_regmean.head_subspace import HeadSubspacePilot,head_bases,project_rows,cosine_leak,aggregate_subspace
from research.head_regmean.old_response import OldResponsePilot
from test_old_response import prepared


def test_projector_and_restricted_solver_ignore_basis_rotation():
    gen=torch.Generator().manual_seed(91)
    old=torch.randn(2,7,generator=gen);qo,qn=head_bases(old)
    torch.testing.assert_close(qo.T@qn,torch.zeros(2,5,dtype=torch.float64),atol=1e-12,rtol=0)
    heads=[project_rows(torch.randn(3,7,generator=gen),qo,True) for _ in range(3)]
    grams=[]
    for n in [8,11,13]:
        z=F.normalize(torch.randn(n,7,generator=gen),dim=1).double();grams.append(z.T@z)
    head,raw,info=aggregate_subspace(heads,grams,[8,11,13],qo,qn)
    assert info['deployed_fp32']['passed'] and cosine_leak(head,qo)<=2e-6
    rotation,_=torch.linalg.qr(torch.randn(5,5,generator=gen,dtype=torch.float64))
    head2,_,info2=aggregate_subspace(heads,grams,[8,11,13],qo,qn@rotation)
    torch.testing.assert_close(head,head2,atol=2e-6,rtol=2e-6)
    assert abs(info['regmean_objective']-info2['regmean_objective'])<1e-6
    torch.testing.assert_close(head.norm(dim=1),raw.norm(dim=1),atol=2e-6,rtol=2e-6)


def test_projected_head_gradient_and_B_separation(tmp_path):
    p=prepared(tmp_path/'gradient');HeadSubspacePilot.setup_subspace(p)
    x=p.raw[0].tensors[0][p.ids('train',1,0)];y=p.raw[0].tensors[1][p.ids('train',1,0)]
    p.model.zero_grad(set_to_none=True);p.backward_loss(x,y)
    expected_B=[v.grad.clone() for v in p.model.b_parameters()]
    expected_head=project_rows(p.model.rows[-1].grad,p.old_basis)
    p.model.zero_grad(set_to_none=True);p.__class__=HeadSubspacePilot;p.backward_loss(x,y)
    assert all(torch.equal(v.grad,g) for v,g in zip(p.model.b_parameters(),expected_B))
    assert torch.equal(p.model.rows[-1].grad,expected_head)
    assert p.model.rows[0].grad is None and all(v.grad is None for v in p.teacher.parameters())


def test_projected_lifecycle_and_checkpoint_restore(tmp_path):
    p=prepared(tmp_path/'full');p.__class__=HeadSubspacePilot;p.setup_subspace()
    old=p.model.rows[0].clone();teacher=copy.deepcopy(p.teacher.state_dict());nr=len(p.s['training_records'])
    p.run();p.restore()
    assert torch.equal(old,p.model.rows[0])
    assert all(torch.equal(v,p.teacher.state_dict()[k]) for k,v in teacher.items())
    assert p.s['merges']==[0,1] and p.s['history'].committed=={0,1}
    assert p.s['preservation_training']['visits']==16
    assert cosine_leak(p.model.rows[-1],p.old_basis)<=2e-6
    assert all(v['head_old_span_cosine_leak_max']<=2e-6 for v in p.s['training_records'][nr:])
    assert all(v['deployed_fp32']['passed'] for v in p.s['certificates'])
    assert all(v['old_span_cosine_leak_max']<=2e-6 for v in p.s['certificates'] if v['task']==1)
