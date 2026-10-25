import torch
import pytest
from torch.nn import functional as F
from research.head_regmean.solver import aggregate_heads, sphere_solve, objective, shrink_gram


def test_linear_matches_direct_sample_regression():
    torch.manual_seed(62)
    xs=[torch.randn(n,5,dtype=torch.float64) for n in [7,19,31]]
    ws=[torch.randn(3,5,dtype=torch.float64) for _ in xs]
    result,_,info=aggregate_heads(ws,[x.T@x for x in xs],list(map(len,xs)),False,alpha=1)
    x=torch.cat(xs);target=torch.cat([x@w.T for x,w in zip(xs,ws)])
    expected=torch.linalg.lstsq(x,target).solution.T
    torch.testing.assert_close(result.double(),expected,rtol=1e-6,atol=1e-7)
    assert info['deployed_fp32']['passed']


def test_single_client_cosine_identity():
    torch.manual_seed(19);w=torch.randn(4,9);x=torch.randn(35,9)
    result,_,info=aggregate_heads([w],[x.double().T@x.double()],[35],True)
    torch.testing.assert_close(F.normalize(result,dim=1),F.normalize(w,dim=1),rtol=1e-6,atol=2e-7)
    assert info['fp64']['passed'] and info['deployed_fp32']['passed']


def test_cosine_effective_solution_invariant_to_positive_local_row_scaling():
    torch.manual_seed(18);ws=[torch.randn(3,6) for _ in range(2)]
    xs=[torch.randn(30,6,dtype=torch.float64) for _ in ws];gs=[x.T@x for x in xs]
    first,_,_=aggregate_heads(ws,gs,[30,30],True)
    second,_,_=aggregate_heads([ws[0]*torch.tensor([.2,2.,4.])[:,None],ws[1]],gs,[30,30],True)
    torch.testing.assert_close(F.normalize(first,dim=1),F.normalize(second,dim=1),rtol=2e-6,atol=2e-7)


def test_sphere_nontrivial_against_dense_circle():
    h=torch.diag(torch.tensor([.3,3.],dtype=torch.float64));q=torch.tensor([[.1,.7]],dtype=torch.float64)
    u,_,_,info=sphere_solve(h,q)
    theta=torch.linspace(0,2*torch.pi,20001,dtype=torch.float64)
    circle=torch.stack([theta.cos(),theta.sin()],1)
    assert float(objective(u,h,q))<=float(objective(circle,h,q).min())+1e-9
    naive=F.normalize(torch.linalg.solve(h,q.T).T,dim=1)
    assert float(objective(u,h,q))<float(objective(naive,h,q))-1e-4


@pytest.mark.parametrize('q',[[0.,0.,0.],[0.,.5,0.]])
def test_sphere_hard_case(q):
    h=torch.diag(torch.tensor([0.,1.,2.],dtype=torch.float64))
    u,lam,_,info=sphere_solve(h,torch.tensor([q],dtype=torch.float64))
    assert info['cases']==['hard'] and info['fp64']['passed']
    torch.testing.assert_close(u.norm(dim=1),torch.ones(1,dtype=torch.float64))


def test_sphere_zero_gram():
    u,_,_,info=sphere_solve(torch.zeros(3,3),torch.zeros(2,3))
    assert info['fp64']['passed'] and torch.isfinite(u).all()


def test_sphere_orthogonal_rotation_invariance():
    torch.manual_seed(14);x=torch.randn(14,6,dtype=torch.float64);h=x.T@x;q=torch.randn(4,6,dtype=torch.float64)
    v,_=torch.linalg.qr(torch.randn(6,6,dtype=torch.float64))
    u,*_=sphere_solve(h,q);rot,*_=sphere_solve(v.T@h@v,q@v)
    torch.testing.assert_close(rot,u@v,rtol=1e-8,atol=1e-8)


def test_raw_moments_add_without_double_sample_weighting():
    torch.manual_seed(16);w=torch.randn(3,7);x=torch.randn(30,7,dtype=torch.float64)
    whole,_,_=aggregate_heads([w],[x.T@x],[30],True)
    split,_,_=aggregate_heads([w,w],[x[:2].T@x[:2],x[2:].T@x[2:]],[2,28],True)
    torch.testing.assert_close(whole,split,atol=2e-7,rtol=1e-6)


def test_real_shape_fp32_deployment_and_psd_shrink():
    torch.manual_seed(23)
    ws=[torch.randn(10,768)*.02 for _ in range(3)]
    xs=[F.normalize(torch.randn(n,768),dim=1).double() for n in [6,40,200]]
    gs=[x.T@x for x in xs]
    result,_,info=aggregate_heads(ws,gs,list(map(len,xs)),True)
    assert result.dtype==torch.float32 and info['deployed_fp32']['passed']
    assert info['regmean_objective']<=info['baseline_objective']
    assert torch.linalg.eigvalsh(shrink_gram(gs[0])).min()>0
