import torch
import numpy as np
from torch.nn import functional as F
from fcl.checkpoint import atomic_save, rng_state
from research.basic_repair.fcl_basic_repair_tools import self_test
from scripts.basic_repair_heads import fit_global_cosine


def test_reference_statistics_and_centering():
    assert self_test()['self_test']=='passed'


def test_client_gradient_sum_matches_one_pooled_head_update():
    torch.manual_seed(91)
    z=F.normalize(torch.randn(23,7),dim=1);y=torch.arange(23)%3
    initial=torch.randn(3,7)
    actual,_=fit_global_cosine([(z[:2],y[:2]),(z[2:9],y[2:9]),(z[9:],y[9:])],initial,3,.001)
    w=torch.nn.Parameter(initial.clone());opt=torch.optim.AdamW([w],lr=.001,weight_decay=0)
    for _ in range(3):
        opt.zero_grad(set_to_none=True);F.cross_entropy(16*z@F.normalize(w,dim=1).T,y).backward();opt.step()
    torch.testing.assert_close(actual,w,atol=2e-7,rtol=1e-6)


def test_endpoint_serialization_does_not_advance_rng(tmp_path):
    torch.manual_seed(92);np.random.seed(92)
    b=torch.randn(7,3);head=torch.randn(10,7)
    before=rng_state()
    atomic_save({'B':[b.detach().cpu().clone()],'head':head.detach().cpu().clone()},tmp_path/'endpoint.pt')
    after=rng_state()
    torch.testing.assert_close(before['torch'],after['torch'],atol=0,rtol=0)
    assert np.array_equal(before['numpy'][1],after['numpy'][1])
