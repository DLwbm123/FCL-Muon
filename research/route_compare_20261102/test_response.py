import copy
import math
import pytest
import torch
from torch.nn import functional as F
from fcl.math import ClientMuon
from research.head_regmean.solver import aggregate_heads
from research.adamw_combined.test_aggregated_response_draft import Toy
from research.route_compare_20261102.response import federated_response


def oracle(model, support, grams, query, sizes, formal_batches, optimizer):
    start = copy.deepcopy(model)
    global_b = [p.detach().clone() for p in start.b_parameters()]
    head = start.rows[-1].detach().clone()
    moment = [[torch.zeros_like(p) for p in global_b] for _ in sizes]
    for r, clients in enumerate(support):
        endpoints, heads, rawgrams = [], [], []
        for k, batches in enumerate(clients):
            local = copy.deepcopy(start)
            with torch.no_grad():
                for p, v in zip(local.b_parameters(), global_b): p.copy_(v)
                local.rows[-1].copy_(head)
            bs = local.b_parameters()
            opt = (ClientMuon(bs, moment[k], [], [], False) if optimizer == 'muon'
                   else torch.optim.AdamW(bs, lr=0., weight_decay=0.))
            headopt = torch.optim.AdamW([local.rows[-1]], lr=.001, weight_decay=0.)
            steps_formal = 2 * formal_batches[k]
            for step, (x, y) in enumerate(batches):
                local.zero_grad(set_to_none=True)
                loss = F.cross_entropy(local(x), y)
                loss.backward()
                rate = .01 * .5 * (1+math.cos(math.pi*(r*steps_formal+step)/(5*steps_formal-1)))
                if optimizer == 'muon':
                    opt.step(rate)
                else:
                    for group in opt.param_groups: group['lr'] = rate
                    opt.step()
                headopt.step()
            with torch.no_grad():
                endpoints.append([p.detach().clone() for p in bs])
                heads.append(local.rows[-1].detach().clone())
                zs = [F.normalize(local.backbone.forward_features(x)[:, 0], dim=1).double() for x,y in grams[k]]
                rawgrams.append(sum(z.T @ z for z in zs) * (sizes[k]/sum(len(z) for z in zs)))
        global_b = [sum(v[j] * (n/sum(sizes)) for v,n in zip(endpoints,sizes)) for j in range(len(global_b))]
        head, _, _ = aggregate_heads(heads, rawgrams, sizes, True)
    with torch.no_grad():
        for p,v in zip(start.b_parameters(),global_b): p.copy_(v)
        start.rows[-1].copy_(head)
        loss = sum(F.cross_entropy(start(x), y) * n/sum(sizes) for [(x,y)],n in zip(query,sizes))
    return global_b, head, loss, moment


@pytest.mark.parametrize('optimizer', ['adamw', 'muon'])
def test_two_round_response_matches_production_and_supports_outer_gradient(optimizer):
    torch.set_num_threads(2)
    torch.manual_seed(11)
    model = Toy().eval()
    original = {k: v.clone() for k,v in model.state_dict().items()}
    x1=torch.tensor([[1., .3], [.2, 1.]])
    x2=torch.tensor([[.7, 1.], [-.4, 1.]])
    y1=torch.tensor([0, 1]);y2=torch.tensor([1, 0])
    support=[[[ (x1,y1) for _ in range(4)],[(x2,y2) for _ in range(4)]] for _ in range(2)]
    grams=[[(x1,y1)],[(x2,y2)]]
    query=[[(x1,y1)],[(x2,y2)]]
    basis={'0':model.adapters['0'].A.clone().requires_grad_(True)}
    out=federated_response(model,basis,support,grams,query,[3,5],[2,3],optimizer,.01,5,2,0,2)
    b,h,loss,m=oracle(model,support,grams,query,[3,5],[2,3],optimizer)
    torch.testing.assert_close(out['B'][0],b[0],rtol=2e-5,atol=2e-7)
    torch.testing.assert_close(out['head'],h,rtol=2e-5,atol=2e-7)
    torch.testing.assert_close(out['loss'],loss,rtol=2e-5,atol=2e-7)
    if optimizer=='muon':
        for actual,expected in zip(out['muon_states'],m):
            torch.testing.assert_close(actual[0],expected[0],rtol=2e-5,atol=2e-7)
    grad=torch.autograd.grad(out['loss'],basis['0'])[0]
    assert torch.isfinite(grad).all()
    direction=torch.tensor([[.3,-.2],[.4,.1]])
    def evaluated(value):
        return federated_response(model,{'0':value},support,grams,query,[3,5],[2,3],
                                  optimizer,.01,5,2,0,2,use_checkpoint=False)['loss'].detach()
    eps=.01
    finite=(evaluated(basis['0'].detach()+eps*direction)-evaluated(basis['0'].detach()-eps*direction))/(2*eps)
    torch.testing.assert_close((grad*direction).sum(),finite,rtol=0,atol=2e-5)
    assert out['counts']=={'trial_visits':32,'trial_steps':16,'gram_visits':8,'query_visits':4,'head_fusions':2}
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in original.items())


@pytest.mark.parametrize('optimizer', ['adamw', 'muon'])
def test_graph_to_production_qv_response_directional_difference(optimizer):
    from fcl.model import QVAdapter
    from research.route_compare_20261102.frame import graph_basis
    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.adapter=QVAdapter(torch.nn.Linear(4,12),2)
            self.adapter.start(torch.eye(4)[:2])
        def forward_features(self,x):
            q,_,v=self.adapter(x).chunk(3,-1)
            return (q+v).unsqueeze(1)
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__();self.backbone=Backbone()
            self.adapters={'0':self.backbone.adapter}
            self.rows=torch.nn.ParameterList([torch.nn.Parameter(torch.randn(2,4),requires_grad=False),
                                               torch.nn.Parameter(torch.randn(2,4))])
        def b_parameters(self): return [self.backbone.adapter.Bq,self.backbone.adapter.Bv]
        def forward(self,x):
            z=F.normalize(self.backbone.forward_features(x)[:,0],dim=1)
            return z @ F.normalize(torch.cat(list(self.rows)),dim=1).T
    torch.manual_seed(11);model=Net().eval();e=torch.eye(4,dtype=torch.float64)[:,:3]
    z0=torch.eye(3,dtype=torch.float64)[:,:2];n=torch.eye(3,dtype=torch.float64)[:,2:]
    xs=[torch.randn(2,4) for _ in range(2)];ys=[torch.tensor([2,3]),torch.tensor([3,2])]
    support=[[[ (xs[k],ys[k]) for _ in range(4)] for k in range(2)] for _ in range(2)]
    gram=[[(xs[k],ys[k])] for k in range(2)]
    query=[[(torch.randn(2,4),torch.tensor([2,3]))] for _ in range(2)]
    direction=torch.tensor([[.3,-.2]],dtype=torch.float64)
    def objective(x):
        a=graph_basis(e,z0,n,x).float()
        return federated_response(model,{'0':a},support,gram,query,[3,5],[2,3],optimizer,
                                  .001,5,2,1,2)['loss']
    x=torch.zeros((1,2),dtype=torch.float64,requires_grad=True)
    value=objective(x); g=torch.autograd.grad(value,x)[0]
    assert torch.isfinite(g).all()
    step=.01
    finite=(objective(x.detach()+step*direction)-objective(x.detach()-step*direction))/(2*step)
    torch.testing.assert_close((g*direction).sum(),finite.double(),rtol=0,atol=5e-4)
