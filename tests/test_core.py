import copy
import random
import numpy as np
import pytest
import torch
from fcl.math import rrs,History,aggregate,muon_operator,compact_polar,ClientMuon
from fcl.model import QVAdapter
from fcl.checkpoint import atomic_save,rng_state,restore_rng


def test_statistics_rrs_and_commit():
    torch.manual_seed(13)
    xs=[torch.randn(7,9,dtype=torch.float64),torch.randn(23,9,dtype=torch.float64)]
    s=sum(x.T@x for x in xs);x=torch.cat(xs)
    torch.testing.assert_close(s,x.T@x)
    h=torch.randn(9,9,dtype=torch.float64);h=h.T@h
    a,details=rrs(s,h,4)
    torch.testing.assert_close(a@a.T,torch.eye(4),atol=3e-6,rtol=1e-6)
    metric=h+details['jitter']*torch.eye(9,dtype=torch.float64)
    # Selected original-space subspace must be invariant under H^-1 S.
    q=a.T.double();action=torch.linalg.solve(metric,s@q)
    torch.testing.assert_close(action,q@(q.T@action),atol=1e-5,rtol=1e-5)
    hist=History();hist.commit(0,{'x':s})
    with pytest.raises(ValueError):hist.commit(0,{'x':s})
    torch.testing.assert_close(hist.raw['x'],s)


def test_spectrum_aggregation_and_mismatch():
    torch.manual_seed(1);a=torch.linalg.qr(torch.randn(8,3)).Q.T
    bs=[torch.randn(6,3),torch.randn(6,3)];w=[.2,.8]
    b=aggregate(bs,[a,a.clone()],w)
    torch.testing.assert_close(b@a,sum(p*(x@a) for p,x in zip(w,bs)))
    for x in [b,bs[1]-bs[0]]:
        torch.testing.assert_close(torch.linalg.svdvals(x),torch.linalg.svdvals(x@a)[:3])
    with pytest.raises(ValueError):aggregate(bs,[a,a+1],w)


def test_qv_zero_k_merge_and_round_continuity():
    torch.manual_seed(4);base=torch.nn.Linear(8,24);layer=QVAdapter(base,3)
    x=torch.randn(2,5,8);original=base(x).detach()
    a=torch.linalg.qr(torch.randn(8,3)).Q.T;layer.start(a)
    torch.testing.assert_close(layer(x),original)
    with torch.no_grad():layer.Bq.normal_();layer.Bv.normal_()
    before=layer(x).detach();torch.testing.assert_close(before[...,8:16],original[...,8:16])
    # A round replaces B with aggregate; it must not call task start/reset.
    old=layer.Bq.detach().clone()
    with torch.no_grad():layer.Bq.copy_(.3*old+.7*(old+.1))
    assert layer.Bq.norm()>0
    before=layer(x).detach();layer.merge();torch.testing.assert_close(layer(x),before)
    with pytest.raises(ValueError):layer.merge()


def ns_reference(g):
    # Deliberately larger Gram, independent polynomial expression.
    x=g.float()/(g.float().norm()+1e-7)
    for _ in range(5):x=3.4445*x-4.7750*x@(x.T@x)+2.0315*x@(x.T@x)@(x.T@x)
    return x*max(g.shape)**.5


def test_two_clients_two_matrices_two_rounds_reference():
    torch.manual_seed(8)
    shapes=[(5,2),(3,4)];weights=[.5,.5]
    server=[torch.zeros(s) for s in shapes];c=[torch.zeros(s) for s in shapes]
    ms=[[torch.zeros(s) for s in shapes] for _ in range(2)]
    cs=copy.deepcopy(ms);rm=copy.deepcopy(ms);rc=copy.deepcopy(cs);rs=copy.deepcopy(server);rg=copy.deepcopy(c)
    for r in range(2):
        ends=[];refs=[]
        for k in range(2):
            params=[torch.nn.Parameter(v.clone()) for v in server];ref=[v.clone() for v in rs]
            opt=ClientMuon(params,ms[k],cs[k],c,True)
            frozen=copy.deepcopy(cs[k])
            for step in range(3):
                for j,p in enumerate(params):
                    grad=torch.randn_like(p);p.grad=grad.clone()
                    rm[k][j]=.9*rm[k][j]+.1*grad
                    ref[j]-=.002*ns_reference(rm[k][j]-rc[k][j]+rg[j])
                opt.step(.002)
                for j,p in enumerate(params):
                    torch.testing.assert_close(p,ref[j],atol=1e-6,rtol=1e-5)
                    torch.testing.assert_close(cs[k][j],frozen[j])
                    torch.testing.assert_close(ms[k][j],rm[k][j])
            ends.append([p.detach().clone() for p in params]);refs.append(ref)
        cs=copy.deepcopy(ms);rc=copy.deepcopy(rm)
        c=[sum(weights[k]*cs[k][j] for k in range(2)) for j in range(2)]
        rg=[sum(weights[k]*rc[k][j] for k in range(2)) for j in range(2)]
        server=[sum(weights[k]*ends[k][j] for k in range(2)) for j in range(2)]
        rs=[sum(weights[k]*refs[k][j] for k in range(2)) for j in range(2)]


@pytest.mark.parametrize('shape',[(9,3),(3,9),(5,5)])
def test_polar_rank_deficiency(shape):
    x=torch.zeros(shape,dtype=torch.float64)
    torch.testing.assert_close(compact_polar(x),x)
    x[0,0]=1.;x[1,1]=1e-9
    q=compact_polar(x)
    assert int(torch.linalg.matrix_rank(q))==2
    assert torch.count_nonzero(q[2:])==0
    torch.testing.assert_close(q[:2,:2],torch.eye(2,dtype=torch.float64))
    y=torch.randn(shape,dtype=torch.float64)
    polar=compact_polar(y)
    torch.testing.assert_close(torch.linalg.svdvals(polar),torch.ones(min(shape),dtype=torch.float64))


def test_checkpoint_next_round(tmp_path):
    torch.manual_seed(9);np.random.seed(9);random.seed(9)
    state={'B':torch.randn(4,2),'m':[torch.randn(4,2)],'c':[torch.randn(4,2)],'server':[torch.randn(4,2)],
           'history':History(),'pending':{'0':torch.eye(4)},'rng':rng_state()}
    state['history'].commit(0,{'0':torch.eye(4)})
    atomic_save(state,tmp_path/'c.pt')
    def next_round(s):
        b=torch.nn.Parameter(s['B'].clone());m=copy.deepcopy(s['m'])
        opt=ClientMuon([b],m,s['c'],s['server'],True)
        for _ in range(3):
            b.grad=torch.randn_like(b)*(np.random.rand()+random.random());opt.step(.01)
        return b,m
    expected=next_round(state)
    loaded=torch.load(tmp_path/'c.pt',weights_only=False);restore_rng(loaded['rng'])
    actual=next_round(loaded)
    torch.testing.assert_close(actual[0],expected[0],rtol=0,atol=0)
    torch.testing.assert_close(actual[1][0],expected[1][0],rtol=0,atol=0)
    assert loaded['history'].committed=={0}


def test_single_client_centralized_degeneracy():
    # Central reference independently follows Eq.3, preserves momentum across rounds.
    from fcl.math import mulora_ssb
    torch.manual_seed(33)
    fed=QVAdapter(torch.nn.Linear(7,21),3)
    central=copy.deepcopy(fed)
    fh,ch=History(),History()
    for task in range(2):
        x=torch.randn(17,7,dtype=torch.float64);s=x.T@x
        a,_=rrs(s,fh.raw.get('x',torch.zeros_like(s)),3)
        ac,_=rrs(s,ch.raw.get('x',torch.zeros_like(s)),3)
        fed.start(a);central.start(ac)
        fm=[torch.zeros_like(fed.Bq),torch.zeros_like(fed.Bv)]
        cm=copy.deepcopy(fm);zeros=copy.deepcopy(fm)
        params=[fed.Bq,fed.Bv];refs=[central.Bq,central.Bv]
        opt=ClientMuon(params,fm,zeros,zeros,False)
        for rnd in range(3):
            for step in range(2):
                lr=.002*(1+np.cos(np.pi*(rnd*2+step)/5))/2
                batch=torch.randn(2,4,7);target=torch.randn(2,4,21)
                for model in [fed,central]:
                    model.zero_grad();((model(batch)-target)**2).mean().backward()
                opt.step(lr)
                with torch.no_grad():
                    for j,p in enumerate(refs):
                        cm[j]=.9*cm[j]+.1*p.grad
                        # Independent no-dimension-multiplier reference.
                        p.add_(ns_reference(cm[j])/max(p.shape)**.5,alpha=-lr)
                for p,q in zip(params,refs):torch.testing.assert_close(p,q,atol=1e-6,rtol=1e-5)
            # FedAvg with one participant is identity, no reset at round boundary.
            for p in params:
                with torch.no_grad():p.copy_(aggregate([p.detach().clone()],[a],[1.]))
        fed.merge();central.merge();fh.commit(task,{'x':s});ch.commit(task,{'x':s})
        torch.testing.assert_close(fed.base.weight,central.base.weight,atol=1e-6,rtol=1e-5)
        assert fh.committed==ch.committed==set(range(task+1))


def test_statistics_access_guard_and_all_seen_contract():
    from fcl.data import current_training_indices
    from fcl.model import ContinualViT
    m={'cpt':2,'partitions':{'train':[[[0,1]],[[2,3]]],'test':[[[999]]]},
       'labels':{'train':[0,1,2,3],'test':None}}
    assert current_training_indices(m,0)==[0,1]
    m['partitions']['train'][0][0].append(2)
    with pytest.raises(ValueError):current_training_indices(m,0)
    class Backbone(torch.nn.Module):
        num_features=7
        blocks=[]
        def __init__(self):
            super().__init__();self.dummy=torch.nn.Parameter(torch.zeros(1))
        def forward_features(self,x):return x
    model=ContinualViT(Backbone(),4,rank=3)
    model.add_head(2);model.add_head(2)
    x=torch.randn(3,1,7)
    assert model(x).shape==(3,4)
    with pytest.raises(TypeError):model(x,task_id=0)
    assert not model.rows[0].requires_grad


def test_model_history_rng_resume_next_round(tmp_path):
    torch.manual_seed(53)
    model=QVAdapter(torch.nn.Linear(8,24),3)
    a=torch.linalg.qr(torch.randn(8,3)).Q.T;model.start(a)
    head=torch.nn.Parameter(torch.randn(4,8));m=[torch.zeros_like(model.Bq),torch.zeros_like(model.Bv)]
    controls=[torch.randn_like(v) for v in m];server=copy.deepcopy(controls)
    h=History();h.commit(0,{'0':torch.eye(8)})
    state={'model':copy.deepcopy(model.state_dict()),'head':head.detach().clone(),'m':copy.deepcopy(m),
           'c':controls,'server':server,'history':h,'pending':{'0':2*torch.eye(8)},'rng':rng_state()}
    atomic_save(state,tmp_path/'round.pt')
    def run(mod,hd,mom):
        opt=ClientMuon([mod.Bq,mod.Bv],mom,controls,server,True)
        ho=torch.optim.AdamW([hd],lr=.001,weight_decay=0)
        for step in range(3):
            x=torch.randn(2,3,8);y=torch.randn(2,3,4)
            mod.zero_grad();ho.zero_grad()
            z=mod(x)[...,:8]@hd.T;loss=(z-y).square().mean();loss.backward()
            opt.step(.001*(3-step)/3);ho.step()
        return mod.state_dict(),hd.detach().clone(),mom
    expected=run(model,head,m)
    loaded=torch.load(tmp_path/'round.pt',weights_only=False)
    other=QVAdapter(torch.nn.Linear(8,24),3);other.load_state_dict(loaded['model']);other.merged=False
    hd=torch.nn.Parameter(loaded['head']);restore_rng(loaded['rng'])
    actual=run(other,hd,loaded['m'])
    for key,value in expected[0].items():torch.testing.assert_close(actual[0][key],value,atol=0,rtol=0)
    torch.testing.assert_close(actual[1],expected[1],atol=0,rtol=0)
    assert loaded['history'].committed=={0}
    torch.testing.assert_close(loaded['pending']['0'],2*torch.eye(8))
