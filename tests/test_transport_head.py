import copy
import torch
from torch.nn import functional as F
from fcl.checkpoint import rng_state
from research.head_regmean.solver import aggregate_heads
from research.head_regmean.transport_head import TransportHeadPilot,aggregate_transport,shrink_cross
from test_old_response import prepared


def test_zero_drift_reduces_to_legacy_and_partition_counts():
    torch.manual_seed(22)
    ws=[torch.randn(3,7) for _ in range(2)];zs=[F.normalize(torch.randn(n,7),dim=1).double() for n in [9,31]];gs=[z.T@z for z in zs]
    old,raw,_=aggregate_heads(ws,gs,[9,31],True)
    new,newraw,cert=aggregate_transport(ws,gs,gs,[9,31])
    torch.testing.assert_close(new,old,atol=2e-7,rtol=2e-6);assert torch.equal(raw,newraw)
    torch.testing.assert_close(new.norm(dim=1),raw.norm(dim=1),rtol=2e-6,atol=2e-7)
    split=[zs[0][:2],zs[0][2:],zs[1]];sg=[z.T@z for z in split]
    ns,_,_=aggregate_transport([ws[0],ws[0],ws[1]],sg,sg,[2,7,31])
    torch.testing.assert_close(new,ns,rtol=2e-6,atol=2e-7)
    assert cert['deployed_fp32']['passed']


def test_nonsymmetric_cross_orientation_transports_function():
    torch.manual_seed(41);d=7
    z=F.normalize(torch.randn(60,d),dim=1).double();r,_=torch.linalg.qr(torch.randn(d,d,dtype=torch.float64));zg=z@r
    w=torch.randn(3,d);cross=z.T@zg
    assert not torch.allclose(cross,cross.T)
    assert torch.equal(shrink_cross(cross),.5*cross+.5*torch.diag(cross.diag()))
    out,_,cert=aggregate_transport([w],[zg.T@zg],[cross],[60],alpha=1)
    expected=F.normalize(w,dim=1).double()@r
    torch.testing.assert_close(F.normalize(out,dim=1).double(),expected,atol=2e-7,rtol=2e-6)
    assert cert['deployed_fp32']['passed']
    torch.testing.assert_close(zg@F.normalize(out,dim=1).double().T,z@F.normalize(w,dim=1).double().T,atol=2e-7,rtol=2e-6)


def test_paired_moments_preserve_rng_global_B_and_exact_correspondence(tmp_path):
    p=prepared(tmp_path/'mom');p.__class__=TransportHeadPilot;bs=p.model.b_parameters()
    local=[v.detach().clone()+.01 for v in bs];glob=[v.detach().clone()-.02 for v in bs]
    with torch.no_grad():
        x=p.raw[0].tensors[0][p.ids('train',1,0)]
        for b,v in zip(bs,local):b.copy_(v)
        zl=F.normalize(p.model.backbone.forward_features(x)[:,0],dim=1).double()
        for b,v in zip(bs,glob):b.copy_(v)
        zg=F.normalize(p.model.backbone.forward_features(x)[:,0],dim=1).double()
    before=rng_state();gl,gg,c,drift=p.paired_moments(0,local,glob)
    torch.testing.assert_close(gl,zl.T@zl,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(gg,zg.T@zg,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(c,zl.T@zg,atol=1e-6,rtol=1e-6)
    assert drift>0 and all(torch.equal(b,v) for b,v in zip(bs,glob))
    assert torch.equal(before['torch'],rng_state()['torch']) and before['python']==rng_state()['python']
    assert [v['samples'] for v in p.s['stats_records'][-2:]]==[5,5]


def test_first_round_exact_training_pair_and_full_restore_lifecycle(tmp_path):
    ref=prepared(tmp_path/'ref')
    while ref.s['phase']!='local_done':ref.step()
    ref_heads=copy.deepcopy(ref.s['heads']);ref_b=copy.deepcopy(ref.s['agg_B']);ref.fuse()
    target=torch.load(ref.out/'round_t1_r1.pt',weights_only=False)
    p=prepared(tmp_path/'run');p.__class__=TransportHeadPilot;teacher=copy.deepcopy(p.teacher.state_dict());old=p.model.rows[0].clone()
    while p.s['phase']!='local_done':p.step();p.restore()
    assert all(torch.equal(a,b) for a,b in zip(ref_heads,p.s['heads']))
    assert all(torch.equal(a,b) for a,b in zip(ref_b,p.s['agg_B']))
    p.expected_first_round=target
    while p.step():p.restore()
    assert p.s['transport_first_round_pair']['passed'] and p.s['transport_endpoints']=={}
    assert p.s['merges']==[0,1] and p.s['history'].committed=={0,1}
    assert all(v['deployed_fp32']['passed'] for v in p.s['certificates'])
    assert torch.equal(old,p.model.rows[0]) and all(torch.equal(v,p.teacher.state_dict()[k]) for k,v in teacher.items())
    records=[v for v in p.s['stats_records'] if v['kind'].startswith('paired_')]
    assert sum(v['samples'] for v in records)==32
    assert all(not torch.count_nonzero(v) for v in p.model.b_parameters())
    assert p.s['preservation_training']['visits']==16
