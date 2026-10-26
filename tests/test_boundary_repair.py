import copy
import torch
import pytest
from fcl.checkpoint import rng_state
from research.boundary_repair.math import inner,norm2,repair,metric,cross_loss
from research.boundary_repair.pilot import BoundaryPilot
from research.boundary_repair.lifecycle import import_boundary,config_hash
from test_aggregation_utility import toy

SETTINGS={'enabled':True,'lambda':1.,'kappa':.25,'epsilon':1e-12,
          'relative_denominator_tolerance':1e-10,'geometry_tolerance':1e-7}


def data():
    torch.manual_seed(71)
    g=[torch.randn(5,3,dtype=torch.float64) for _ in range(4)]
    h=[torch.randn_like(x) for x in g];base=[torch.randn_like(x) for x in g]
    qs=[]
    for _ in g:
        x=torch.randn(3,3,dtype=torch.float64);qs.append(torch.eye(3)+x.T@x)
    return g,h,base,qs,[torch.linalg.inv(q) for q in qs]


def test_global_projection_cap_and_degeneracy():
    g,h,base,qs,invs=data();head=1.-inner(h,base)
    full,info=repair(g,h,base,head,qs,invs,kappa=100)
    assert info['reason']=='full'
    assert abs(float(inner(g,full)))<1e-12 and abs(float(inner(h,full))+1)<1e-12
    cap,ci=repair(g,h,base,head,qs,invs,kappa=1e-4)
    assert ci['reason']=='capped' and abs(float(inner(g,cap)))<1e-12
    assert float(norm2(cap,qs).sqrt())<=1.000000001e-4*float(norm2(base,qs).sqrt())
    assert abs(ci['ideal_remaining_cross']-(1-ci['scale']))<1e-12
    for a,b in zip(cap,full):torch.testing.assert_close(a,b*ci['scale'])
    zeros=[torch.zeros_like(v) for v in g]
    z,zi=repair(g,g,base,1.-inner(g,base),qs,invs)
    assert zi['reason']=='degenerate_projection' and all(torch.count_nonzero(v)==0 for v in z)
    z,zi=repair(zeros,h,base,head,qs,invs,kappa=100)
    assert zi['reason']=='full' and abs(float(inner(h,z))+1)<1e-12
    z,zi=repair(g,zeros,base,torch.tensor(1.),qs,invs)
    assert zi['reason']=='no_B_cross_gradient'
    z,zi=repair(g,h,zeros,torch.tensor(1.),qs,invs)
    assert zi['scale']==0 and all(torch.count_nonzero(v)==0 for v in z)
    z,zi=repair(g,h,base,-inner(h,base),qs,invs)
    assert zi['reason']=='no_conflict'


def test_metric_uses_full_history_scale_and_shared_geometry():
    a=torch.tensor([[1.,0.,0.],[0.,1.,0.]])
    h=torch.diag(torch.tensor([1.,2.,30.]))
    q,inv,scale=metric(a,h)
    assert scale==11.
    torch.testing.assert_close(q,torch.diag(torch.tensor([1+1/11,1+2/11],dtype=torch.float64)))
    torch.testing.assert_close(q@inv,torch.eye(2,dtype=torch.float64))


def build(path,enabled=True,restore=False):
    p=toy(path,kind='federated');p.cfg['boundary_repair']=dict(SETTINGS,enabled=enabled)
    p.__class__=BoundaryPilot
    if restore:p.restore()
    return p


def test_off_and_T1_exact_original_updates_rng(tmp_path):
    a=toy(tmp_path/'a',kind='federated');a.run();end=rng_state()
    b=build(tmp_path/'b',enabled=False);b.run()
    for k,v in a.model.state_dict().items():assert torch.equal(v,b.model.state_dict()[k])
    assert a.s['training_records']==b.s['training_records'] and torch.equal(end['torch'],rng_state()['torch'])
    a=toy(tmp_path/'t1a',kind='federated')
    while a.s['task']==0:a.step()
    end=rng_state()
    b=build(tmp_path/'t1b')
    while b.s['task']==0:b.step()
    for k,v in a.model.state_dict().items():assert torch.equal(v,b.model.state_dict()[k])
    assert a.s['training_records']==b.s['training_records'] and torch.equal(end['torch'],rng_state()['torch'])


def test_repair_resume_at_every_transition_exact(tmp_path):
    a=build(tmp_path/'a');a.run();end=rng_state()
    b=build(tmp_path/'b')
    while b.step():b=build(b.out,restore=True)
    for k,v in a.model.state_dict().items():assert torch.equal(v,b.model.state_dict()[k])
    assert a.s['training_records']==b.s['training_records'] and a.s['stages']==b.s['stages']
    assert b.s['merges']==[0,1,2] and b.s['history'].committed=={0,1,2}
    assert len(b.s['training_records'])==12 and torch.equal(end['torch'],rng_state()['torch'])
    assert sum(v['steps'] for v in b.s['repair_records'])==20
    assert len(b.s['finite_step_windows'])==6


def test_horizon_import_same_trajectory_and_reject_config(tmp_path):
    a=toy(tmp_path/'long',kind='federated');a.run()
    prefix=toy(tmp_path/'short',kind='federated');prefix.cfg['tasks']=2;prefix.run()
    src=torch.load(prefix.out/'latest.pt',weights_only=False)
    exp={'completed_tasks':2,'config_sha256':config_hash(src['config']),'identity':src['identity'],
         'clock':src['clock'],'selection_mode':'federated','toy':True}
    b=toy(tmp_path/'new',kind='federated');b.clock={'started':7.,'deadline':1e30};import_boundary(b,src,exp);b.run()
    for k,v in a.model.state_dict().items():assert torch.equal(v,b.model.state_dict()[k])
    assert a.s['training_records']==b.s['training_records'] and a.s['stages']==b.s['stages']
    bad=toy(tmp_path/'bad',kind='federated');bad.cfg['lr']*=2
    with pytest.raises(ValueError,match='Scientific'):import_boundary(bad,src,exp)
