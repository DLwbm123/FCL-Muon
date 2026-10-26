import copy
import random
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset
from fcl.math import mulora_ssb, rrs
from fcl.model import MuLoRAFL
from fcl.checkpoint import rng_state
from research.head_regmean.confirmation import ConfirmationPilot
from research.aggregation_utility.selection import candidate_pool, ssb_batch, quantities, fit_search, certify
from research.aggregation_utility.pilot import UtilityPilot, split_indices, passive_probe, zero_probe
from research.aggregation_utility.lifecycle import import_prefix
from test_head_regmean_pilot import Backbone,Augment,loader

SETTINGS=dict(candidate_width=4,formal_rank=2,split_seed=76103,check_fraction=.2,search_steps=25,
              initial_step=.25,backtracks=16,inward_derivative=.001,constraint_relative_tolerance=2e-6,
              minimum_relative_gain=1e-4,orthogonal_tolerance=1e-5)


def test_batched_ssb_matches_original_and_projection_must_precede_ssb():
    torch.manual_seed(1)
    c=torch.randn(3,2,9,6);z=torch.linalg.qr(torch.randn(6,2)).Q
    projected=c@z
    actual=ssb_batch(.1*projected)
    reference=torch.stack([torch.stack([mulora_ssb(.1*v) for v in client]) for client in projected])
    torch.testing.assert_close(actual,reference,atol=2e-6,rtol=2e-5)
    assert not torch.allclose(actual,ssb_batch(.1*c)@z,atol=1e-4,rtol=1e-4)
    concatenated=mulora_ssb(torch.cat(list(projected[0]),dim=0)*.1)
    assert not torch.allclose(actual[0].flatten(0,1),concatenated,atol=1e-4,rtol=1e-4)


def test_candidate_pool_contains_exact_rrs_and_extra_independent_directions():
    torch.manual_seed(2)
    x=torch.randn(30,8,dtype=torch.float64);s=x.T@x
    y=torch.randn(30,8,dtype=torch.float64);h=y.T@y
    a0,_=rrs(s,h,2)
    p,z0,_=candidate_pool(s,h,a0,4)
    assert torch.equal((p@z0).T,a0)
    torch.testing.assert_close(p.T@p,torch.eye(4),atol=1e-6,rtol=0)
    assert torch.linalg.matrix_rank(p)==4


def test_signed_federated_utility_rotation_and_first_order_response():
    torch.manual_seed(3)
    c=torch.randn(3,2,9,6);p=torch.linalg.qr(torch.randn(12,6)).Q
    z=torch.linalg.qr(torch.randn(6,2)).Q;weights=torch.tensor([.1,.3,.6])
    h=torch.diag(torch.arange(1,7,dtype=torch.float64))
    q=quantities(z,c,weights,h,'federated')
    rotation=torch.linalg.qr(torch.randn(2,2)).Q
    qr=quantities(z@rotation,c,weights,h,'federated')
    torch.testing.assert_close(q['utility'],qr['utility'],rtol=1e-5,atol=1e-5)
    delta=-torch.einsum('vdr,pr,ep->vde',q['direction'],z,p)
    full_gradient=torch.einsum('k,kvdp,ep->vde',weights,c,p)
    torch.testing.assert_close((full_gradient*delta).sum(),-q['utility'],rtol=1e-5,atol=1e-5)
    expected=torch.einsum('vdr,rs,vds->',q['direction'].double(),z.double().T@h@z.double(),q['direction'].double())
    torch.testing.assert_close(q['risk'],expected)
    cancelling=torch.stack([c[0],-c[0]])
    cancelled=quantities(z,cancelling,torch.tensor([.5,.5]),h,'federated')
    assert float(cancelled['utility'])==0 and float(cancelled['update_norm'])==0


def test_zero_probe_full_gradient_identity_and_state_restoration():
    torch.manual_seed(4);model=MuLoRAFL(Backbone(),4,2,16);model.add_head(2);model.eval()
    x=torch.randn(3,3,5);pool=torch.linalg.qr(torch.randn(5,4)).Q
    adapter=model.adapters['0'];adapter.base.weight.requires_grad_(True)
    model(x).square().sum().backward()
    full=adapter.base.weight.grad.detach().clone()
    expected=[full[:5]@pool,full[10:]@pool]
    model.zero_grad(set_to_none=True)
    before=model(x).detach();state=rng_state();flags=[p.requires_grad for p in model.parameters()]
    with passive_probe(model),zero_probe(model,{'0':pool}) as probes:
        torch.rand(7)
        actual=model(x)
        assert torch.equal(actual.detach(),before)
        actual.square().sum().backward()
        for got,want in zip(probes['0'],expected):torch.testing.assert_close(got.grad,want,rtol=2e-5,atol=2e-5)
    assert torch.equal(state['torch'],rng_state()['torch'])
    assert flags==[p.requires_grad for p in model.parameters()]
    assert not adapter._forward_hooks


def test_fit_only_search_and_single_check_fallback():
    torch.manual_seed(15)
    c=torch.randn(3,2,12,4);c[:,:,:,:2]*=.2
    weights=torch.tensor([.2,.3,.5]);h=torch.diag(torch.tensor([8.,7.,.1,.2],dtype=torch.float64))
    z0=torch.eye(4)[:,:2];p=torch.linalg.qr(torch.randn(8,4)).Q
    z,report=fit_search(c,weights,h,z0,'federated',SETTINGS)
    assert report['steps_accepted']>0
    assert report['fit_candidate']['utility']>report['baseline']['utility']
    accepted,cert=certify(p,z,z0,c,weights,c.clone(),weights,h,'federated',SETTINGS)
    assert cert['accepted'] and all(cert['checks'].values())
    rejected,bad=certify(p,z,z0,c,weights,-c,weights,h,'federated',SETTINGS)
    assert not bad['accepted'] and not bad['checks']['independent_check']
    assert torch.equal(rejected,(p@z0).T.contiguous())
    # Check data cannot alter the selected fit candidate: API separation is explicit.
    z2,_=fit_search(c,weights,h,z0,'federated',SETTINGS)
    assert torch.equal(z,z2)


def toy(path,kind='rrs',restore=False,plain=False):
    torch.set_num_threads(2);torch.manual_seed(17);np.random.seed(17);random.seed(17)
    x=torch.randn(24,3,5,generator=torch.Generator().manual_seed(9));y=torch.arange(24)//4
    raw=[TensorDataset(x,y),TensorDataset(x.clone(),y.clone())]
    partitions=[[[t*8+i for i in [0,1,2,4,5]],[t*8+i for i in [3,6,7]],[]] for t in range(3)]
    manifest={'partitions':{s:partitions for s in ['train','test']},'labels':{'train':y.tolist()}}
    cfg=dict(dataset='toy',seed=0,alpha=.05,clients=3,tasks=3,cpt=2,rounds=2,epochs=1,batch=2,rank=2,
             lr=.0002,cosine_scale=16,aggregation='regmean',selection=copy.deepcopy(SETTINGS),selection_mode=kind,
             retain_task_checkpoints=False,retain_round_checkpoints=False,retain_first_round_endpoints=False)
    cls=ConfirmationPilot if plain else UtilityPilot
    p=cls(MuLoRAFL(Backbone(),6,2,16),raw,manifest,loader,Augment(),nn.Identity(),cfg,path,
          {'manifest':'toy','weights':'toy'},{'started':1.,'deadline':1e30})
    if restore:p.restore()
    return p


def test_rrs_control_probe_leaves_training_trajectory_exact(tmp_path):
    baseline=toy(tmp_path/'plain',plain=True);baseline.run()
    instrumented=toy(tmp_path/'rrs');instrumented.run()
    for key,value in baseline.model.state_dict().items():assert torch.equal(value,instrumented.model.state_dict()[key])
    for key in ['stages','matrix','merges','training_records','stats_records','certificates']:
        assert baseline.s[key]==instrumented.s[key]
    assert len(instrumented.s['selection_records'])==2
    assert all(r['accepted_layers']==0 for r in instrumented.s['selection_records'])
    assert all(r['fit_samples']+r['check_samples']==8 for r in instrumented.s['selection_records'])


def test_shared_prefix_import_and_selected_resume_are_exact(tmp_path):
    p=toy(tmp_path/'prefix',kind='prefix')
    while not (p.s['task']==1 and p.s['phase']=='task_start'):p.step()
    source=torch.load(p.out/'latest.pt',weights_only=False)
    full=toy(tmp_path/'full',kind='federated');import_prefix(full,source);full.run()
    recovered=toy(tmp_path/'recovered',kind='federated');import_prefix(recovered,source)
    while recovered.step():
        recovered=toy(recovered.out,kind='federated',restore=True)
    for key,value in full.model.state_dict().items():assert torch.equal(value,recovered.model.state_dict()[key])
    assert full.s['stages']==recovered.s['stages']
    assert full.s['merges']==[0,1,2] and full.s['history'].committed=={0,1,2}
    assert len(full.s['training_records'])==12 and sum(v['visited'] for v in full.s['training_records'])==48
    invalid=copy.deepcopy(source);invalid['clock']['started']=9
    with pytest.raises(ValueError,match='clock'):import_prefix(toy(tmp_path/'bad',kind='central'),invalid)


def test_stratified_partition_no_overlap_and_singletons_stay_fit():
    ids=list(range(11));labels=[0]*5+[1]*5+[2]
    fit,check=split_indices(ids,labels,0,1,3,76103,.2)
    assert set(fit).isdisjoint(check) and sorted(fit+check)==ids and 10 in fit
    assert {labels[i] for i in fit}=={0,1,2}
    assert (fit,check)==split_indices(ids,labels,0,1,3,76103,.2)
