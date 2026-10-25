import copy
import json
import random
import time
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader,Subset
from fcl.model import MuLoRAFL
from fcl.checkpoint import rng_state
from research.head_regmean.pilot import Pilot,observe


class Block(nn.Module):
    def __init__(self):
        super().__init__();self.attn=nn.Module();self.attn.qkv=nn.Linear(5,15);self.norm=nn.LayerNorm(5)
    def forward(self,x):
        q,k,v=self.attn.qkv(x).chunk(3,-1)
        return self.norm(x+.1*torch.tanh(q+v+v.mean(1,keepdim=True)))

class Backbone(nn.Module):
    def __init__(self):super().__init__();self.num_features=5;self.blocks=nn.ModuleList([Block()])
    def forward_features(self,x):
        for b in self.blocks:x=b(x)
        return x

class Augment(nn.Module):
    def forward(self,x):return x+.01*torch.randn_like(x)

def loader(raw,ids,batch,shuffle=False):return DataLoader(Subset(raw,ids),batch_size=batch,shuffle=shuffle,num_workers=0)

def make(path,restore=False):
    torch.set_num_threads(2);torch.manual_seed(17);np.random.seed(17);random.seed(17)
    gen=torch.Generator().manual_seed(9);x=torch.randn(16,3,5,generator=gen);y=torch.arange(16)//4
    raw=[TensorDataset(x,y),TensorDataset(x.clone(),y.clone())]
    manifest={'partitions':{s:[[[0,1,2,4,5],[3,6,7],[]],[[8,9,10,12,13],[11,14,15],[]]] for s in ['train','test']}}
    cfg=dict(dataset='toy',seed=0,alpha=.05,clients=3,tasks=2,cpt=2,rounds=2,epochs=1,batch=2,rank=2,lr=.0002,cosine_scale=16)
    p=Pilot(MuLoRAFL(Backbone(),4,2,16),raw,manifest,loader,Augment(),nn.Identity(),cfg,path,
        {'manifest':'m','weights':'w'},{'started':1.,'deadline':1e30})
    if restore:p.restore()
    return p

def legacy(p):
    return {'task':0,'round':1,'initialized':True,'head_count':1,'head_type':'cosine','head_bias':False,
        'head_aggregation':'constrained_regmean','config':p.cfg,'history':copy.deepcopy(p.s['history']),
        'pending':copy.deepcopy(p.s['pending']),'scheduler':{'round':1},'manifest_hash':'m','weight_hash':'w',
        'regmean_certificate':p.s['certificates'][0], 'model':copy.deepcopy(p.model.state_dict()),
        'client_states':copy.deepcopy(p.s['client_states']),'original_start':1.,'rng':rng_state()}

def test_production_engine_resume_every_client_and_boundary(tmp_path):
    full=make(tmp_path/'full');full.run()
    resumed=make(tmp_path/'resume');phases=set()
    while resumed.step():
        phases.add(resumed.s['phase']);resumed=make(tmp_path/'resume',True)
    assert {'clients','local_done','round_start','ready_evaluation','evaluated','merged','committed','complete'}<=phases
    for name,value in full.model.state_dict().items():torch.testing.assert_close(value,resumed.model.state_dict()[name],rtol=0,atol=0)
    assert full.s['stages']==resumed.s['stages']
    assert full.s['merges']==[0,1] and full.s['history'].committed=={0,1}
    assert len(full.s['training_records'])==8 and len(full.s['certificates'])==4
    assert sum(r['visited'] for r in full.s['training_records'])==32
    assert full.s['communication']==resumed.s['communication']
    assert len([r for r in full.s['stats_records'] if r['kind']=='RRS'])==2
    for a in resumed.model.adapters.values():
        with pytest.raises(ValueError,match='Duplicate'):a.merge()
    with pytest.raises(ValueError):resumed.s['history'].commit(1,{})

def test_legacy_R1_import_preserves_trajectory_and_rejects_bad_protocol(tmp_path):
    original=make(tmp_path/'original')
    while not (original.s['round']==1 and original.s['phase']=='round_start'):original.step()
    source=legacy(original);original.run()
    restored=make(tmp_path/'imported');restored.import_round1(source)
    assert torch.equal(rng_state()['torch'],source['rng']['torch'])
    for k in source['client_states']:
        for a,b in zip(restored.s['client_states'][k]['m'],source['client_states'][k]['m']):assert torch.equal(a,b)
    restored.run()
    for name,value in original.model.state_dict().items():torch.testing.assert_close(value,restored.model.state_dict()[name],rtol=0,atol=0)
    assert original.s['stages']==restored.s['stages']
    assert len(restored.s['training_records'])==6 # no replay of first two nonempty clients
    bad=copy.deepcopy(source);bad['config']['epochs']=2
    with pytest.raises(ValueError,match='protocol'):make(tmp_path/'bad').import_round1(bad)

def test_local_endpoint_Gram_and_observation_RNG(tmp_path):
    p=make(tmp_path/'g');p.step();p.step()
    before=rng_state();modes=[m.training for m in p.model.modules()]
    x=p.raw[0].tensors[0][p.ids('train',0,0)]
    with torch.no_grad():z=torch.nn.functional.normalize(p.model.backbone.forward_features(x)[:,0],dim=1).double()
    actual=p.local_gram(0)
    torch.testing.assert_close(actual,z.T@z,rtol=1e-6,atol=1e-7)
    assert torch.equal(before['torch'],rng_state()['torch']) and before['python']==rng_state()['python']
    assert modes==[m.training for m in p.model.modules()]
    p.model.train()
    with observe(p.model):assert not p.model.training;torch.rand(3);np.random.rand();random.random()
    assert p.model.training and torch.equal(before['torch'],rng_state()['torch'])

def test_old_heads_frozen_and_deployed_scale_is_raw_average_norm(tmp_path):
    p=make(tmp_path/'freeze')
    while p.s['task']==0:p.step()
    old=p.model.rows[0].detach().clone()
    while p.step():assert torch.equal(old,p.model.rows[0])
    assert not p.model.rows[0].requires_grad
    saved=torch.load(p.out/'round_t1_r2.pt',weights_only=False)
    torch.testing.assert_close(saved['head'].norm(dim=1),saved['raw_FedAvg_head'].norm(dim=1),rtol=1e-6,atol=1e-7)
    assert saved['certificate']['deployed_fp32']['passed']
    assert not torch.equal(saved['head'],saved['raw_FedAvg_head'])

def test_phase_clock_identity_and_current_data_guard(tmp_path):
    p=make(tmp_path/'guard');p.step();p.save()
    with pytest.raises(ValueError,match='Old/future'):p.check_labels(torch.tensor([2]))
    x=torch.load(p.out/'latest.pt',weights_only=False);x['clock']['started']=2.;torch.save(x,p.out/'latest.pt')
    with pytest.raises(ValueError,match='clock'):make(p.out,True)
    p.clock['deadline']=time.time()-1
    with pytest.raises(TimeoutError):p.step()

def test_all_seen_competition_is_not_restricted_accuracy(tmp_path):
    p=make(tmp_path/'metrics');p.run();p.s['task']=1
    # Explicitly make task1 scores win for every sample while task2 restricted
    # metrics still classify within its two columns; this checks the denominator.
    def logits(x):
        out=torch.zeros(len(x),4);out[:,0]=2;out[:,2]=1
        return out
    p.model.forward=logits
    result=p.evaluate()
    assert result['new_accuracy']==0 and result['restricted'][1]==.5
    assert result['old_accuracy']==.5 and result['all_seen']==.25
    assert result['new_to_old_error']==1 and result['old_to_new_error']==0
