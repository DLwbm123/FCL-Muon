import random
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset
from fcl.model import MuLoRAFL
from research.head_regmean.confirmation import ConfirmationPilot
from test_head_regmean_pilot import Backbone,Augment,loader
from scripts.ten_task_audit import trajectory


def make_ten(path,compact,restore=False):
    torch.set_num_threads(2);torch.manual_seed(17);np.random.seed(17);random.seed(17)
    x=torch.randn(80,3,5,generator=torch.Generator().manual_seed(9));y=torch.arange(80)//4
    raw=[TensorDataset(x,y),TensorDataset(x.clone(),y.clone())]
    partitions=[[[t*8+i for i in [0,1,2,4,5]],[t*8+i for i in [3,6,7]],[]] for t in range(10)]
    m={'partitions':{s:partitions for s in ['train','test']}}
    cfg=dict(dataset='toy',seed=0,alpha=.05,clients=3,tasks=10,cpt=2,rounds=2,epochs=1,batch=2,
             rank=2,lr=.0002,cosine_scale=16,aggregation='regmean')
    if compact:cfg.update(retain_task_checkpoints=False,retain_round_checkpoints=False,retain_first_round_endpoints=False)
    p=ConfirmationPilot(MuLoRAFL(Backbone(),20,2,16),raw,m,loader,Augment(),nn.Identity(),cfg,path,
                        {'manifest':'toy','weights':'toy'},{'started':1.,'deadline':1e30})
    if restore:p.restore()
    return p


def test_ten_tasks_compact_retention_exact_trajectory_and_boundary_restore(tmp_path):
    full=make_ten(tmp_path/'full',False);full.run()
    compact=make_ten(tmp_path/'compact',True);boundaries=[]
    while compact.step():
        if compact.s['phase'] in ['committed','task_start','complete']:
            boundaries.append((compact.s['task'],compact.s['phase']))
            compact=make_ten(compact.out,True,True)
    assert (3,'task_start') in boundaries and (10,'complete') in boundaries
    for key,value in full.model.state_dict().items():assert torch.equal(value,compact.model.state_dict()[key])
    for key in ['stages','matrix','merges','communication','training_records','stats_records','certificates']:
        assert full.s[key]==compact.s[key]
    assert compact.s['merges']==list(range(10)) and compact.s['history'].committed==set(range(10))
    assert len(compact.s['training_records'])==40 and sum(r['visited'] for r in compact.s['training_records'])==160
    assert len(compact.s['certificates'])==20 and all(c['deployed_fp32']['passed'] for c in compact.s['certificates'])
    assert len(compact.s['matrix'])==10 and [len(r) for r in compact.s['matrix']]==list(range(1,11))
    assert all(not torch.count_nonzero(b) for b in compact.model.b_parameters())
    assert sorted(p.name for p in compact.out.glob('*.pt'))==['latest.pt']
    assert len(list(full.out.glob('task*.pt')))==10
    compact.s['task']=9;reloaded=compact.evaluate()
    assert reloaded['confusion']==compact.s['stages'][-1]['confusion']
    assert reloaded['centered']==compact.s['stages'][-1]['centered']


def test_trajectory_new_acquisition_and_forgetting_denominators():
    matrix=[[.9-.01*(t-i) for i in range(t+1)] for t in range(10)]
    got=trajectory(matrix)
    assert got['a_tt']==[.9]*10
    assert got['mean_new_T2_T10']==pytest.approx(.9)
    assert got['final_old']==pytest.approx(.85)
    assert got['final_new']==.9 and got['forgetting']==pytest.approx(.05)
    with pytest.raises(AssertionError):trajectory(matrix[:3])
