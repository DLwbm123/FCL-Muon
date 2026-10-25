import copy
import torch
from research.head_regmean.frozen_encoder import FrozenEncoderPilot
from research.head_regmean.head_loss import separated_backward
from test_head_regmean_pilot import make


def prepared(path):
    p=make(path)
    while p.s['task']==0:p.step()
    p.step()
    p.__class__=FrozenEncoderPilot
    p.cfg.update(aggregation='regmean',head_loss_mode='centered_all_seen',encoder_frozen=True)
    return p


def test_frozen_head_gradient_matches_adapting_encoder(tmp_path):
    p=prepared(tmp_path/'grad')
    x=p.raw[0].tensors[0][p.ids('train',1,0)];y=p.raw[0].tensors[1][p.ids('train',1,0)]
    separated_backward(p.model,x,y,1,2,'centered_all_seen')
    expected=p.model.rows[-1].grad.clone()
    p.model.zero_grad(set_to_none=True);p.freeze_encoder();p.backward_loss(x,y)
    assert torch.equal(expected,p.model.rows[-1].grad)
    assert all(v.grad is None for v in p.model.backbone.parameters())


def test_frozen_lifecycle_preserves_encoder_and_old_logits(tmp_path):
    p=prepared(tmp_path/'full');p.freeze_encoder()
    start=copy.deepcopy(p.model.backbone.state_dict());old=p.model.rows[0].clone()
    x=p.raw[1].tensors[0][:8]
    with torch.no_grad():scores=p.model(x)[:,:2].clone()
    nr=len(p.s['training_records']);comm=dict(p.s['communication']);p.run()
    assert all(torch.equal(v,p.model.backbone.state_dict()[k]) for k,v in start.items())
    assert torch.equal(old,p.model.rows[0])
    with torch.no_grad():assert torch.equal(scores,p.model(x)[:,:2])
    assert all(v['B_steps']==0 and v['head_steps']>0 for v in p.s['training_records'][nr:])
    assert p.s['communication'].get('B_upload',0)==comm.get('B_upload',0)
    assert p.s['communication'].get('B_download',0)==comm.get('B_download',0)
    assert p.s['merges']==[0,1] and p.s['history'].committed=={0,1}
    assert all(v['deployed_fp32']['passed'] for v in p.s['certificates'])
    p.restore()
    with torch.no_grad():assert torch.equal(scores,p.model(x)[:,:2])
