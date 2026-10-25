import torch
from research.head_regmean.confirmation import ConfirmationPilot
from test_head_regmean_pilot import make


def paired(path,aggregation):
    p=make(path);p.__class__=ConfirmationPilot;p.cfg['aggregation']=aggregation
    return p


def test_confirmation_preserves_original_regmean_training(tmp_path):
    original=make(tmp_path/'original');original.run()
    new=paired(tmp_path/'new','regmean');new.run()
    for k,v in original.model.state_dict().items():
        torch.testing.assert_close(v,new.model.state_dict()[k],rtol=0,atol=0)
    assert original.s['matrix']==new.s['matrix']
    assert original.s['communication']==new.s['communication']
    assert all(a['confusion']==b['confusion'] for a,b in zip(original.s['stages'],new.s['stages']))


def test_paired_endpoints_rng_and_fedavg_rule(tmp_path):
    r=paired(tmp_path/'regmean','regmean');r.run()
    f=paired(tmp_path/'fedavg','fedavg');f.run()
    a=torch.load(r.out/'first_round_endpoints.pt',weights_only=False)
    b=torch.load(f.out/'first_round_endpoints.pt',weights_only=False)
    for name in ['heads','B','A']:
        assert all(torch.equal(x,y) for x,y in zip(a[name],b[name]))
    assert a['sizes']==b['sizes'] and torch.equal(a['rng']['torch'],b['rng']['torch'])
    assert a['rng']['python']==b['rng']['python']
    assert f.s['certificates']==[]
    assert all(v['kind']=='RRS' for v in f.s['stats_records'])
    assert 'head_Gram_upload' not in f.s['communication']
    rd=torch.load(f.out/'round_t0_r1.pt',weights_only=False)
    expected=torch.zeros_like(b['heads'][0])
    for head,n in zip(b['heads'],b['sizes']):expected.add_(head,alpha=n/sum(b['sizes']))
    assert torch.equal(expected,rd['head'])
