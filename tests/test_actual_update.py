import json
from pathlib import Path
import pytest
import numpy as np
import torch
from fcl.checkpoint import rng_state
from research.fed_spectra.actual_update import ActualUpdatePilot,global_shape_update
from research.fed_spectra.lifecycle import qualify
from test_fed_spectra import build as old_build,clean


def build(path,eta=.1,restore=False):
    p=old_build(path,'controlled_adam')
    p.cfg.update(method='actual_update_global',actual_update={'operator':{'eta':eta},'deployed_norm_tolerance':5e-4})
    result=ActualUpdatePilot(p.model,p.raw,p.manifest,p.loader,p.train_tf,p.test_tf,p.cfg,p.out,p.identity,p.clock)
    if restore:result.restore()
    return result


def test_analytic_diagonal_interpolation_norm_and_rank():
    a=torch.diag(torch.tensor([3.,1.,0.],dtype=torch.float64))
    b=torch.diag(torch.tensor([1.,3.,0.],dtype=torch.float64))
    base=.75*a+.25*b
    out,r=global_shape_update([a,b],[.75,.25],base,.1)
    p=torch.tensor([2.5**2,1.5**2],dtype=torch.float64);p/=p.sum()
    q=torch.tensor([.7,.3],dtype=torch.float64);q=(q+1e-12)/(q+1e-12).sum()
    target=p**.9*q**.1;target/=target.sum()
    torch.testing.assert_close(out.diag()[:2],target.sqrt()*base.norm(),rtol=1e-12,atol=1e-12)
    assert out[2,2]==0 and r['retained_rank']==2
    assert abs(float(out.norm()/base.norm())-1)<1e-12


@pytest.mark.parametrize('repeated',[False,True])
def test_coordinate_equivariance_permutation_and_scale(repeated):
    g=torch.Generator().manual_seed(15)
    ds=[torch.randn(7,3,dtype=torch.float64,generator=g) for _ in range(3)];w=[.2,.3,.5]
    if repeated:
        u=torch.linalg.qr(torch.randn(7,3,dtype=torch.float64,generator=g)).Q
        base=u@torch.diag(torch.tensor([2.,2.,1.],dtype=torch.float64))
    else:base=sum(p*d for p,d in zip(w,ds))
    out,_=global_shape_update(ds,w,base,.1)
    q=torch.linalg.qr(torch.randn(3,3,dtype=torch.float64,generator=g)).Q
    rotated,_=global_shape_update([d@q for d in ds],w,base@q,.1)
    torch.testing.assert_close(rotated,out@q,rtol=1e-10,atol=1e-10)
    reordered,_=global_shape_update(ds[::-1],w[::-1],base,.1)
    torch.testing.assert_close(reordered,out,rtol=1e-12,atol=1e-12)
    scaled,_=global_shape_update([13*d for d in ds],w,13*base,.1)
    torch.testing.assert_close(scaled,13*out,rtol=1e-10,atol=1e-10)


def test_zero_cancellation_identity_and_invalid_input():
    d=torch.arange(12,dtype=torch.float64).reshape(4,3)
    zero=torch.zeros_like(d)
    for ds,base in [([zero],zero),([d,-d],zero)]:
        out,_=global_shape_update(ds,[1.]*len(ds),base,.1);assert torch.equal(out,zero)
    out,_=global_shape_update([d],[1.],d,0.);assert torch.equal(out,d)
    with pytest.raises(ValueError):global_shape_update([d],[-1.],d,.1)
    with pytest.raises(ValueError):global_shape_update([d],[1.],d,1.1)
    with pytest.raises(ValueError):global_shape_update([d*float('nan')],[1.],d,.1)


def test_eta_zero_matches_original_control_complete_trajectory(tmp_path):
    reference=old_build(tmp_path/'reference','controlled_adam');reference.run();rng=rng_state()
    actual=build(tmp_path/'zero',eta=0.);actual.run()
    for key,v in reference.model.state_dict().items():assert torch.equal(v,actual.model.state_dict()[key]),key
    for key in ['stages','training_records','communication','certificates']:
        assert clean(reference.s[key])==clean(actual.s[key]),key
    assert torch.equal(rng['torch'],rng_state()['torch'])
    assert not actual.s['spectra_probes'] and not actual.s['spectra_geometry']


def test_resume_every_transition_qualification_and_frozen_old_heads(tmp_path):
    full=build(tmp_path/'full');full.run();rng=rng_state()
    resumed=build(tmp_path/'resume');old=None
    while resumed.step():
        if resumed.s['task']==1 and resumed.s['phase']=='task_start':old=resumed.model.rows[0].detach().clone()
        if old is not None:assert torch.equal(old,resumed.model.rows[0])
        resumed=build(resumed.out,restore=True)
    for key,v in full.model.state_dict().items():assert torch.equal(v,resumed.model.state_dict()[key]),key
    for key in ['stages','training_records','communication','actual_update_rounds']:
        assert clean(full.s[key])==clean(resumed.s[key]),key
    assert torch.equal(rng['torch'],rng_state()['torch'])
    q=qualify(full,3)
    assert q['qualified'] and q['reload_raw_predictions_equal'] and q['reload_centered_predictions_equal']
    assert len(q['actual_update_rounds'])==6 and q['extra_probe_forward_images']==0
    assert len(full.s['training_records'])==12 and q['training_visits']==48
    from scripts.actual_update_report import stage_metrics,paired
    for view in ['raw','centered']:
        rows=stage_metrics(q['stages'],view)
        assert len(rows)==3 and all(r['W']+r['S']==r['errors'] for r in rows)
    original=full.out/'task3_private_logits.npz'
    pair=paired(original,original,full.cfg['cpt'])
    assert pair['raw']['corrected']==pair['raw']['destroyed']==0
    with np.load(original) as f:data={k:f[k] for k in f.files}
    data['indices']=data['indices'][::-1]
    wrong=tmp_path/'wrong_order.npz';np.savez(wrong,**data)
    with pytest.raises(AssertionError):paired(original,wrong,full.cfg['cpt'])


def test_frozen_configs_differ_only_in_eta_and_output():
    root=Path(__file__).parents[1]
    a=json.loads((root/'configs/actual_update_control_seed0.json').read_text())
    b=json.loads((root/'configs/actual_update_global_seed0.json').read_text())
    assert a.pop('output')!=b.pop('output')
    assert a['actual_update']['operator'].pop('eta')==0.
    assert b['actual_update']['operator'].pop('eta')==.1
    assert a==b


def test_registered_repetition_seeds_and_matched_rng():
    import random
    from scripts.spectra_train import initialize_rng
    root=Path(__file__).resolve().parents[1]
    snapshots=[]
    for seed in (0,1,2):
        a=json.loads((root/f'configs/actual_update_control_seed{seed}.json').read_text())
        b=json.loads((root/f'configs/actual_update_global_seed{seed}.json').read_text())
        assert a['seed']==b['seed']==seed
        assert a['manifest']==b['manifest']==f'manifests/seq-cifar100_n10_a0.05_s{seed}.json'
        assert a['actual_update']['operator'].pop('eta')==0.
        assert b['actual_update']['operator'].pop('eta')==.1
        a.pop('output');b.pop('output');assert a==b
        if seed==0:base=a.copy()
        else:
            for k in ('seed','manifest','protocol_version'):a[k]=base[k]
            assert a==base
        def sample():return torch.rand(4),np.random.rand(4),random.random()
        initialize_rng(seed);x=sample()
        initialize_rng(seed);y=sample()
        assert torch.equal(x[0],y[0]) and np.array_equal(x[1],y[1]) and x[2]==y[2]
        snapshots.append(x[0])
    assert all(not torch.equal(snapshots[i],snapshots[j]) for i in range(3) for j in range(i))
