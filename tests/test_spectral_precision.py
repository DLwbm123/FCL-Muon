"""Independent analytic/FP64 checks for FP32 proposals and applied increments."""
from types import SimpleNamespace
import pytest
import torch
from fcl.spectral_qp import (SolverConfig, certificate, actual_update_certificate,
                             solve_spectral_qp, spectral_project)
from fcl.history_optimizer import HistoryOptimizer


def test_effective_rank_uses_amplitudes_and_labels_legacy_energy():
    from fcl.math import spectrum
    info=spectrum(torch.diag(torch.tensor([9.,1.])))
    assert abs(info['effective_rank']-1.3841454885)<2e-7
    assert abs(info['legacy_energy_entropy_rank']-1.0680781756)<2e-7
    assert info['effective_rank_definition']=='singular_value_amplitude_entropy'


def test_certificate_matches_analytic_diagonal_problem():
    g=torch.diag(torch.tensor([2.,1.]));h=torch.diag(torch.tensor([100.,1.]))
    d=torch.diag(torch.tensor([-.02,-1.]));cfg=SolverConfig()
    info=certificate(g,h,d,1.,cfg)
    # Independent scalar arithmetic on exactly the FP32 values being audited.
    a,b=map(float,d.diag());k1,k2=2+100*a,1+b
    expected=k1*a+k2*b+abs(k1)+abs(k2)
    assert abs(info['gap']-expected)<1e-15
    assert abs(info['objective']-(2*a+b+50*a*a+.5*b*b))<1e-15
    assert info['certificate_precision']=='torch.float64' and info['converged']


@pytest.mark.parametrize('eta,failed_check',[(1e-7,'feasible_within_tolerance'),(2e-7,'gap_satisfied')])
def test_exact_proposal_can_fail_after_fp32_parameter_addition(eta,failed_check):
    g=torch.ones(1,1);h=torch.zeros(1,1);cfg=SolverConfig()
    d,proposal=solve_spectral_qp(g,h,eta,config=cfg)
    before=torch.ones_like(g);after=before+d
    actual=actual_update_certificate(g,h,before,after,eta,cfg)
    assert proposal['converged'] and not actual['converged']
    assert not actual[failed_check]
    assert float(after-before)!=float(d)


def test_endpoint_subtraction_is_done_after_promotion():
    g=torch.ones(1,1);h=torch.zeros(1,1)
    before=torch.ones(1,1);after=torch.full((1,1),-1e-8)
    info=actual_update_certificate(g,h,before,after,1.,SolverConfig())
    exact_difference=float(before)-float(after)
    assert info['operator_norm']==exact_difference
    assert info['operator_norm']!=float((after-before).abs())


def test_applied_certificate_uses_same_psd_repair_as_solver():
    g=torch.eye(2);h=torch.diag(torch.tensor([-1e-8,1.]));cfg=SolverConfig()
    d,proposal=solve_spectral_qp(g,h,.2,config=cfg)
    actual=actual_update_certificate(g,h,torch.zeros_like(d),d,.2,cfg)
    assert actual['certificate_psd_correction']==proposal['psd_correction']>0
    assert actual['objective']==proposal['objective'] and actual['gap']==proposal['gap']


def test_relative_gap_above_rtol_is_not_by_itself_failure():
    # Absolute tolerance matters when eta*||G||_* is small.
    g=torch.tensor([[1e-3]],dtype=torch.float64);h=g*0
    info=certificate(g,h,torch.tensor([[-0.99985e-5]],dtype=torch.float64),1e-5,SolverConfig())
    assert info['relative_gap']>1e-4
    assert info['gap']<info['gap_limit'] and info['converged']


@pytest.mark.parametrize('zero_mode',[False,True])
def test_small_nonzero_modes_are_retained_without_filling_nullspace(zero_mode):
    values=[1.]+[9e-5]*31
    if zero_mode:values[-1]=0.
    width=31 if zero_mode else 32
    g=torch.zeros(768,32);g[:32]=torch.diag(torch.tensor(values))
    d,info=solve_spectral_qp(g,torch.zeros(32,32),1.,config=SolverConfig())
    assert info['rank_tolerance']<9e-5 and info['converged']
    torch.testing.assert_close(d[:width,:width],-torch.eye(width))
    assert not torch.count_nonzero(d[width:]) and not torch.count_nonzero(d[:,width:])


def test_iteration_budget_does_not_claim_success():
    torch.manual_seed(21);g=torch.randn(9,3,dtype=torch.float64)
    h=torch.diag(torch.tensor([.1,10.,100.],dtype=torch.float64))
    cfg=SolverConfig(max_iter=0,gap_rtol=1e-12,gap_atol=1e-14)
    _,info=solve_spectral_qp(g,h,.2,config=cfg)
    assert not info['converged'] and info['iteration_limit_reached']


def test_optimizer_keeps_proposal_and_every_failed_update():
    b=torch.nn.Parameter(torch.ones(1,1));events=[]
    ext=SimpleNamespace(provider='mulora',method='history_spectral',lam=0.,
        cores=[{'H':torch.zeros(1,1),'F':torch.zeros(1,1),'q0':0.}],
        caches={0:[None]},solver=SolverConfig(),rho_factor=1.,audit_enabled=False,
        event=lambda kind,**kw:events.append(kw))
    opt=HistoryOptimizer([b],[torch.zeros_like(b)],[torch.zeros_like(b)],[torch.zeros_like(b)],ext,0)
    for _ in range(2):b.grad=torch.ones_like(b);opt.step(1e-7)
    opt.flush(1,0);record=events[0]
    assert record['nonconverged']==2 and len(record['first_step_per_layer'])==1
    assert [d['step'] for d in record['failed_updates']]==[0,1]
    assert record['failure_counts']['parameter_addition']==2
    assert all(d['proposed_certificate']['converged'] for d in record['failed_updates'])
    assert all(not d['actual_certificate']['converged'] for d in record['failed_updates'])


@pytest.mark.parametrize('precision',['input','float64'])
def test_pro_candidate_counterexample_in_actual_torch(precision):
    import json
    from pathlib import Path
    case=json.loads((Path(__file__).parent/'fixtures/pro_spectral_counterexample.json').read_text())
    g=torch.tensor(case['g']);h=torch.tensor(case['h'])
    cfg=SolverConfig(reduced_dtype=precision)
    d,info=solve_spectral_qp(g,h,case['eta'],config=cfg)
    # A budget-exhausted reference must not be waved through by this regression.
    assert info['converged'] and info['proposed_certificate']['converged']
    assert info['reduced_certificate'] is not None
    torch.testing.assert_close(torch.tensor(info['objective']),
        (g.double()*d.double()).sum()+.5*((d.double()@h.double())*d.double()).sum(),check_dtype=False)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA driver regression')
def test_cuda_projection_and_restored_proposal_against_fp64():
    torch.manual_seed(30);g=torch.randn(768,32,device='cuda')*1e-5
    eta=2e-4;h=torch.diag(torch.logspace(-3,0,32,device='cuda'))
    projected=g
    for _ in range(20):projected=spectral_project(projected,eta)
    norm=float(torch.linalg.svdvals(projected.double(),driver='gesvd').max())
    assert norm<=eta*(1+64*torch.finfo(g.dtype).eps)
    d,info=solve_spectral_qp(g,h,eta)
    independent=certificate(g,h,d,eta,SolverConfig())
    assert info['svd_driver']=='gesvd' and independent['converged']
    assert independent['feasible_within_tolerance']
