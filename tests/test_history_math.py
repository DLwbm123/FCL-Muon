import copy
import pytest
import torch
from fcl.functional_history import InputHistory,common_core,risk_value
from fcl.spectral_qp import SolverConfig,solve_spectral_qp,solve_frobenius_qp,compact_polar,objective
from fcl.history_optimizer import paired_update_audit,propose

CFG=SolverConfig(max_iter=200,gap_rtol=1e-9,gap_atol=1e-12)


def test_three_anchors_exact_centering_and_new_basis():
    torch.manual_seed(4);d=5;o=4
    w={p:torch.randn(o,d,dtype=torch.float64) for p in ['q','v']}
    bank=InputHistory(d,o);explicit=[]
    for t,n in enumerate([3,11,6]):
        delta={p:torch.randn(o,d,dtype=torch.float64)*.1 for p in w}
        bank.recenter(t,delta,f'c{t}')
        w={p:w[p]+delta[p] for p in w}
        x=torch.randn(n,d,dtype=torch.float64)*torch.linspace(.3,2,n)[:,None]
        anchors={p:x@w[p].T for p in w};c=x.T@x
        prior_e={p:bank.risks[p]['e'] for p in w}
        bank.register(t,c,{p:float(anchors[p].square().sum()) for p in w},n,f'c{t}')
        assert all(bank.risks[p]['e']==prior_e[p] for p in w)
        explicit.append((x,anchors))
        a=torch.linalg.qr(torch.randn(d,3,dtype=torch.float64)).Q.T;b=torch.randn(o,3,dtype=torch.float64)*.05
        for p in w:
            core=bank.project(a,p)
            expected=sum(.5*((xx@(w[p]+b@a).T)-yy[p]).square().sum() for xx,yy in explicit)/core['Z']
            torch.testing.assert_close(risk_value(core,b),expected)
            grad=core['F']+b@core['H'];eps=1e-6;direction=torch.randn_like(b)
            fd=(risk_value(core,b+eps*direction)-risk_value(core,b-eps*direction))/(2*eps)
            torch.testing.assert_close(fd,(grad*direction).sum(),atol=1e-8,rtol=1e-6)
        with pytest.raises(ValueError):bank.recenter(t,delta,'bad')
        with pytest.raises(ValueError):bank.register(t,c,{'q':0,'v':0},n,f'c{t}')


def test_reference_center_and_shared_uniform_core_jensen():
    torch.manual_seed(3);a=torch.linalg.qr(torch.randn(5,2,dtype=torch.float64)).Q.T
    banks={k:{'0':InputHistory(5,5)} for k in range(2)}
    for k,n in enumerate([2,19]):
        x=torch.randn(n,5,dtype=torch.float64);bank=banks[k]['0'];z=float(x.square().sum())
        bank.recenter(0,{'q':torch.zeros(5,5),'v':torch.zeros(5,5)},'one');bank.register(0,x.T@x,{'q':z,'v':z},n,'one')
    cores,metadata=common_core(banks,{'0':a})
    expected=sum(banks[k]['0'].project(a,'q')['H'] for k in banks)/4
    torch.testing.assert_close(cores[0]['H'],expected)
    assert all(x['client_weight']==.5 and x['layer_weight']==.5 for x in metadata)
    b1,b2=torch.randn(5,2,dtype=torch.float64),torch.randn(5,2,dtype=torch.float64)
    assert risk_value(cores[0],.2*b1+.8*b2)<=.2*risk_value(cores[0],b1)+.8*risk_value(cores[0],b2)+1e-12
    torch.testing.assert_close((.2*b1+.8*b2)@a,.2*(b1@a)+.8*(b2@a))


def test_analytic_qp_and_uniform_scaling_counterexample():
    g=torch.diag(torch.tensor([2.,1.],dtype=torch.float64));h=torch.diag(torch.tensor([100.,1.],dtype=torch.float64))
    d,info=solve_spectral_qp(g,h,1.,config=CFG)
    torch.testing.assert_close(d,torch.diag(torch.tensor([-.02,-1.],dtype=torch.float64)),atol=1e-9,rtol=1e-9)
    assert info['converged'] and abs(info['objective']+.52)<1e-10
    assert float(objective(g,h,-torch.eye(2,dtype=torch.float64)))==47.5
    alpha=3/101;assert abs(float(objective(g,h,-alpha*torch.eye(2,dtype=torch.float64)))+9/202)<1e-12


@pytest.mark.parametrize('mode',['singular','anisotropic','zero_h','zero_g','zero_eta','tiny_eta'])
def test_solver_degenerate_and_feasible(mode):
    torch.manual_seed(6);g=torch.randn(7,3,dtype=torch.float64);h=torch.diag(torch.tensor([0.,1.,20.],dtype=torch.float64));eta=.4
    if mode=='anisotropic':h=torch.diag(torch.tensor([1e-8,1.,1e4],dtype=torch.float64))
    if mode=='zero_h':h.zero_()
    if mode=='zero_g':g.zero_()
    if mode=='zero_eta':eta=0
    if mode=='tiny_eta':eta=1e-12
    d,info=solve_spectral_qp(g,h,eta,config=CFG)
    assert float(torch.linalg.matrix_norm(d,ord=2))<=eta+1e-12
    assert info['objective']<=1e-12
    assert info['converged'] or info['iterations']==CFG.max_iter
    if mode in ['zero_g','zero_eta']:assert not torch.count_nonzero(d)


def test_warm_transport_new_q_new_radius_and_bad_objective():
    torch.manual_seed(19);g=torch.randn(8,3,dtype=torch.float64);h=torch.eye(3,dtype=torch.float64)
    d0,info=solve_spectral_qp(g,h,.4,config=CFG)
    g2=g+.02*torch.randn_like(g)
    cold,c=solve_spectral_qp(g2,h,.2,config=CFG)
    warm,w=solve_spectral_qp(g2,h,.2,previous_delta=d0,config=CFG)
    assert w['warm_start_used'];torch.testing.assert_close(objective(g2,h,cold),objective(g2,h,warm),atol=1e-9,rtol=1e-8)
    _,bad=solve_spectral_qp(g,h,.4,previous_delta=-d0,config=CFG)
    assert not bad['warm_start_used'] and bad['warm_start_objective']>0


def test_frobenius_singular_nullspace_and_psd_rejection():
    g=torch.diag(torch.tensor([2.,1.],dtype=torch.float64));h=torch.diag(torch.tensor([0.,3.],dtype=torch.float64))
    d,info=solve_frobenius_qp(g,h,.7)
    assert float(d.norm())<=.7+1e-12 and info['mu']>0
    torch.testing.assert_close(g+d@(h+info['mu']*torch.eye(2,dtype=torch.float64)),torch.zeros_like(g),atol=1e-10,rtol=1e-10)
    with pytest.raises(ValueError):solve_spectral_qp(g,-torch.eye(2,dtype=torch.float64),1.)


def test_no_history_exact_polar_actual_radius_and_paired_audit():
    torch.manual_seed(8);m=torch.randn(5,2,dtype=torch.float64);a=torch.linalg.qr(torch.randn(6,2,dtype=torch.float64)).Q.T
    b=torch.zeros_like(m);core={'H':torch.zeros(2,2,dtype=torch.float64),'F':torch.zeros_like(m),'q0':0.}
    exact=propose('mulora_fl_exact_polar',m,b,core,.003,0)[0]
    d=propose('history_spectral',m,b,core,.003,100,solver=CFG)[0]
    torch.testing.assert_close(d,exact)
    torch.testing.assert_close(torch.linalg.svdvals(d@a)[:2],torch.linalg.svdvals(d))
    assert not torch.count_nonzero(compact_polar(torch.zeros_like(m))[0])
    rank1=torch.zeros_like(m);rank1[0,0]=1;assert int(torch.linalg.matrix_rank(compact_polar(rank1)[0]))==1
    before=torch.get_rng_state().clone();audit=paired_update_audit(m,b,core,.003,1,a,CFG)
    assert torch.equal(before,torch.get_rng_state());assert len(audit)==6


def test_center_move_same_actual_weight_and_registration_order():
    torch.manual_seed(45);a=torch.linalg.qr(torch.randn(4,2,dtype=torch.float64)).Q.T
    w=torch.randn(4,4,dtype=torch.float64);x=torch.randn(9,4,dtype=torch.float64);c=x.T@x
    bank=InputHistory(4,4);bank.recenter(0,{'q':w*0,'v':w*0},'zero');bank.register(0,c,{'q':float((x@w.T).square().sum()),'v':1.},9,'zero')
    b=torch.randn(4,2,dtype=torch.float64)*.1;before=risk_value(bank.project(a,'q'),b)
    bank.recenter(1,{'q':b@a,'v':w*0},'next');after=risk_value(bank.project(a,'q'),b*0)
    torch.testing.assert_close(before,after)
    with pytest.raises(ValueError):InputHistory(4,4).register(0,c,{'q':1,'v':1},9,'wrong')


def test_empty_history_floor_and_raw_negative_not_hidden():
    a=torch.eye(3,dtype=torch.float64)[:2];bank=InputHistory(3,3)
    assert bank.project(a,'q')['q0']==0
    bank.recenter(0,{'q':torch.zeros(3,3),'v':torch.zeros(3,3)},'z');bank.register(0,torch.eye(3),{'q':0,'v':0},3,'z')
    assert bank.project(a,'q')['floor_active']
    bank.risks['q']['e']=-1.
    with pytest.raises(ValueError):bank.recenter(1,{'q':torch.zeros(3,3),'v':torch.zeros(3,3)},'bad')
