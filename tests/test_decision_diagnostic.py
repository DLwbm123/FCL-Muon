import copy
from types import SimpleNamespace
import torch
from research.fed_spectra.decision_diagnostic import pair_statistics,measure_decision_pair


def fixture():
    a=torch.tensor([[0.,0.,4.,0.],[4.,0.,3.,0.],[0.,0.,4.,1.]])
    b=torch.tensor([[4.,0.,3.,0.],[0.,0.,4.,0.],[5.,0.,4.,1.]])
    return a,b,torch.tensor([2,2,3])


def test_opposite_flips_and_group_contributions():
    a,b,y=fixture();r=pair_statistics(a,b,y,2)
    assert r['transition_counts']==[[0,1,0],[1,0,0],[0,0,1]]
    assert r['counts']['correct_to_wrong']==r['counts']['wrong_to_correct']==1
    assert r['counts']['prediction_changed']==3 and r['counts']['wrong_to_different_wrong']==1
    for j in range(4):
        assert abs(sum(v[j] for v in r['group_delta_sums'])-(r['candidate_metric_sums'][j]-r['baseline_metric_sums'][j]))<1e-6


def test_pair_state_isolation_and_weighted_reconstruction():
    class Model(torch.nn.Linear):
        def b_parameters(self):return [self.weight]
    a,b,y=fixture();model=Model(3,4,bias=False)
    with torch.no_grad():model.weight.copy_(a.T)
    model.weight.grad=torch.ones_like(model.weight)
    groups=[[0,1],[2]];data=torch.eye(3)
    p=SimpleNamespace(model=model,raw=[None,None],device=torch.device('cpu'),s={'task':1},cfg={'clients':2,'batch':1,'cpt':2},
      ids=lambda split,t,k:groups[k],test_tf=lambda x:x,budget=lambda:None,check_labels=lambda y:None,
      loader=lambda raw,ids,batch:[(data[[i]],y[[i]]) for i in ids])
    state=copy.deepcopy(model.state_dict());rng=torch.get_rng_state().clone();grad=model.weight.grad.clone()
    r=measure_decision_pair(p,groups,[a.T],[b.T])
    assert r['images']==3 and r['student_forward_images']==6
    assert abs(r['weighted_fractions']['wrong_to_correct']-1/3)<1e-12
    assert abs(r['weighted_fractions']['correct_to_wrong']-1/3)<1e-12
    assert model.training and torch.equal(rng,torch.get_rng_state()) and torch.equal(model.weight.grad,grad)
    assert all(torch.equal(v,state[k]) for k,v in model.state_dict().items())
    for key in r['baseline_metrics']:
        assert abs(sum(g['metric_delta_contribution'][key] for g in r['groups'])-r['candidate_metrics'][key]+r['baseline_metrics'][key])<1e-6
