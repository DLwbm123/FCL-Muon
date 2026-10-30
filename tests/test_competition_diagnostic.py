import copy
from types import SimpleNamespace
import torch
import pytest
from research.fed_spectra.competition_diagnostic import competition_terms,measure_competition,gradient_alignment


def test_decomposition_and_old_logit_gradient():
    z=torch.tensor([[3.,2.,1.,0.],[0.,1.,2.,3.]],requires_grad=True);y=torch.tensor([2,3])
    ce,full,competition=competition_terms(z,y,2,2)
    torch.testing.assert_close(full,ce+competition)
    g=torch.autograd.grad(competition,z)[0]
    assert (g[:,:2]>0).all() and (g[:,2:]<0).all()
    with pytest.raises(ValueError):competition_terms(z,y,0,4)


def test_probe_isolation_weights_and_competition_counts():
    class Model(torch.nn.Linear):
        def b_parameters(self):return [self.weight]
    model=Model(2,4,bias=False)
    with torch.no_grad():model.weight.copy_(torch.tensor([[3.,0.],[0.,0.],[2.,0.],[0.,2.]]))
    data=torch.eye(2);labels=torch.tensor([2,3]);groups=[[0],[1]]
    p=SimpleNamespace(model=model,raw=[None,None],device=torch.device('cpu'),s={'task':1},cfg={'clients':2,'batch':2,'cpt':2},
      ids=lambda split,t,k:groups[k],test_tf=lambda x:x,budget=lambda:None,check_labels=lambda y:None,
      loader=lambda raw,ids,batch:[(data[ids],labels[ids])])
    before=copy.deepcopy(model.state_dict());rng=torch.get_rng_state().clone()
    row,grad=measure_competition(p,groups,True)
    assert row['weighted']['current_accuracy']==1 and row['weighted']['all_seen_accuracy']==.5
    assert row['weighted']['correct_current_stolen_fraction']==.5
    assert row['student_forward_images']==row['student_backward_images']==2
    assert model.training and model.weight.grad is None and torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
    ce,full,competition=competition_terms(model(data),labels,2,2)
    expected=torch.autograd.grad(competition,model.weight)[0].double()
    torch.testing.assert_close(grad[0],expected)
    a=gradient_alignment(grad,grad,[model.weight.detach()],[model.weight.detach()-grad[0].float()*.01])
    assert abs(a['objective_competition_gradient_cosine']-1)<1e-12 and a['competition_predicted_delta']<0
