import copy
from types import SimpleNamespace
import numpy as np
import torch
from fcl.checkpoint import rng_state
from research.fed_spectra.alignment_diagnostic import select_probes,measure


def test_fixed_probes_and_weighted_objective_gradient():
    groups=[list(range(40)),list(range(40,60))]
    before=torch.get_rng_state().clone();probes=select_probes(groups,1)
    assert probes==select_probes(groups,1) and [len(v) for v in probes]==[32,20]
    assert torch.equal(before,torch.get_rng_state())
    assert all(set(p)<=set(g) for p,g in zip(probes,groups))
    torch.manual_seed(9)
    class Model(torch.nn.Linear):
        def b_parameters(self):return [self.weight]
    model=Model(2,4,bias=False);teacher=copy.deepcopy(model).requires_grad_(False)
    data=torch.tensor([[1.,0.],[0.,1.],[1.,1.],[.5,-1.]])
    labels=torch.tensor([2,3,2,3]);groups=[[0],[1,2,3]]
    p=SimpleNamespace(model=model,raw=[None,None],device=torch.device('cpu'),s={'task':1},
        cfg={'clients':2,'batch':2,'cpt':2},settings={'anchor_weight':1.,'anchor_temperature':2.},
        ids=lambda split,t,k:groups[k],get_teacher=lambda:teacher,test_tf=lambda x:x,budget=lambda:None,
        check_labels=lambda y:None,loader=lambda raw,ids,batch:[(data[ids],labels[ids])])
    state=rng_state();base,g=measure(p,groups,True)
    assert torch.equal(state['torch'],rng_state()['torch']) and np.array_equal(state['numpy'][1],rng_state()['numpy'][1])
    assert abs(base['weighted']['kl'])<1e-6 and base['backward_images']==4
    z=model(data);expected=torch.nn.functional.cross_entropy(z[:,2:],labels-2)
    assert abs(float(expected)-base['weighted']['loss'])<1e-6
    original=model.weight.detach().clone()
    with torch.no_grad():model.weight.add_(g[0].to(model.weight),alpha=-.01)
    changed,_=measure(p,groups)
    assert changed['weighted']['loss']<base['weighted']['loss']
    with torch.no_grad():model.weight.copy_(original)
    repeated,_=measure(p,groups)
    assert abs(repeated['weighted']['loss']-base['weighted']['loss'])<1e-7
