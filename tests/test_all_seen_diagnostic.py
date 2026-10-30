import copy
import torch
from torch.nn import functional as F
from research.fed_spectra.competition_diagnostic import competition_terms
from research.fed_spectra.all_seen_diagnostic import combined_gradient,local_conditions


def test_combined_gradient_equals_direct_all_seen_objective():
    z=torch.tensor([[2.,1.,3.,0.],[1.,0.,2.,4.]],dtype=torch.float64,requires_grad=True);y=torch.tensor([2,3])
    ce,full,c=competition_terms(z,y,2,2)
    target=torch.tensor([[.4,.6],[.7,.3]],dtype=torch.float64)
    kl=F.kl_div(z[:,:2].log_softmax(-1),target,reduction='batchmean')
    g=torch.autograd.grad(ce+.8*kl,z,retain_graph=True)[0]
    h=torch.autograd.grad(c,z,retain_graph=True)[0]
    direct=torch.autograd.grad(full+.8*kl,z)[0]
    torch.testing.assert_close(combined_gradient([g],[h])[0],direct,atol=1e-12,rtol=1e-12)


def rows():
    return [dict(seed=s,matrix_candidate_reproduced=True,heads_unchanged=True,baseline_restored=True,selection_confirmation_disjoint=True,
      all_seen_candidate={'probes':{'confirmation':dict(paired_decisions={'coverage':.99},combined_delta_vs_baseline=-.01,combined_delta_vs_original=-.001,original_objective_delta_vs_baseline=-.001,all_seen_accuracy_delta_vs_baseline=0.)}}) for s in range(3)]


def test_loss_improvement_without_accuracy_cannot_pass():
    r=rows();conditions=local_conditions(r)
    assert not conditions['mean_all_seen_accuracy_strictly_improves'] and not all(conditions.values())
    for x in r:x['all_seen_candidate']['probes']['confirmation']['all_seen_accuracy_delta_vs_baseline']=.001
    assert all(local_conditions(r).values())


def test_bad_seed_cannot_hide_in_positive_mean():
    r=rows()
    for x,a in zip(r,[.1,-.01,.1]):x['all_seen_candidate']['probes']['confirmation']['all_seen_accuracy_delta_vs_baseline']=a
    c=local_conditions(r)
    assert c['mean_all_seen_accuracy_strictly_improves'] and not c['all_seen_accuracy_not_worse_all']
    r=rows();r[1]['all_seen_candidate']['probes']['confirmation']['combined_delta_vs_original']=0.
    assert not local_conditions(r)['combined_objective_improves_original_all']
