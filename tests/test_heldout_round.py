import random
import pytest
import torch
from research.fed_spectra.heldout_round import split_clients,GuardedLoader


def test_split_deterministic_disjoint_and_small_clients():
    groups=[[],[0],[1,2],[3,4,5,6,7],list(range(8,18))];state=random.getstate()
    a,b=split_clients(groups,2)
    assert random.getstate()==state and (a,b)==split_clients(groups,2)
    assert list(map(len,b))==[0,0,1,1,2]
    assert list(map(len,a))==[0,1,1,4,8]
    assert not set(sum(a,[]))&set(sum(b,[])) and sorted(sum(a+b,[]))==list(range(18))
    with pytest.raises(ValueError):split_clients([[0],[0]],0)


def test_training_and_validation_gradient_guards():
    raw=object();load=lambda raw,ids,batch,shuffle:[(torch.ones(len(ids),2),torch.zeros(len(ids),dtype=torch.long))]
    guard=GuardedLoader(load,raw,[0,1],[2,3]);model=torch.nn.Linear(2,2)
    handle=model.register_forward_pre_hook(guard.model_guard)
    with pytest.raises(AssertionError):guard(raw,[2],1)
    x,y=next(iter(guard(raw,[0,1],2,True)));model(x).sum().backward()
    guard.allow_validation=True
    with pytest.raises(AssertionError):guard(raw,[1,2],2)
    with pytest.raises(AssertionError):guard(raw,[2],1,True)
    with pytest.raises(AssertionError):guard(raw,[5],1)
    x,y=next(iter(guard(raw,[2,3],2)))
    with pytest.raises(AssertionError):model(x)
    with torch.no_grad():model(x)
    assert guard.validation_model_forward_images==2 and guard.training_access_images==2 and guard.validation_access_images==2
    handle.remove()
