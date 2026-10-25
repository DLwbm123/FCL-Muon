import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from fcl.model import QVAdapter
from fcl.history_extension import HistoricalCoordinator
from fcl.checkpoint import atomic_save,rng_state,restore_rng


def test_collector_rejects_old_train_indices_before_access():
    cfg={'method':'history_spectral','clients':1,'batch':2}
    ext=HistoricalCoordinator(cfg)
    manifest={'cpt':2,'partitions':{'train':[[[0,1]],[[0,2]]]},'labels':{'train':[0,1,2]}}
    model=SimpleNamespace(adapters={'0':QVAdapter(torch.nn.Linear(3,9),2)})
    ext.attach(model,[None,None],manifest,None,None,lambda *a,**kw:None,None)
    with pytest.raises(ValueError):ext.before_merge(1)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA lifecycle check')
def test_actual_optimizer_bank_cache_rng_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(91);np.random.seed(91)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.block=QVAdapter(torch.nn.Linear(6,18),2);self.adapters={'0':self.block}
            self.block.start(torch.linalg.qr(torch.randn(6,2)).Q.T)
        def b_parameters(self):return [self.block.Bq,self.block.Bv]
    model=Model().cuda();cfg={'method':'history_spectral','clients':2,'lambda_history':.5,'paired_audit':False}
    events=[]
    ext=HistoricalCoordinator(cfg);ext.attach(model,None,None,None,None,lambda *a,**kw:events.append((a,kw)),None)
    for k in range(2):
        bank=ext.banks[k]['0'];x=torch.randn(7+k,6,dtype=torch.float64)
        bank.recenter(0,{'q':torch.zeros(6,6),'v':torch.zeros(6,6)},'c0');bank.register(0,x.T@x,{'q':12.,'v':20.},len(x),'c0')
    ext.begin_task(1)
    parameters=model.b_parameters();m=[torch.zeros_like(p) for p in parameters];controls=[torch.zeros_like(p) for p in parameters]
    opt=ext.optimizer(parameters,m,controls,controls,0)
    for p in parameters:p.grad=torch.randn_like(p)
    opt.step(.001)
    assert ext.caches[0][0] is not None
    snapshot={'model':copy.deepcopy(model.state_dict()),'research':ext.checkpoint_state(),'m':copy.deepcopy(m),'rng':rng_state()}
    atomic_save(snapshot,tmp_path/'state.pt')
    for p in parameters:p.grad=torch.randn_like(p)*np.random.rand()
    expected=opt.step(.0007);expected_m=copy.deepcopy(m)
    state=torch.load(tmp_path/'state.pt',map_location='cpu',weights_only=False)
    other=Model().cuda();other.load_state_dict(state['model'])
    restored=HistoricalCoordinator(cfg);restored.attach(other,None,None,None,None,lambda *a,**kw:None,None);restored.restore_state(state['research'])
    restored_m=[v.cuda() for v in state['m']];other_opt=restored.optimizer(other.b_parameters(),restored_m,controls,controls,0)
    restore_rng(state['rng'])
    for p in other.b_parameters():p.grad=torch.randn_like(p)*np.random.rand()
    actual=other_opt.step(.0007)
    for x,y in zip(actual,expected):torch.testing.assert_close(x,y,atol=2e-7,rtol=1e-5)
    for x,y in zip(restored_m,expected_m):torch.testing.assert_close(x,y)
    assert restored.banks[0]['0'].registered=={0}
    old_c=restored.banks[0]['0'].C.clone()
    restored.begin_task(2)
    assert all(v is None for v in restored.caches[0]);torch.testing.assert_close(restored.banks[0]['0'].C,old_c)
    assert ext.optimizer(parameters,m,controls,controls,1).ext.cores is opt.ext.cores
