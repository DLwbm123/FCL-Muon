import random
import unittest
import numpy as np
import torch
from research.adamw_combined.pilot import AdamAdapter, probe, accept_candidate

class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.b=torch.nn.Parameter(torch.tensor([[.2,.1],[-.2,.3]]))
    def b_parameters(self):return [self.b]
    def forward(self, x):return x @ self.b

class Fake:
    def __init__(self):
        self.model=Tiny(); self.device=torch.device('cpu'); self.cfg={'batch':2,'cpt':1}
        self.s={'task':1};self.raw=[None];self.manifest={'partitions':{'train':[[],[[0,1],[2]]]}}
        self.test_tf=lambda x:x
        self.x=torch.tensor([[1.,2.],[2.,-1.],[-1.,2.]])
    def loader(self, raw, ids, batch):
        for i in range(0,len(ids),batch):
            q=ids[i:i+batch];yield self.x[q],torch.ones(len(q),dtype=torch.long)
    def budget(self):pass
    def check_labels(self,y):assert bool((y==1).all())

class Tests(unittest.TestCase):
    def test_candidate_gate(self):
        before={"coverage":1., "all":1., "current":.4}
        self.assertTrue(accept_candidate(before,{"all":.9,"current":.4}))
        self.assertFalse(accept_candidate(before,{"all":1.,"current":.4}))
        self.assertFalse(accept_candidate(before,{"all":.9,"current":.41}))
        self.assertFalse(accept_candidate({**before,"coverage":.94},{"all":.9,"current":.4}))
    def test_adam_adapter_matches_pytorch_multiple_steps(self):
        p=torch.nn.Parameter(torch.tensor([[.1,-.3]]));q=torch.nn.Parameter(p.detach().clone())
        a=AdamAdapter([p]);b=torch.optim.AdamW([q],lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0.)
        for g in (torch.tensor([[.01,-.4]]),torch.tensor([[-.3,.04]])):
            p.grad=g.clone();q.grad=g.clone();a.step(.001);b.step()
        self.assertTrue(torch.equal(p,q))
    def test_probe_weighting_and_state_restoration(self):
        p=Fake(); p.model.train();p.model.b.grad=torch.full_like(p.model.b,7.)
        before=torch.get_rng_state().clone();py=random.getstate();nps=np.random.get_state()
        result, grads=probe(p,[[0,1],[2]],True)
        x=p.x; b=p.model.b
        a=F.cross_entropy(p.model(x[:2]),torch.ones(2,dtype=torch.long),reduction='mean')
        c=F.cross_entropy(p.model(x[2:]),torch.ones(1,dtype=torch.long),reduction='mean')
        expected=torch.autograd.grad((2*a+c)/3,b)[0]
        self.assertTrue(torch.allclose(grads[0],expected.double(),atol=1e-7))
        self.assertAlmostEqual(result['all'],float((2*a+c)/3),places=6)
        self.assertEqual(result['samples'],3);self.assertEqual(result['coverage'],1.)
        self.assertTrue(p.model.training);self.assertTrue(torch.equal(p.model.b.grad,torch.full_like(b,7.)))
        self.assertTrue(torch.equal(before,torch.get_rng_state()));self.assertEqual(py,random.getstate())
        self.assertTrue(np.array_equal(nps[1],np.random.get_state()[1]))

from torch.nn import functional as F
if __name__=='__main__':unittest.main()
