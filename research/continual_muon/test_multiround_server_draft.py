import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from research.continual_muon.multiround_server_draft import MainlinePilot
from research.head_regmean.pilot import Pilot


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.q=torch.nn.Parameter(torch.tensor([[1.,0.]]));self.v=torch.nn.Parameter(torch.tensor([[0.,1.]]))
        self.adapters={'x':SimpleNamespace(A=torch.eye(2))}
    def b_parameters(self):return [self.q,self.v]


class MultiRoundIntegrationTest(unittest.TestCase):
    def test_exact_fallback_after_true_baseline_fuse(self):
        p=object.__new__(MainlinePilot);p.model=TinyModel();p.cfg={'clients':2}
        p.manifest={'partitions':{'train':[[],[],[[1,2],[3,4]]]}}
        p.s={'task':2,'phase':'local_done','round':0,'client_states':{0:{'m':[torch.zeros(1,2)]*2},1:{'m':[torch.zeros(1,2)]*2}},
             'history':SimpleNamespace(raw={'x':torch.eye(2)})}
        def fuse(pilot):
            with torch.no_grad():pilot.model.q.add_(1);pilot.model.v.add_(2)
            pilot.s.update(round=1,phase='round_start')
        grads=[torch.zeros(1,2,dtype=torch.float64)]*2
        def probe(*args):return grads,grads,{'samples':4,'student_forward_images':4,'student_backward_images':8}
        def fake_step(updates,*args,**kwargs):return updates,{'accepted':False,'reason':'empty_or_unsolved_cone'}
        with patch.object(Pilot,'fuse',fuse),patch('research.continual_muon.multiround_server_draft.gradients',probe), \
             patch('research.continual_muon.multiround_server_draft.actual_step',fake_step):
            p.budget=lambda:None;p.fuse()
        torch.testing.assert_close(p.model.q,torch.tensor([[2.,1.]]))
        torch.testing.assert_close(p.model.v,torch.tensor([[2.,3.]]))
        self.assertEqual(len(p.s['mainline_server_records']),1)
        self.assertFalse(p.s['mainline_server_records'][0]['decision']['accepted'])
        self.assertEqual(p.s['mainline_server_records'][0]['fit_probe_cost']['samples'],4)


if __name__=='__main__':unittest.main()
