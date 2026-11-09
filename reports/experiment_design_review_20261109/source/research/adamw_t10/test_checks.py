"""Selected-route training regression, readouts, recovery and atomic storage."""
import copy
import multiprocessing
import os
import tempfile
import time
import unittest
from pathlib import Path
import torch
from torch.nn import functional as F
from fcl.model import QVAdapter
from fcl.checkpoint import rng_state,restore_rng
from research.route_compare_20261102.pilot import RoutePilot
from research.head_regmean.pilot import Pilot
from .pilot import TenTaskPilot,centered_logits
from .storage import lock,allocated,write


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__();self.num_features=4;self.adapter=QVAdapter(torch.nn.Linear(4,12),2)
    def forward_features(self,x):
        q,_,v=self.adapter(x).chunk(3,dim=-1);return q+v


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.backbone=Backbone();self.adapters={'0':self.backbone.adapter}
        self.rows=torch.nn.ParameterList();self.backbone.requires_grad_(False)
        for b in self.b_parameters():b.requires_grad_(True)
    def b_parameters(self):return [self.backbone.adapter.Bq,self.backbone.adapter.Bv]
    def add_head(self,cpt):
        for row in self.rows:row.requires_grad_(False)
        self.rows.append(torch.nn.Parameter(torch.randn(cpt,4)*.02))
    def forward(self,x):
        z=F.normalize(self.backbone.forward_features(x)[:,0],dim=-1)
        return 4*F.linear(z,F.normalize(torch.cat(list(self.rows)),dim=-1))


def load(raw,ids,batch,shuffle=False):
    ids=[ids[i] for i in (torch.randperm(len(ids)).tolist() if shuffle else range(len(ids)))]
    return [tuple(torch.stack(v) for v in zip(*(raw[j] for j in ids[i:i+batch]))) for i in range(0,len(ids),batch)]


def storage_writer(root):
    for i in range(100):write(root,Path(root)/'job.json',{'i':i})


class Checks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(37)
        self.raw=[(torch.randn(3,4),torch.tensor(t*2+i%2)) for t in range(2) for i in range(12)]
        self.manifest={'labels':{s:[int(y) for _,y in self.raw] for s in ('train','test')},
            'partitions':{s:[[list(range(t*12,t*12+6)),list(range(t*12+6,(t+1)*12))] for t in range(2)] for s in ('train','test')}}
        self.cfg={'rank':2,'clients':2,'cpt':2,'tasks':2,'rounds':2,'epochs':2,'batch':2,'lr':.002,
            'retain_task_checkpoints':False,'retain_round_checkpoints':False,
            'route':{'optimizer':'adamw','mode':'selected','pool_width':3,'split_seed':76103,
            'trial_seed':192017,'sample_cap':32,'short_rounds':2,'steps':[.05,.025,.0125],
            'lambdas':[0.,.001,.01,.1,1.,10.,100.],'trust_radius':1.,'orthogonal_tolerance':1e-5,
            'loss_tolerance':1e-7,'space_tolerance':1e-5,'outer_iterations':2}}

    def build(self,path,cls=TenTaskPilot):
        return cls(Model().eval(),[self.raw,self.raw],copy.deepcopy(self.manifest),load,lambda x:x,lambda x:x,
                   copy.deepcopy(self.cfg),path,{'test':True},{'started':1.,'deadline':1e12})

    def test_selected_training_exact_and_recovery(self):
        with tempfile.TemporaryDirectory() as root:
            os.environ['STUDY_ROOT']=root
            plain=self.build(Path(root)/'plain',RoutePilot)
            initial=copy.deepcopy(plain.model.state_dict());start_rng=rng_state()
            plain.run();plain_rng=rng_state()
            extended=self.build(Path(root)/'extended');extended.model.load_state_dict(initial);restore_rng(start_rng)
            checkpoint=None
            while extended.s['phase']!='complete':
                extended.step()
                if (extended.s['task'],extended.s['round'],extended.s['phase'])==(1,1,'round_start'):
                    checkpoint=copy.deepcopy(torch.load(extended.out/'latest.pt',weights_only=False))
            self.assertIsNotNone(checkpoint)
            self.assertEqual(extended.s['matrix'],plain.s['matrix'])
            self.assertEqual(extended.s['communication'],plain.s['communication'])
            self.assertEqual(extended.s['route_records'][0]['calls'],plain.s['route_records'][0]['calls'])
            self.assertTrue(torch.equal(plain_rng['torch'],rng_state()['torch']))
            for k,v in plain.model.state_dict().items():self.assertTrue(torch.equal(v,extended.model.state_dict()[k]))
            resumed=self.build(Path(root)/'resumed');torch.save(checkpoint,resumed.out/'latest.pt');resumed.restore();resumed.run()
            for k,v in extended.model.state_dict().items():self.assertTrue(torch.equal(v,resumed.model.state_dict()[k]))
            self.assertEqual(resumed.s['merges'],[0,1]);self.assertEqual(resumed.s['history'].committed,{0,1})
            self.assertFalse(hasattr(resumed,'response_server_operator'))
            self.assertIsInstance(resumed.adapter_optimizer(resumed.model.b_parameters(),[]).optimizer,torch.optim.AdamW)

    def test_raw_readout_matches_original_and_centering_fixed(self):
        with tempfile.TemporaryDirectory() as root:
            os.environ['STUDY_ROOT']=root;p=self.build(root)
            while p.s['phase']!='ready_evaluation':p.step()
            a=Pilot.evaluate(p);b=p.evaluate(False)
            for k in a:self.assertEqual(a[k],b[k])
            x=torch.tensor([[7.,9.,1.,4.],[1.,2.,8.,6.]])
            expected=torch.tensor([[-1.,1.,-1.5,1.5],[-.5,.5,1.,-1.]])
            self.assertTrue(torch.equal(centered_logits(x,2),expected))
            self.assertTrue(torch.equal(centered_logits(x+torch.tensor([10.,10.,-7.,-7.]),2),expected))

    def test_storage_scan_during_atomic_replace(self):
        with tempfile.TemporaryDirectory() as root:
            proc=multiprocessing.get_context('spawn').Process(target=storage_writer,args=(root,));proc.start()
            for _ in range(150):
                with lock(root):self.assertGreaterEqual(allocated(root),0)
                time.sleep(.001)
            proc.join(15);self.assertEqual(proc.exitcode,0)


if __name__=='__main__':unittest.main()
