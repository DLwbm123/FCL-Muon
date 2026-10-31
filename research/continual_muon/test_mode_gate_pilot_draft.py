import copy
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from research.head_regmean.pilot import Pilot
from research.continual_muon.mode_gate_pilot_draft import ModeGatePilot


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.q=torch.nn.Parameter(torch.tensor([[1.,0.],[0.,1.]]));self.v=torch.nn.Parameter(torch.tensor([[2.,0.],[0.,2.]]))
        self.adapters={'x':SimpleNamespace(A=torch.eye(2))}
    def b_parameters(self):return [self.q,self.v]


def fake_client(p):
    k=p.s['next_client'];base=[v.detach().clone() for v in p.model.b_parameters()]
    with torch.no_grad():
        for j,v in enumerate(p.model.b_parameters()):v.add_(float((k+1)*(j+1)))
    p.local_gram(k)
    with torch.no_grad():
        for v,old in zip(p.model.b_parameters(),base):v.copy_(old)
    p.s['next_client']+=1
    if p.s['next_client']==2:p.s['phase']='local_done'


def fake_fuse(p):
    with torch.no_grad():
        for v,mean in zip(p.model.b_parameters(),p.s['mode_moments']['sums']):v.add_(mean.to(v.dtype))
    p.s.update(round=p.s['round']+1,phase='round_start')


class ModeGatePilotTest(unittest.TestCase):
    def test_real_endpoint_observation_isolated_and_checkpointable(self):
        p=object.__new__(ModeGatePilot);p.model=Tiny();p.cfg={'clients':2};p.mode_variant='full'
        p.manifest={'partitions':{'train':[[],[],[[1],[2,3,4]]]}}
        p.s={'task':2,'round':0,'phase':'clients','next_client':0,'mode_variant':'full',
             'history':SimpleNamespace(raw={'x':torch.eye(2)}),'pending':{'x':2*torch.eye(2)}}
        p.budget=lambda:None
        before=torch.get_rng_state().clone();old=[v.detach().clone() for v in p.model.b_parameters()]
        with patch.object(Pilot,'train_client',fake_client),patch.object(Pilot,'local_gram',lambda p,k:torch.eye(2)),patch.object(Pilot,'fuse',fake_fuse):
            p.train_client()
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
            self.assertTrue(all(torch.equal(v,b) for v,b in zip(p.model.b_parameters(),old)))
            buf=io.BytesIO();torch.save(p.s,buf);buf.seek(0);p.s=torch.load(buf,weights_only=False)
            p.train_client()
            moments=copy.deepcopy(p.s['mode_moments'])
            torch.testing.assert_close(moments['sums'][0],torch.full((2,2),1.75,dtype=torch.float64))
            torch.testing.assert_close(moments['seconds'][0],torch.full((2,2),6.5,dtype=torch.float64))
            self.assertEqual(moments['client_count'],2);self.assertAlmostEqual(moments['weight'],1.)
            p.fuse()
        self.assertEqual(p.s['phase'],'round_start');self.assertNotIn('mode_moments',p.s)
        self.assertEqual(len(p.s['mode_gate_records'][0]['layers']),2)
        self.assertEqual(p.s['mode_gate_records'][0]['theoretical_client_second_moment_bytes'],128)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))


if __name__=='__main__':unittest.main()
