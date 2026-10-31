import copy
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from research.head_regmean.pilot import Pilot
from research.continual_muon.consensus_pilot_draft import ConsensusPilot


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
        for v,mean in zip(p.model.b_parameters(),p.s['consensus_moments']['mean_updates']):v.add_(mean.to(v.dtype))
    p.s.update(round=p.s['round']+1,phase='round_start')


class ConsensusPilotTest(unittest.TestCase):
    def test_endpoint_capture_rng_checkpoint_and_variants(self):
        for mode in ('full','history_neutral','server_polar'):
            p=object.__new__(ConsensusPilot);p.model=Tiny();p.cfg={'clients':2};p.consensus_variant=mode
            p.manifest={'partitions':{'train':[[],[],[[1],[2,3,4]]]}}
            p.s={'task':2,'round':0,'phase':'clients','next_client':0,'consensus_variant':mode,
                 'history':SimpleNamespace(raw={'x':torch.eye(2)})};p.budget=lambda:None
            before=torch.get_rng_state().clone();old=[v.detach().clone() for v in p.model.b_parameters()]
            with patch.object(Pilot,'train_client',fake_client),patch.object(Pilot,'local_gram',lambda p,k:torch.eye(2)),patch.object(Pilot,'fuse',fake_fuse):
                p.train_client()
                self.assertTrue(torch.equal(before,torch.get_rng_state()))
                self.assertTrue(all(torch.equal(v,b) for v,b in zip(p.model.b_parameters(),old)))
                buf=io.BytesIO();torch.save(p.s,buf);buf.seek(0);p.s=torch.load(buf,weights_only=False)
                p.train_client();state=copy.deepcopy(p.s['consensus_moments'])
                torch.testing.assert_close(state['mean_updates'][0],torch.full((2,2),1.75,dtype=torch.float64))
                self.assertEqual(state['client_count'],2);self.assertAlmostEqual(state['weight'],1.)
                self.assertEqual(state['local_ssb_calls'],[0,0] if mode=='server_polar' else [2,2])
                p.fuse()
            self.assertEqual(p.s['phase'],'round_start');self.assertNotIn('consensus_moments',p.s)
            record=p.s['consensus_records'][0]
            self.assertEqual(record['consensus_variant'],mode);self.assertEqual(len(record['layers']),2)
            self.assertEqual(record['additional_update_upload_bytes'],0)
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
            p.consensus_variant='full' if mode!='full' else 'server_polar'
            with self.assertRaises(ValueError):p._check_variant()


if __name__=='__main__':unittest.main()
