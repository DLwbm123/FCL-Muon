import copy,unittest
import torch
from research.stat_head.statistics import merge as legacy_merge
from .statistics import local_packet,merge,projection,class_wire


class AblationChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(44)
        self.z0=torch.randn(90,12,dtype=torch.float64)
        self.z=self.z0+torch.randn_like(self.z0)*.04;self.y=torch.arange(90)%3;self.R=projection(12,5)
        self.packet=local_packet(self.z0,self.z,self.y,3,self.R,12)[0]
        self.old={'counts':torch.tensor([10.,10.]),'means':torch.randn(2,12,dtype=torch.float64),'scatter':torch.eye(12,dtype=torch.float64)}

    def test_identity_has_no_translation_or_anchor_dependence(self):
        a=merge([self.packet],self.old,self.R,mode='identity',return_map=True)
        other=local_packet(self.z0*20+3,self.z,self.y,3,self.R,12)[0]
        b=merge([other],self.old,self.R,mode='identity',return_map=True)
        self.assertTrue(torch.equal(a[4][0],torch.eye(12,dtype=torch.float64)))
        self.assertEqual(float(a[4][1].abs().max()),0)
        self.assertTrue(torch.equal(a[0]['means'][:2],self.old['means']))
        self.assertTrue(torch.equal(a[1],b[1]))
        stripped={k:self.packet[k] for k in ('counts','means','scatter')}
        minimal=merge([stripped],self.old,self.R,mode='identity')
        self.assertTrue(torch.equal(minimal[1],a[1]))

    def test_full_regression_and_mean_only(self):
        full=merge([self.packet],self.old,self.R,mode='full')
        old=legacy_merge([self.packet],self.old,self.R)
        for k in ('counts','means','scatter'):self.assertTrue(torch.equal(full[0][k],old[0][k]))
        self.assertTrue(torch.equal(full[1],old[1]));self.assertTrue(torch.equal(full[2],old[2]))
        mean=merge([self.packet],self.old,self.R,mode='mean_only')
        ident=merge([self.packet],self.old,self.R,mode='identity')
        self.assertTrue(torch.equal(mean[0]['means'],full[0]['means']))
        self.assertTrue(torch.equal(mean[0]['scatter'],ident[0]['scatter']))

    def test_branch_state_isolation(self):
        original=copy.deepcopy(self.old);rng=torch.get_rng_state().clone()
        results={m:merge([self.packet],self.old,self.R,mode=m)[0] for m in ('identity','full','mean_only')}
        for k in original:self.assertTrue(torch.equal(self.old[k],original[k]))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        results['identity']['means'].zero_()
        self.assertGreater(float(results['full']['means'].abs().sum()),0)

    def test_wire_accounting(self):
        z=torch.randn(20,768);y=torch.arange(20)%10;R=projection(768,64)
        _,wire,_=local_packet(z,z,y,10,R,64)
        self.assertEqual(len(wire),446852)
        self.assertEqual(len(class_wire(wire,768,64,10,'identity')),230524)


if __name__=='__main__':unittest.main()
