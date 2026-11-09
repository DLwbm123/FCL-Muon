import unittest
import torch
from .statistics import local_packet,merge,projection


class StatisticsChecks(unittest.TestCase):
    def setUp(self):torch.manual_seed(19);torch.set_num_threads(2)

    def test_split_merge_zero_drift_and_bytes(self):
        z=torch.randn(72,12,dtype=torch.float64);y=torch.arange(72)%3;R=projection(12,5)
        full,wire,_=local_packet(z,z,y,3,R,12)
        parts=[local_packet(z[i::3],z[i::3],y[i::3],3,R,12)[0] for i in range(3)]
        sf,Wf,bf,_=merge([full],None,R);ss,Ws,bs,_=merge(parts,None,R)
        self.assertLess(float((sf['scatter']-ss['scatter']).norm()/sf['scatter'].norm()),1e-6)
        self.assertLess(float((Wf-Ws).norm()/Wf.norm()),1e-5)
        self.assertEqual(len(wire),8+3*8+4*(5+12+25+60+36)+44+4*(144+12))
        old={'counts':torch.tensor([10.]),'means':torch.randn(1,12),'scatter':torch.eye(12)}
        out,_,_,_=merge([full],old,R)
        self.assertTrue(torch.allclose(out['means'][:1],old['means'].double(),atol=1e-10))

    def test_affine_transport_and_gradient(self):
        z0=torch.randn(120,9,dtype=torch.float64);R=projection(9,4)
        U=torch.randn(9,4,dtype=torch.float64)*.03;a=torch.randn(9,dtype=torch.float64)*.01
        zr=(z0+z0@R@U.T+a).requires_grad_();y=torch.arange(120)%3
        packet,_,_=local_packet(z0,zr,y,3,R,9,track=True)
        old={'counts':torch.tensor([4.]),'means':torch.randn(1,9),'scatter':torch.eye(9)}
        out,W,b,_=merge([packet],old,R,ridge_u=0,floor=1e-10)
        expected=old['means'].double()@(torch.eye(9)+U@R.T).T+a
        self.assertLess(float((out['means'][:1]-expected).norm()),1e-5)
        grad=torch.autograd.grad(W.square().sum()+b.square().sum(),zr)[0]
        self.assertTrue(torch.isfinite(grad).all());self.assertGreater(float(grad.norm()),0)

    def test_no_query_class_fabrication(self):
        z=torch.randn(12,8);R=projection(8,4)
        p,_,_=local_packet(z,z,torch.zeros(12,dtype=torch.long),3,R,4)
        with self.assertRaisesRegex(ValueError,'missing'):merge([p],None,R)


if __name__=='__main__':unittest.main()
