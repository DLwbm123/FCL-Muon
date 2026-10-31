import unittest
import torch
from research.continual_muon.teacher_response_draft import old_response_loss


class TeacherResponseTest(unittest.TestCase):
    def test_identical_outputs_have_zero_gradient(self):
        old=torch.tensor([[1.,-2.,.5]],dtype=torch.float64)
        x=old.clone().requires_grad_();loss,parts=old_response_loss(x,old)
        self.assertAlmostEqual(float(loss),0.,places=12)
        torch.testing.assert_close(torch.autograd.grad(loss,x)[0],torch.zeros_like(x),atol=1e-12,rtol=0)
        self.assertAlmostEqual(float(parts['relative_kl']),0.,places=12)

    def test_common_shift_detected_and_teacher_detached(self):
        teacher=torch.tensor([[2.,-.5]],dtype=torch.float64,requires_grad=True)
        x=(teacher.detach()+.4).requires_grad_();loss,parts=old_response_loss(x,teacher)
        self.assertAlmostEqual(float(parts['relative_kl']),0.,places=12)
        self.assertAlmostEqual(float(parts['evidence_shift_sq']),.08,places=12)
        kl,_=old_response_loss(x,teacher,include_evidence=False);self.assertAlmostEqual(float(kl),0.,places=12)
        gx,gt=torch.autograd.grad(loss,(x,teacher),allow_unused=True);self.assertGreater(float(gx.abs().sum()),0.)
        self.assertIsNone(gt)

    def test_permutation_finiteness_and_invalid_inputs(self):
        a=torch.tensor([[5.,-7.,1.],[.2,.1,.4]])
        b=torch.tensor([[4.,-5.,0.],[.1,.3,.5]])
        loss,_=old_response_loss(a,b);perm=torch.tensor([2,0,1])
        loss2,_=old_response_loss(a[:,perm],b[:,perm]);torch.testing.assert_close(loss,loss2)
        self.assertTrue(torch.isfinite(loss))
        with self.assertRaises(ValueError):old_response_loss(a,b[:,:2])
        with self.assertRaises(ValueError):old_response_loss(a,torch.full_like(b,float('nan')))


if __name__=='__main__':unittest.main()
