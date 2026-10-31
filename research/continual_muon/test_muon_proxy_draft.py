import unittest
import torch
from research.continual_muon.muon_proxy_draft import discrepancy,history_damped_direction


class MuonProxyTest(unittest.TestCase):
    def test_same_state_and_linear_operator_cancel(self):
        m=torch.tensor([[1.,2.],[0.,3.]])
        torch.testing.assert_close(discrepancy([m,m],[.25,.75]),torch.zeros_like(m,dtype=torch.float64),atol=1e-7,rtol=0)
        torch.testing.assert_close(discrepancy([m,-m],[.25,.75],operator=lambda x:2*x),torch.zeros_like(m,dtype=torch.float64))
        with self.assertRaises(ValueError):discrepancy([m,m],[.25,.5])

    def test_history_damping_degenerate_and_finite(self):
        a=torch.tensor([[1.,0.],[0.,0.]])
        b=torch.tensor([[0.,1.],[0.,0.]])
        direction,receipt=history_damped_direction([a,b],[.5,.5],torch.eye(2))
        self.assertEqual(receipt['reason'],'computed');self.assertTrue(torch.isfinite(direction).all())
        zero,receipt=history_damped_direction([a,b],[.5,.5],torch.zeros(2,2))
        self.assertEqual(receipt['reason'],'empty_history');self.assertEqual(torch.count_nonzero(zero),0)
        with self.assertRaises(ValueError):history_damped_direction([a,b],[.5,.5],-torch.eye(2))


if __name__=='__main__':unittest.main()
