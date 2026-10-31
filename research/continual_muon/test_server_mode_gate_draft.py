import unittest
import torch
from research.continual_muon.server_mode_gate_draft import aggregate_update_dispersion,dispersion_from_moments,gated_update,gated_parameter


class ModeGateTest(unittest.TestCase):
    def test_update_dispersion_and_weights(self):
        a=torch.tensor([[1.,0.],[0.,2.]],dtype=torch.float64)
        avg,cov=aggregate_update_dispersion([a,a],[.25,.75])
        torch.testing.assert_close(avg,a);torch.testing.assert_close(cov,torch.zeros(2,2,dtype=torch.float64))
        avg,cov=aggregate_update_dispersion([a,-a],[.5,.5])
        torch.testing.assert_close(avg,torch.zeros_like(a));torch.testing.assert_close(cov,a.T@a)
        avg,cov=aggregate_update_dispersion([a,2*a],[.25,.75])
        second=.25*a.T@a+.75*(2*a).T@(2*a)
        torch.testing.assert_close(dispersion_from_moments(avg,second),cov,atol=1e-12,rtol=1e-12)
        with self.assertRaises(ValueError):aggregate_update_dispersion([a,a],[.25,.5])

    def test_noncommuting_psd_and_rotation_equivariance(self):
        torch.manual_seed(7)
        u=torch.randn(5,3,dtype=torch.float64);m=torch.randn(3,3,dtype=torch.float64)
        n=torch.randn(3,3,dtype=torch.float64);o=torch.randn(3,3,dtype=torch.float64)
        h=m.T@m;s=n.T@n;c=o.T@o
        out,rec=gated_update(u,h,s,c)
        self.assertLessEqual(rec['gate_max_eigenvalue'],1+1e-9)
        if rec['accepted']:
            self.assertLessEqual(rec['norm_after'],rec['norm_before']*(1+1e-6)+1e-12)
            self.assertLessEqual(rec['history_after'],rec['history_before']+1e-10*max(rec['history_before'],1.))
        else:self.assertTrue(torch.equal(out,u))
        R=torch.linalg.qr(torch.randn(3,3,dtype=torch.float64)).Q
        rotated,rrec=gated_update(u@R,R.T@h@R,R.T@s@R,R.T@c@R)
        self.assertEqual(rec['accepted'],rrec['accepted'])
        torch.testing.assert_close(rotated,out@R,atol=1e-9,rtol=1e-9)
        with self.assertRaises(ValueError):gated_update(u,-torch.eye(3),s,c)
        with self.assertRaises(ValueError):gated_update(u,torch.diag(torch.tensor([1e-6,1e-6,-1e-12])),s,c)
        zero,info=gated_update(u,torch.zeros(3,3),s,c)
        self.assertEqual(info['reason'],'empty_required_moment');self.assertTrue(torch.equal(zero,u))

    def test_neutral_history_changes_gate_but_not_final_history_audit(self):
        origin=torch.zeros(1,2);baseline=torch.tensor([[1.,1.]])
        history=torch.diag(torch.tensor([1.,9.]));current=torch.eye(2);cov=torch.zeros(2,2)
        full,frec=gated_parameter(origin,baseline,history,current,cov)
        neutral,nrec=gated_parameter(origin,baseline,history,current,cov,gate_history=torch.eye(2))
        self.assertTrue(frec['accepted']);self.assertTrue(nrec['accepted'])
        self.assertFalse(torch.equal(full,neutral))
        self.assertLessEqual(nrec['history_after'],nrec['history_before']+1e-10*max(nrec['history_before'],1.))

    def test_nonzero_origin_physical_step(self):
        origin=torch.tensor([[100000000.,0.]],dtype=torch.float32)
        base=torch.tensor([[100000008.,0.]],dtype=torch.float32)
        h=torch.eye(2);current=torch.eye(2);dispersion=torch.zeros(2,2)
        result,receipt=gated_parameter(origin,base,h,current,dispersion)
        if receipt['accepted']:
            self.assertLessEqual(receipt['history_after'],receipt['history_before']+1e-10*max(receipt['history_before'],1.))
            self.assertFalse(torch.equal(result,base))
        else:self.assertTrue(torch.equal(result,base))


if __name__=='__main__':unittest.main()
