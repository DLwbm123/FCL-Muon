import unittest
import torch
from research.continual_muon.consensus_aggregation_draft import aggregate, online_whitener, finish_online


class ConsensusTest(unittest.TestCase):
    def test_shared_muon_direction_preserves_physical_norm(self):
        origin=torch.zeros(2,2,dtype=torch.float64)
        u=torch.diag(torch.tensor([1.,2.],dtype=torch.float64))
        out,r=aggregate(origin,u,[u,u],[.5,.5],torch.eye(2))
        self.assertTrue(r['accepted']);self.assertEqual(r['ssb_calls'],3)
        self.assertAlmostEqual(r['norm_after']/r['norm_before'],1.,places=6)
        self.assertFalse(torch.equal(out,u));self.assertGreater(r['alignment_dot'],0)
        self.assertLessEqual(r['history_after'],r['history_before']+1e-10*max(r['history_before'],1.))

    def test_fp32_rounding_bound_is_explicit(self):
        u=torch.diag(torch.tensor([1.,2.]))
        _,r=aggregate(torch.zeros_like(u),u,[u,u],[.5,.5],torch.eye(2))
        self.assertTrue(r['accepted'])
        self.assertGreater(r['roundoff_history_bound'],0)
        self.assertLessEqual(r['history_after']-r['history_ideal'],r['roundoff_history_bound']+1e-15)

    def test_cancellation_mismatch_and_fallback(self):
        origin=torch.zeros(2,2);u=torch.eye(2)
        out,r=aggregate(origin,origin,[u,-u],[.5,.5],torch.eye(2))
        self.assertFalse(r['accepted']);self.assertEqual(r['reason'],'zero_global_update');self.assertTrue(torch.equal(out,origin))
        with self.assertRaises(ValueError):aggregate(origin,u,[u,u],[.5,.4],torch.eye(2))
        with self.assertRaises(ValueError):aggregate(origin,u,[2*u,2*u],[.5,.5],torch.eye(2))
        out,r=aggregate(origin,u,[u,u],[.5,.5],torch.zeros(2,2))
        self.assertEqual(r['reason'],'empty_history');self.assertTrue(torch.equal(out,u))

    def test_client_permutation_and_basis_rotation(self):
        origin=torch.zeros(2,2,dtype=torch.float64)
        a=torch.diag(torch.tensor([1.,2.],dtype=torch.float64));b=torch.diag(torch.tensor([2.,1.],dtype=torch.float64))
        weights=[.25,.75];base=weights[0]*a+weights[1]*b;h=torch.eye(2,dtype=torch.float64)
        one,r1=aggregate(origin,base,[a,b],weights,h)
        two,r2=aggregate(origin,base,[b,a],weights[::-1],h)
        self.assertEqual(r1['accepted'],r2['accepted']);torch.testing.assert_close(one,two)
        R=torch.tensor([[0.,-1.],[1.,0.]],dtype=torch.float64)
        rotated,r3=aggregate(origin@R,base@R,[a@R,b@R],weights,R.T@h@R)
        self.assertEqual(r1['accepted'],r3['accepted']);torch.testing.assert_close(rotated,one@R,atol=1e-6,rtol=1e-6)

    def test_online_sums_match_direct_reference(self):
        from fcl.math import mulora_ssb
        origin=torch.zeros(2,2,dtype=torch.float64)
        a=torch.diag(torch.tensor([1.,2.],dtype=torch.float64));b=torch.diag(torch.tensor([2.,1.],dtype=torch.float64))
        h=torch.diag(torch.tensor([2.,1.],dtype=torch.float64));weights=[.25,.75];base=.25*a+.75*b
        for use_history in (True,False):
            whiten=online_whitener(h,use_history)
            q=sum(w*mulora_ssb((u@whiten).float()).double() for w,u in zip(weights,[a,b]))
            online,r=finish_online(origin,base,base,q,h,local_ssb_calls=2)
            direct,d=aggregate(origin,base,[a,b],weights,h,use_history=use_history)
            self.assertEqual(r['accepted'],d['accepted']);self.assertEqual(r['reason'],d['reason'])
            torch.testing.assert_close(online,direct,atol=1e-12,rtol=1e-12)

    def test_nonzero_origin_and_history_ablation_are_budgeted(self):
        origin=torch.tensor([[10000.,0.],[0.,10000.]])
        u=torch.diag(torch.tensor([1.,2.]));base=origin+u
        history=torch.diag(torch.tensor([1.,4.]))
        for enabled in (True,False):
            out,r=aggregate(origin,base,[u,u],[.5,.5],history,use_history=enabled)
            if r['accepted']:
                self.assertLessEqual(abs(r['norm_after']/r['norm_before']-1),1e-6)
                self.assertLessEqual(r['history_ideal'],r['history_before']+1e-10*max(r['history_before'],1.))
                self.assertLessEqual(r['history_after'],r['history_before']+1e-10*max(r['history_before'],1.)+r['roundoff_history_bound']+1e-15)
            else:self.assertTrue(torch.equal(out,base))


if __name__=='__main__':unittest.main()
