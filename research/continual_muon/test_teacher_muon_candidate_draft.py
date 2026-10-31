import unittest
import torch
from research.continual_muon.teacher_muon_candidate_draft import propose


class TeacherMuonCandidateTest(unittest.TestCase):
    def test_physical_descent_and_trust(self):
        o=torch.zeros(2,2);b=torch.eye(2);g=torch.diag(torch.tensor([1.,2.]))
        out,info=propose([o],[b],[g],.25)
        self.assertTrue(info['proposed']);self.assertLess(info['physical_response_dot'],0)
        self.assertLessEqual(info['correction_to_update_norm'],1+1e-6)
        self.assertFalse(torch.equal(out[0],b));self.assertEqual(info['ssb_calls'],1)

    def test_zero_response_and_zero_update_fallback(self):
        o=torch.zeros(2,2);b=torch.eye(2);g=torch.ones(2,2)
        for old,base,grad,loss,reason in [(o,b,g,0.,'zero_response'),(o,o,g,.5,'zero_global_update'),
                                         (o,b,torch.zeros_like(g),.5,'zero_response_gradient')]:
            out,info=propose([old],[base],[grad],loss)
            self.assertFalse(info['proposed']);self.assertEqual(info['reason'],reason);self.assertTrue(torch.equal(out[0],base))

    def test_disagreement_reverts_and_matching_consensus_agrees(self):
        from fcl.math import mulora_ssb
        o=torch.zeros(2,2);b=torch.eye(2);g=torch.diag(torch.tensor([1.,2.]))
        normal,r=propose([o],[b],[g],.25)
        aligned,s=propose([o],[b],[g],.25,local_polar=[mulora_ssb(g)])
        self.assertTrue(s['proposed']);self.assertAlmostEqual(s['consensus_cosine'],1.,places=6)
        torch.testing.assert_close(normal[0],aligned[0])
        out,r=propose([o],[b],[g],.25,local_polar=[-mulora_ssb(g)])
        self.assertEqual(r['reason'],'conflicting_client_consensus');self.assertTrue(torch.equal(out[0],b))

    def test_nonzero_origin_and_invalid_values(self):
        old=torch.tensor([[2.,0.],[0.,3.]]);base=old+torch.eye(2);g=torch.eye(2)
        out,info=propose([old],[base],[g],.2)
        if info['proposed']:
            self.assertLess(info['physical_response_dot'],-1e-12)
            self.assertLessEqual(info['correction_to_update_norm'],1+1e-6)
        else:self.assertTrue(torch.equal(out[0],base))
        with self.assertRaises(ValueError):propose([old],[base],[g],float('nan'))
        with self.assertRaises(ValueError):propose([old],[base],[g],-.1)


if __name__=='__main__':unittest.main()
