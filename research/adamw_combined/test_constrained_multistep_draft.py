import unittest
import torch
from research.adamw_combined.constrained_multistep_draft import constrained_candidate


class ConstrainedCandidateTest(unittest.TestCase):
    def test_feasible_improvement_and_budget(self):
        pool=torch.eye(3); baseline=pool[:1].clone()
        history=torch.diag(torch.tensor([1.,3.,0.]))
        moment=torch.diag(torch.tensor([0.,9.,4.]))
        candidate, rec=constrained_candidate(pool,baseline,history,moment)
        self.assertTrue(rec['accepted'])
        self.assertGreater(rec['utility'],rec['baseline_utility'])
        self.assertLessEqual(rec['history_response'],rec['history_budget']+1e-5)
        self.assertFalse(torch.equal(candidate,baseline))

    def test_no_gain_exact_fallback_and_bad_inputs(self):
        pool=torch.eye(2); baseline=pool[:1].clone(); history=torch.eye(2)
        candidate, rec=constrained_candidate(pool,baseline,history,torch.diag(torch.tensor([2.,1.])))
        self.assertFalse(rec['accepted'])
        self.assertTrue(torch.equal(candidate,baseline))
        with self.assertRaises(ValueError):
            constrained_candidate(pool,baseline,history,torch.full((2,2),float('nan')))


class EndpointMomentTest(unittest.TestCase):
    def test_heterogeneous_cancellation_and_weighting(self):
        from research.adamw_combined.constrained_multistep_draft import endpoint_moment
        e=torch.tensor([[[[0.,2.]]], [[[0.,-2.]]]])
        torch.testing.assert_close(endpoint_moment(e,[1,1],'mean'),torch.zeros(2,2,dtype=torch.float64))
        torch.testing.assert_close(endpoint_moment(e,[1,1],'individual'),torch.diag(torch.tensor([0.,4.],dtype=torch.float64)))
        unequal=endpoint_moment(e,[3,1],'mean')
        self.assertEqual(float(unequal[1,1]),1.)
        with self.assertRaises(ValueError): endpoint_moment(e,[0,0],'mean')
        with self.assertRaises(ValueError): endpoint_moment(e,[1,1],'unknown')


if __name__=='__main__':unittest.main()
