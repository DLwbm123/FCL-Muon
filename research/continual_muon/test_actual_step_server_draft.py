import unittest
import torch
from research.continual_muon.actual_step_server_draft import _project_cone, actual_step, deployed_parameters


class ActualStepTest(unittest.TestCase):
    def test_projection_and_kkt(self):
        q=torch.tensor([1.,2.,3.],dtype=torch.float64)
        gs=[torch.tensor(v,dtype=torch.float64) for v in ([1,0,0],[0,1,0],[0,0,1])]
        result,info=_project_cone(q,gs)
        self.assertTrue(info['solved']);torch.testing.assert_close(result,torch.zeros_like(q))
        self.assertEqual(info['active'],[0,1,2])
        q=torch.tensor([1.,-2.,3.],dtype=torch.float64)
        result,info=_project_cone(q,[gs[0],2*gs[0],gs[2]])
        torch.testing.assert_close(result,torch.tensor([0.,-2.,0.],dtype=torch.float64))

    def test_actual_deployment_and_fallback(self):
        u=torch.tensor([[1.,0.],[0.,0.]],dtype=torch.float32)
        q=torch.tensor([[-1.,-1.],[-1.,0.]],dtype=torch.float32)
        ga=torch.tensor([[0.,1.],[0.,0.]])
        gc=torch.tensor([[0.,0.],[1.,0.]])
        out,info=actual_step([u],[q],[ga],[gc],[torch.eye(2)])
        self.assertTrue(info['accepted']);self.assertLess(info['all_directional'],0)
        self.assertLessEqual(info['current_directional'],1e-12)
        self.assertLessEqual(info['proposal_history'],info['baseline_history']+1e-10)
        self.assertLessEqual(info['relative_step_norm'],.1+1e-6)
        out,info=actual_step([u],[torch.zeros_like(q)],[ga],[gc],[torch.eye(2)])
        self.assertFalse(info['accepted']);self.assertTrue(torch.equal(out[0],u))
        with self.assertRaises(ValueError):actual_step([u],[q],[ga],[gc],[-torch.eye(2)])

    def test_physical_fp32_addition_and_exact_fallback(self):
        origin=torch.tensor([[100000000.,0.]],dtype=torch.float32)
        baseline=torch.tensor([[100000008.,0.]],dtype=torch.float32)
        update=torch.tensor([[7.,0.]],dtype=torch.float64)
        ga=torch.tensor([[1.,0.]],dtype=torch.float64)
        gc=torch.zeros_like(ga)
        out,info=deployed_parameters([origin],[baseline],[update],[ga],[gc],[torch.eye(2)])
        self.assertFalse(info['accepted']);self.assertEqual(info['reason'],'post_addition_budget')
        self.assertTrue(torch.equal(out[0],baseline))


if __name__=='__main__':unittest.main()
