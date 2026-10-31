import unittest
import torch
from research.adamw_combined.adamw_utility_draft import quantities


class TestAdamUtility(unittest.TestCase):
    def test_matches_individual_adamw_and_weighted_mean(self):
        torch.manual_seed(7)
        c = torch.randn(3,2,4,3, dtype=torch.float64)
        c[0,0,0,0] = 1e-10
        z = torch.eye(3,dtype=torch.float64); weights=torch.tensor([.2,.3,.5],dtype=torch.float64)
        q = quantities(z,c,weights,torch.eye(3,dtype=torch.float64))
        directions=[]
        for g in c:
            p=torch.nn.Parameter(torch.zeros_like(g));o=torch.optim.AdamW([p],lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0)
            p.grad=g.clone();o.step();directions.append(-p.detach()/.001)
        actual=sum(w*d for w,d in zip(weights,directions))
        self.assertTrue(torch.allclose(q['direction'],actual,rtol=1e-12,atol=1e-12))
        self.assertAlmostEqual(float(q['utility']),float((torch.einsum('k,kvdr->vdr',weights,c)*actual).sum()),places=11)
    def test_float32_near_epsilon(self):
        c=torch.tensor([[[[0.,1e-12,1e-8,1.],[-2e-8,-1e-4,.1,-2.]]]],dtype=torch.float32)
        z=torch.eye(4);p=torch.nn.Parameter(torch.zeros_like(c[0]));opt=torch.optim.AdamW([p],lr=.0002,betas=(.9,.999),eps=1e-8,weight_decay=0.)
        p.grad=c[0].clone();opt.step()
        q=quantities(z,c,[1.],torch.eye(4))
        self.assertTrue(torch.allclose(q["direction"],-p.detach()/.0002,atol=2e-7,rtol=2e-6))
    def test_cancellation_zero_and_derivative(self):
        z=torch.eye(2,dtype=torch.float64,requires_grad=True)
        g=torch.tensor([[[[.3,.5]]],[[[-.3,-.5]]]],dtype=torch.float64)
        q=quantities(z,g,[.5,.5],torch.eye(2,dtype=torch.float64))
        self.assertEqual(float(q['update_norm']),0.)
        self.assertEqual(float(q['utility']),0.)
        self.assertTrue(torch.isfinite(torch.autograd.grad(q['utility'],z)[0]).all())
    def test_adam_direction_depends_on_basis_rotation(self):
        z=torch.eye(2,dtype=torch.float64); g=torch.tensor([[[[1.,3.]]]],dtype=torch.float64)
        rotation=torch.tensor([[1.,-1.],[1.,1.]],dtype=torch.float64)/(2.**.5)
        a=quantities(z,g,[1.],torch.eye(2,dtype=torch.float64))["direction"].reshape(2)
        b=quantities(z@rotation,g,[1.],torch.eye(2,dtype=torch.float64))["direction"].reshape(2)
        self.assertGreater(float((z@a-z@rotation@b).norm()),.5)
    def test_fit_search_and_disjoint_check_fallback(self):
        from research.aggregation_utility.selection import fit_search, certify
        torch.manual_seed(2)
        c=torch.randn(3,2,12,4);c[:,:,:,:2]*=.2
        w=torch.tensor([.2,.3,.5]);h=torch.diag(torch.tensor([8.,7.,.1,.2],dtype=torch.float64))
        z0=torch.eye(4)[:,:2];pool=torch.linalg.qr(torch.randn(8,4)).Q
        config=dict(search_steps=25,initial_step=.25,backtracks=16,inward_derivative=.001,
                    constraint_relative_tolerance=2e-6,minimum_relative_gain=1e-4,orthogonal_tolerance=1e-5)
        z,trace=fit_search(c,w,h,z0,"adamw",config)
        a,cert=certify(pool,z,z0,c,w,c,w,h,"adamw",config)
        self.assertGreater(trace["steps_accepted"],0);self.assertTrue(cert["accepted"])
        rejected,bad=certify(pool,z,z0,c,w,-c,w,h,"adamw",config)
        self.assertFalse(bad["accepted"]);self.assertFalse(bad["checks"]["independent_check"])
        self.assertTrue(torch.equal(rejected,(pool@z0).T.contiguous()))
    def test_invalid_weights_and_nonfinite(self):
        z=torch.eye(2);c=torch.ones(2,1,2,2);h=torch.eye(2)
        with self.assertRaises(ValueError):quantities(z,c,[0,0],h)
        c[0,0,0,0]=float('nan')
        with self.assertRaises(ValueError):quantities(z,c,[.5,.5],h)

if __name__ == '__main__': unittest.main()
