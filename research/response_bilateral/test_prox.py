import unittest
import torch
from .prox import response_bilateral_prox as prox, geometry, history_spectrum, server_map, effective_norm
from research.route_compare_20261102.frame import graph_basis


class ProxTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(786)
        self.b = torch.randn(4, 2, dtype=torch.float64)
        self.a = torch.linalg.qr(torch.randn(5, 2, dtype=torch.float64)).Q.T
        self.u = torch.linalg.qr(torch.randn(4, 3, dtype=torch.float64)).Q
        self.v = torch.linalg.qr(torch.randn(5, 3, dtype=torch.float64)).Q
        self.w = torch.tensor([1., .7, .2], dtype=torch.float64)
        x = torch.randn(7, 5, dtype=torch.float64)
        self.c = x.T@x

    def mapped(self, b=None, a=None, **kw):
        return prox(self.b if b is None else b, self.a if a is None else a,
                    self.u, self.v, self.w, self.c, **kw)

    def test_dense_hessian_oracle_kkt_penalty_coordinate_and_fedavg(self):
        b, a = self.b, self.a
        for metric in ('response', 'euclidean'):
            j, r, _, _ = geometry(a, self.v, self.w, self.c, metric)
            l = (self.u*self.w[None])@self.u.T
            hessian = torch.kron(torch.eye(4, dtype=b.dtype), r.contiguous()) + .1*torch.kron(l.contiguous(), j.contiguous())
            oracle = torch.linalg.solve(hessian, (b@r).flatten()).reshape(b.shape)
            out = self.mapped(metric=metric)
            torch.testing.assert_close(out, oracle, rtol=1e-10, atol=1e-12)
            residual = (out-b)@r+.1*l@out@j
            print({'metric': metric, 'oracle_relative_error': float((out-oracle).norm()/oracle.norm()),
                   'KKT_relative_residual': float(residual.norm()/(b@r).norm())}, flush=True)
            self.assertLess(float(residual.norm()/(b@r).norm()), 1e-10)
            def penalty(value):
                return ((self.w.sqrt()[:, None]*(self.u.T@value@a@self.v))*self.w.sqrt()[None]).square().sum()
            distance = ((out-b)@r*(out-b)).sum()
            self.assertLessEqual(float(distance+.1*penalty(out)), float(.1*penalty(b))+1e-12)
            self.assertLessEqual(float(penalty(out)), float(penalty(b))+1e-12)
            self.assertLess(float(((torch.eye(4, dtype=b.dtype)-self.u@self.u.T)@(out-b)).norm()), 1e-12)
            o = torch.linalg.qr(torch.randn(2, 2, dtype=b.dtype)).Q
            changed = self.mapped(b@o.T, o@a, metric=metric)
            torch.testing.assert_close(changed@o@a, out@a, rtol=1e-10, atol=1e-12)
            other = torch.randn_like(b)
            torch.testing.assert_close(self.mapped(.3*b+.7*other, metric=metric),
                .3*out+.7*self.mapped(other, metric=metric), rtol=1e-10, atol=1e-12)
        print('dense_oracle_KKT_coordinate_FedAvg: FP64 relative tolerance 1e-10', flush=True)

    def test_zero_boundaries_and_nontrivial_zero_B_jacobian(self):
        self.assertIs(self.mapped(gamma=0), self.b)
        self.assertIs(prox(self.b, self.a, self.u[:, :0], self.v[:, :0], self.w[:0], self.c), self.b)
        self.assertIs(prox(self.b, self.a, self.u, self.v, self.w*0, self.c), self.b)
        z = torch.zeros_like(self.b, requires_grad=True)
        out = self.mapped(z)
        self.assertEqual(int(torch.count_nonzero(out)), 0)
        direction = torch.randn_like(z)
        _, actual = torch.autograd.functional.jvp(lambda b: self.mapped(b), z, direction)
        torch.testing.assert_close(actual, self.mapped(direction), rtol=1e-10, atol=1e-12)
        self.assertGreater(float((actual-direction).norm()), 1e-5)
        empty = history_spectrum(torch.zeros(4, 5), 4)
        self.assertEqual(empty['info']['q'], 0)
        rank1 = history_spectrum(torch.ones(4, 5), 4)
        self.assertEqual(rank1['info']['q'], 1)

    def test_gradcheck_A_B_and_chart(self):
        a, b = self.a.clone().requires_grad_(), self.b.clone().requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(lambda aa, bb: self.mapped(bb, aa), (a, b),
                        eps=1e-6, atol=1e-5, rtol=1e-4))
        da, db, weight = torch.randn_like(a), torch.randn_like(b), torch.randn_like(b)
        value = (self.mapped(b, a)*weight).sum()
        ga, gb = torch.autograd.grad(value, (a, b))
        step = 1e-6
        finite = ((self.mapped(b.detach()+step*db, a.detach()+step*da)*weight).sum()
                 -(self.mapped(b.detach()-step*db, a.detach()-step*da)*weight).sum())/(2*step)
        analytic = (ga*da).sum()+(gb*db).sum()
        print({'total_AB_directional_derivative_absolute_error': float((analytic-finite).abs())}, flush=True)
        e = torch.eye(5, dtype=torch.float64)
        z0, n = e[:, :2], e[:, 2:]
        x = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)*.1
        self.assertTrue(torch.autograd.gradcheck(lambda xx: self.mapped(a=graph_basis(e, z0, n, xx)),
                                               (x,), eps=1e-6, atol=1e-5, rtol=1e-4))

    def test_scalar_control_dtype_and_invalid_geometry(self):
        mem = {'U': self.u, 'V': self.v, 'alpha': self.w}
        shadow, _ = server_map(self.b, self.a, mem, self.c, 'E3')
        out, c = server_map(self.b, self.a, mem, self.c, 'E1')
        torch.testing.assert_close(effective_norm(out, self.a), effective_norm(shadow, self.a), rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(out, self.b*c)
        self.assertTrue(torch.autograd.gradcheck(lambda aa, bb: server_map(bb, aa, mem, self.c, 'E1')[0],
            (self.a.clone().requires_grad_(), self.b.clone().requires_grad_()), eps=1e-6, atol=1e-5, rtol=1e-4))
        actual = self.mapped(self.b.float(), self.a.float())
        expected = self.mapped(self.b.float().double(), self.a.float().double()).float()
        self.assertTrue(torch.equal(actual, expected))
        print({'FP32_cast_max_abs_error': float((actual-expected).abs().max())}, flush=True)
        with self.assertRaises(ArithmeticError):
            prox(self.b, self.a, self.u, self.v, self.w, -self.c)
        with self.assertRaises(ArithmeticError): self.mapped(self.b*float('nan'))
        _, c0 = server_map(torch.zeros_like(self.b), self.a, mem, self.c, 'E1')
        self.assertEqual(c0, 1.)


if __name__ == '__main__': unittest.main()
