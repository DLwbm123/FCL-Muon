import unittest
from types import SimpleNamespace
import torch
from research.continual_muon.teacher_probe_draft import probe


class Student(torch.nn.Module):
    def __init__(self):
        super().__init__();self.b=torch.nn.Parameter(torch.tensor([[.3,-.2,.5,.1,.9,-.4]]));self.register_buffer('counter',torch.tensor(0))
    def b_parameters(self):return [self.b]
    def forward(self,x):self.counter.add_(1);torch.rand(1);return x@self.b


class Teacher(torch.nn.Module):
    def __init__(self):
        super().__init__();self.p=torch.nn.Parameter(torch.tensor([[.1,.2,-.1,.4]]));self.register_buffer('counter',torch.tensor(0))
    def forward(self,x):self.counter.add_(1);torch.rand(1);return x@self.p


class TeacherProbeTest(unittest.TestCase):
    def test_probe_cost_and_restoration(self):
        student=Student().train();teacher=Teacher().train();student.b.grad=torch.full_like(student.b,7.);teacher.p.grad=torch.full_like(teacher.p,3.)
        def loader(raw,ids,batch):
            return [(torch.ones(len(ids),1),torch.full((len(ids),),4,dtype=torch.long))] if ids else []
        p=SimpleNamespace(model=student,s={'task':2},manifest={'partitions':{'train':[[],[],[[1,2],[3,4]]] }},
            raw=[object(),None],loader=loader,cfg={'batch':16,'cpt':2},device=torch.device('cpu'),
            budget=lambda:None,check_labels=lambda y:None,test_tf=lambda x:x)
        before=torch.get_rng_state().clone();g,loss,cost=probe(p,teacher,[[1],[3]])
        self.assertEqual(cost['samples'],2);self.assertEqual(cost['student_backward_images'],2)
        self.assertEqual(cost['teacher_forward_images'],2);self.assertEqual(cost['weight_coverage'],1.)
        self.assertGreater(float(g[0].abs().sum()),0);self.assertGreater(loss['teacher_response'],0)
        self.assertTrue(torch.equal(before,torch.get_rng_state()));self.assertTrue(student.training and teacher.training)
        self.assertEqual(int(student.counter),0);self.assertEqual(int(teacher.counter),0)
        torch.testing.assert_close(student.b.grad,torch.full_like(student.b,7.));torch.testing.assert_close(teacher.p.grad,torch.full_like(teacher.p,3.))
        with self.assertRaises(ValueError):probe(p,teacher,[[999],[3]])

    def test_client_sample_weights(self):
        from research.continual_muon.teacher_response_draft import old_response_loss
        student=Student();teacher=Teacher()
        def loader(raw,ids,batch):
            return [(torch.tensor(ids,dtype=torch.float32).view(-1,1),torch.full((len(ids),),4,dtype=torch.long))] if ids else []
        p=SimpleNamespace(model=student,s={'task':2},manifest={'partitions':{'train':[[],[],[[1,2,3],[4]]] }},
            raw=[object(),None],loader=loader,cfg={'batch':16,'cpt':2},device=torch.device('cpu'),
            budget=lambda:None,check_labels=lambda y:None,test_tf=lambda x:x)
        aggregate,polar,loss,cost=probe(p,teacher,[[1],[4]],include_local_polar=True)
        one_grad,_,_=probe(p,teacher,[[1],[]]);four_grad,_,_=probe(p,teacher,[[],[4]])
        from fcl.math import mulora_ssb
        torch.testing.assert_close(aggregate[0],.75*one_grad[0]+.25*four_grad[0],rtol=1e-12,atol=1e-12)
        expected=.75*mulora_ssb(one_grad[0].float()).double()+.25*mulora_ssb(four_grad[0].float()).double()
        torch.testing.assert_close(polar[0],expected,rtol=1e-12,atol=1e-12)
        self.assertEqual(cost['local_ssb_calls'],2)
        with torch.no_grad():
            one,_=old_response_loss(student(torch.ones(1,1))[:,:4],teacher(torch.ones(1,1)))
            four,_=old_response_loss(student(torch.full((1,1),4.))[:,:4],teacher(torch.full((1,1),4.)))
        self.assertAlmostEqual(loss['teacher_response'],.75*float(one)+.25*float(four),places=12)
        self.assertEqual(cost['samples'],2)


if __name__=='__main__':unittest.main()
