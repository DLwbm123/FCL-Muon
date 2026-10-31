import unittest
from types import SimpleNamespace
import torch
from research.continual_muon.teacher_eval_draft import evaluate,choose
from research.continual_muon.test_teacher_probe_draft import Student,Teacher


class TeacherEvalTest(unittest.TestCase):
    def test_scoring_no_grad_and_restore(self):
        student=Student().train();teacher=Teacher().train();student.b.grad=torch.full_like(student.b,7.);teacher.p.grad=torch.full_like(teacher.p,3.)
        def loader(raw,ids,batch):
            return [(torch.tensor(ids,dtype=torch.float32).view(-1,1),torch.full((len(ids),),4,dtype=torch.long))] if ids else []
        p=SimpleNamespace(model=student,s={'task':2},manifest={'partitions':{'train':[[],[],[[1,2,3],[4]]] }},
            raw=[object(),None],loader=loader,cfg={'batch':16,'cpt':2},device=torch.device('cpu'),
            budget=lambda:None,check_labels=lambda y:None,test_tf=lambda x:x)
        before=torch.get_rng_state().clone();m,c=evaluate(p,teacher,[[1],[4]])
        self.assertGreater(m['teacher_response'],0);self.assertEqual(c['samples'],2);self.assertEqual(c['student_backward_images'],0)
        self.assertTrue(torch.equal(before,torch.get_rng_state()));self.assertTrue(student.training and teacher.training)
        self.assertEqual(int(student.counter),0);self.assertEqual(int(teacher.counter),0)
        torch.testing.assert_close(student.b.grad,torch.full_like(student.b,7.));torch.testing.assert_close(teacher.p.grad,torch.full_like(teacher.p,3.))
        _,held=evaluate(p,teacher,[[5],[6]],allowed_client_groups=[[5],[6]])
        self.assertEqual(held['samples'],2)
        with self.assertRaises(ValueError):evaluate(p,teacher,[[1],[6]],allowed_client_groups=[[1],[6]])

    def test_selection_rechecks_all_goals_and_exact_fallback(self):
        b=[torch.tensor([[1.]])];c=[torch.tensor([[2.]])];p={'proposed':True}
        before={'teacher_response':1.,'ce_all':2.,'ce_current':3.}
        after={'teacher_response':.9,'ce_all':2.,'ce_current':3.}
        out,r=choose(b,c,p,before,after,1.);self.assertTrue(r['accepted']);self.assertTrue(torch.equal(out[0],c[0]))
        for v in ({**after,'teacher_response':1.}, {**after,'ce_all':2.1}, {**after,'ce_current':3.1}):
            out,r=choose(b,c,p,before,v,1.);self.assertFalse(r['accepted']);self.assertTrue(torch.equal(out[0],b[0]))
        out,r=choose(b,c,p,before,after,.9);self.assertFalse(r['accepted']);self.assertTrue(torch.equal(out[0],b[0]))


if __name__=='__main__':unittest.main()
