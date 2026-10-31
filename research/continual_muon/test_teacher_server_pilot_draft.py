import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from research.head_regmean.pilot import Pilot
from research.fed_spectra.heldout_round import GuardedLoader
from research.continual_muon.teacher_server_pilot_draft import TeacherServerPilot
from research.continual_muon.test_teacher_probe_draft import Student,Teacher


def fake_fuse(p):
    with torch.no_grad():p.model.b.add_(.1)
    p.s.update(round=p.s['round']+1,phase='round_start')


class TeacherServerPilotTest(unittest.TestCase):
    def test_freeze_precedes_validation_and_rng_restores(self):
        for mode in ('full','no_consensus','kl_only'):
            sizes=[32]*8+[24,1];fits=[];holds=[];next_id=0
            for n in sizes:
                fits.append(list(range(next_id,next_id+n)));next_id+=n
            for k in range(10):
                n=32 if k<7 else (12 if k==7 else 0)
                holds.append(list(range(1000+100*k,1000+100*k+n)))
            raw=object()
            def loader(raw,ids,batch,shuffle=False):
                return [(torch.ones(len(ids),1),torch.full((len(ids),),4,dtype=torch.long))] if ids else []
            guard=GuardedLoader(loader,raw,[i for g in fits for i in g],[i for g in holds for i in g])
            p=object.__new__(TeacherServerPilot);p.model=Student().train();p.teacher=Teacher().train()
            p.model.b.grad=torch.full_like(p.model.b,7.);p.teacher.p.grad=torch.full_like(p.teacher.p,3.)
            p.heldout_groups=holds;p.teacher_variant=mode;p.loader=guard;p.raw=[raw,None]
            p.cfg={'clients':10,'batch':32,'cpt':2};p.device=torch.device('cpu');p.test_tf=lambda x:x
            p.manifest={'partitions':{'train':[[],[],fits]}};p.s={'task':2,'round':0,'phase':'local_done','teacher_variant':mode}
            p.budget=lambda:None;p.check_labels=lambda y:None;events=[]
            p.event=lambda kind,**v:events.append((kind,v,guard.validation_access_images))
            before=torch.get_rng_state().clone()
            with patch.object(Pilot,'fuse',fake_fuse):p.fuse()
            self.assertEqual(len(events),1);self.assertEqual(events[0][0],'teacher_selection_frozen');self.assertEqual(events[0][2],0)
            self.assertEqual(guard.training_access_images,562);self.assertEqual(guard.validation_access_images,472)
            self.assertFalse(guard.allow_validation);self.assertTrue(torch.equal(before,torch.get_rng_state()))
            self.assertTrue(p.model.training and p.teacher.training);self.assertEqual(int(p.model.counter),0);self.assertEqual(int(p.teacher.counter),0)
            torch.testing.assert_close(p.model.b.grad,torch.full_like(p.model.b,7.));torch.testing.assert_close(p.teacher.p.grad,torch.full_like(p.teacher.p,3.))
            record=p.s['teacher_server_records'][0]
            self.assertEqual(record['gradient_cost']['samples'],281);self.assertEqual(record['confirmation_selected_cost']['samples'],236)
            self.assertEqual(record['confirmation_reads_before_selection'],0);self.assertEqual(record['confirmation_reads_after'],472)
            self.assertEqual(record['gradient_cost']['local_ssb_calls'],0 if mode=='no_consensus' else 10)


if __name__=='__main__':unittest.main()
