"""Codec algebra/wire tests and selected-route lifecycle regressions."""
import copy
import os
import tempfile
import unittest
from pathlib import Path
import torch
from fcl.checkpoint import rng_state,restore_rng
from research.adamw_t10.pilot import TenTaskPilot
import research.adamw_t10.test_checks as reference
from .codec import encode,decode,HEADER
from .pilot import CompressedPilot


class CodecChecks(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(17)

    def test_wire_psd_diagonal_and_diagnostics(self):
        x=torch.randn(40,12,dtype=torch.float64);s=x.T@x
        packet=encode(s,rank=3);a,info=decode(packet)
        self.assertEqual(len(packet),HEADER.size+(12*3+12)*4)
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(a).min()),-1e-10)
        torch.testing.assert_close(a.diag(),s.diag(),rtol=2e-7,atol=1e-10)
        self.assertAlmostEqual(info['relative_fro_error'],float((a-s).norm()/s.norm()),places=14)
        self.assertLessEqual(info['top_rank_trace_fraction'],1.)

    def test_dense_oracle_and_diagonal_spectrum(self):
        x=torch.randn(20,8,dtype=torch.float64);s=x.T@x
        dense,_=decode(encode(s,mode='dense64'));self.assertTrue(torch.equal(dense,s))
        diagonal=torch.diag(torch.arange(1,9,dtype=torch.float64))
        a,_=decode(encode(diagonal,rank=2));torch.testing.assert_close(a,diagonal,rtol=1e-7,atol=1e-7)
        full,_=decode(encode(s,rank=8));torch.testing.assert_close(full,s,rtol=1e-6,atol=1e-6)
        zero,_=decode(encode(torch.zeros(8,8),rank=2));self.assertEqual(int(torch.count_nonzero(zero)),0)

    def test_invalid_payload_and_non_psd_fail_closed(self):
        packet=encode(torch.eye(8),rank=2)
        with self.assertRaises(ValueError):decode(packet[:-1])
        with self.assertRaises(ValueError):decode(packet+b'x')
        with self.assertRaises(ArithmeticError):encode(-torch.eye(8),rank=2)


class LifecycleChecks(unittest.TestCase):
    def setUp(self):
        self.fixture=reference.Checks();self.fixture.setUp()

    def build(self,path,cls=CompressedPilot,mode='lowrank_diag'):
        p=self.fixture.build(path,cls)
        p.cfg['rrs_compression']={'mode':mode,'rank':1}
        return p

    def test_dense_codec_selected_training_exact(self):
        with tempfile.TemporaryDirectory() as root:
            os.environ['STUDY_ROOT']=root
            plain=self.build(Path(root)/'plain',TenTaskPilot)
            initial=copy.deepcopy(plain.model.state_dict());start_rng=rng_state()
            plain.run();end_rng=rng_state()
            wrapped=self.build(Path(root)/'wrapped',mode='dense64')
            wrapped.model.load_state_dict(initial);restore_rng(start_rng);wrapped.run()
            self.assertEqual(plain.s['matrix'],wrapped.s['matrix'])
            self.assertTrue(torch.equal(end_rng['torch'],rng_state()['torch']))
            for k,v in plain.model.state_dict().items():self.assertTrue(torch.equal(v,wrapped.model.state_dict()[k]))
            for k,v in plain.s['history'].raw.items():self.assertTrue(torch.equal(v,wrapped.s['history'].raw[k]))
            self.assertEqual(len(wrapped.s['rrs_compression_records']),4)
            self.assertNotIn('RRS_raw_upload',wrapped.s['communication'])
            for k,v in plain.s['communication'].items():
                if k!='RRS_raw_upload':self.assertEqual(v,wrapped.s['communication'][k])

    def test_compressed_selected_checkpoint_resume(self):
        with tempfile.TemporaryDirectory() as root:
            os.environ['STUDY_ROOT']=root;p=self.build(Path(root)/'active')
            while (p.s['task'],p.s['round'],p.s['phase'])!=(1,1,'round_start'):p.step()
            saved=copy.deepcopy(torch.load(p.out/'latest.pt',weights_only=False));p.run()
            final=copy.deepcopy(p.model.state_dict());rng=rng_state()
            resumed=self.build(Path(root)/'resumed');torch.save(saved,resumed.out/'latest.pt');resumed.restore();resumed.run()
            for k,v in final.items():self.assertTrue(torch.equal(v,resumed.model.state_dict()[k]))
            self.assertTrue(torch.equal(rng['torch'],rng_state()['torch']))
            self.assertEqual(p.s['communication'],resumed.s['communication'])
            self.assertEqual(resumed.s['merges'],[0,1]);self.assertEqual(resumed.s['history'].committed,{0,1})
            records=resumed.s['rrs_compression_records'];self.assertEqual(len(records),4)
            self.assertTrue(all(x['rank']==1 for x in records))
            self.assertEqual(sum(x['wire_bytes'] for x in records),resumed.s['communication']['RRS_compressed_upload'])
            self.assertFalse(hasattr(resumed,'response_server_operator'))


if __name__=='__main__':unittest.main()
