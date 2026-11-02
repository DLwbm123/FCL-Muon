import copy
import unittest
from research.adamw_combined.holdout_firewall_draft import GuardedTrainingDataset


class GuardedAccessTests(unittest.TestCase):
    def test_expired_budget_prevents_underlying_read(self):
        class Raw:
            targets = [0, 1]
            reads = 0
            def __len__(self): return 2
            def __getitem__(self, i):
                self.reads += 1
                return i
        raw = Raw()
        guarded = GuardedTrainingDataset(raw, {1: [[1]]})
        def expired(): raise TimeoutError('expired')
        guarded.budget_check = expired
        with self.assertRaises(TimeoutError): guarded[0]
        guarded.freeze_candidate(1)
        with guarded.confirmation(1):
            with self.assertRaises(TimeoutError): guarded[1]
        self.assertEqual(raw.reads, 0)
        self.assertEqual(guarded.holdout_reads, 0)

    def test_freeze_scope_and_exception_close_access(self):
        class Raw:
            targets = [0, 1, 1, 2]
            def __len__(self): return len(self.targets)
            def __getitem__(self, i): return ('synthetic', self.targets[i])
        guarded = GuardedTrainingDataset(Raw(), {1: [[1]], 2: [[3]]})
        self.assertEqual(guarded[0], ('synthetic', 0))
        with self.assertRaises(IndexError): guarded[-1]
        with self.assertRaises(PermissionError): guarded[1]
        with self.assertRaises(ValueError):
            with guarded.confirmation(1): pass
        guarded.freeze_candidate(1)
        with self.assertRaisesRegex(RuntimeError, 'example'):
            with guarded.confirmation(1):
                self.assertEqual(guarded[1], ('synthetic', 1))
                with self.assertRaises(PermissionError): guarded[3]
                raise RuntimeError('example')
        self.assertEqual(guarded.holdout_reads, 1)
        with self.assertRaises(PermissionError): guarded[1]
        guarded.finish_task(1)
        with self.assertRaises(ValueError): guarded.freeze_candidate(1)
        guarded.freeze_candidate(2)
        with guarded.confirmation(2):
            self.assertEqual(guarded[3], ('synthetic', 2))
            with self.assertRaises(PermissionError): guarded[1]
from research.adamw_combined.holdout_firewall_draft import build_firewalled_manifest


class FirewallTest(unittest.TestCase):
    def test_disjoint_complete_deterministic_and_t1_unchanged(self):
        m = {'cpt':2, 'labels':{'train':[0]*4+[1]*4+[2]*4+[3]*4+[4]*4+[5]*4},
             'partitions':{'train':[[[0,1,4,5],[2,3,6,7]],
                                    [[8,9,12,13],[10,11,14,15]],
                                    [[16,17,20,21],[18,19,22,23]]],
                           'test':[[[3]],[[5]],[[7]]]}}
        original = copy.deepcopy(m)
        a, hold = build_firewalled_manifest(m, [1,2], 17, .25)
        b, again = build_firewalled_manifest(m, [1,2], 17, .25)
        self.assertEqual((a,hold),(b,again))
        self.assertEqual(m,original)
        self.assertEqual(a['partitions']['train'][0],m['partitions']['train'][0])
        self.assertEqual(a['partitions']['test'],m['partitions']['test'])
        for task in (1,2):
            for client in (0,1):
                kept = set(a['partitions']['train'][task][client]); reserved = set(hold[task][client])
                self.assertFalse(kept & reserved)
                self.assertEqual(kept | reserved,set(m['partitions']['train'][task][client]))
                self.assertEqual(len(reserved),1)

    def test_invalid_or_uncovered_clients_are_rejected(self):
        m = {'cpt':1, 'labels':{'train':[0,1,1]},
             'partitions':{'train':[[[0]],[[1,2]]]}}
        with self.assertRaises(ValueError): build_firewalled_manifest(m,[0],3,.2)
        with self.assertRaises(ValueError): build_firewalled_manifest(m,[1],3,1.)
        m['partitions']['train'][1][0] = [1]
        with self.assertRaises(ValueError): build_firewalled_manifest(m,[1],3,.2)


if __name__ == '__main__':
    unittest.main()
