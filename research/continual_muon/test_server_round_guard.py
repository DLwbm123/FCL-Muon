import unittest
from research.continual_muon.current_task_guard import CurrentTaskGuard
from research.fed_spectra.heldout_round import GuardedLoader, split_clients


class ServerRoundGuardTest(unittest.TestCase):
    def test_confirmation_cannot_enter_training_or_prior_task(self):
        groups = [list(range(20)), list(range(20, 40))]
        fit, hold = split_clients(groups, 0)
        raw = object()
        def loader(source, ids, batch, shuffle=False):
            return [(None, ids)]
        taskguard = CurrentTaskGuard(loader, raw, {1: range(100, 110), 2: range(40)})
        taskguard.enter_t3()
        guard = GuardedLoader(taskguard, raw, [i for g in fit for i in g], [i for g in hold for i in g])
        self.assertEqual(len(list(guard(raw, fit[0], 32, True))), 1)
        with self.assertRaises(AssertionError):
            guard(raw, hold[0], 32)
        self.assertEqual(guard.training_validation_accesses, 1)
        guard.allow_validation = True
        self.assertEqual(len(list(guard(raw, hold[0], 32))), 1)
        with self.assertRaises(AssertionError):
            guard(raw, [100], 32)


if __name__ == '__main__':
    unittest.main()
