import unittest
from research.continual_muon.current_task_guard import CurrentTaskGuard


class GuardTest(unittest.TestCase):
    def test_task_isolation_and_actual_visits(self):
        raw = object()
        def loader(source, indices, batch, shuffle):
            return [(None, indices[i:i + batch]) for i in range(0, len(indices), batch)]
        guard = CurrentTaskGuard(loader, raw, {1: [1, 2, 3], 2: [4, 5]})
        old = guard(raw, [1, 2, 3], 2, True)
        self.assertEqual(len(list(old)), 2)
        self.assertEqual(guard.visits[1], 3)
        with self.assertRaises(ValueError):
            guard(raw, [1, 4], 2)
        with self.assertRaises(ValueError):
            guard(object(), [1], 2)
        guard.enter_t3()
        with self.assertRaises(ValueError):
            list(old)
        with self.assertRaises(ValueError):
            guard(raw, [1], 2)
        self.assertEqual(len(list(guard(raw, [4, 5], 2))), 1)
        self.assertEqual(guard.visits, {1: 3, 2: 2})
        with self.assertRaises(ValueError):
            guard.enter_t3()


if __name__ == '__main__':
    unittest.main()
