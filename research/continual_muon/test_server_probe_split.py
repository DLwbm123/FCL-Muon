import unittest
from research.continual_muon.server_probe_split import split


class ProbeSplitTest(unittest.TestCase):
    def test_deterministic_disjoint_and_empty_client(self):
        value = split([[4, 1, 3, 2], [9], [10, 8]])
        self.assertEqual(value, [([1, 2], [3, 4]), ([], []), ([8], [10])])
        with self.assertRaises(ValueError):
            split([[1, 2], [2, 3]])
        with self.assertRaises(ValueError):
            split([[1, 2]], 1)


if __name__ == '__main__':
    unittest.main()
