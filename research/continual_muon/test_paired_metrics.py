import json
import unittest
from research.continual_muon.paired_metrics import compare, conditions


class PairedMetricsTest(unittest.TestCase):
    def test_counts_and_forgetting(self):
        y = [0, 1, 2, 3, 4, 5]
        baseline = [0, 1, 2, 3, 0, 0]
        proposed = [0, 0, 2, 0, 4, 0]
        r = compare(y, proposed, baseline, 2, .75, [.75, .75])
        self.assertEqual(r['corrected_vs_zero'], 1)
        self.assertEqual(r['broken_vs_zero'], 2)
        self.assertEqual(r['old_correct'], 2)
        self.assertEqual(r['new_correct'], 1)
        self.assertAlmostEqual(r['forgetting'], .25)
        with self.assertRaises(ValueError):
            compare(y[:-1], proposed, baseline, 2, .75, [.75, .75])

    def test_gate_includes_negative_conditions(self):
        zero = {'old_correct': 100, 'new_correct': 100, 'forgetting': .1, 'raw_accuracy': .8}
        direct = {'raw_accuracy': .81}
        guarded = {'old_correct': 99, 'new_correct': 90, 'forgetting': .1, 'raw_accuracy': .81,
                   'corrected_vs_zero': 9, 'broken_vs_zero': 3}
        result = conditions({'zero': zero, 'projected': direct, 'guarded': guarded})
        json.dumps(result, allow_nan=False)
        self.assertTrue(result['raw_gain_at_least_6_of_3000'])
        self.assertFalse(result['old_accuracy_not_lower'])
        self.assertTrue(result['new_accuracy_no_more_than_10_of_1000_lower'])


if __name__ == '__main__':
    unittest.main()
