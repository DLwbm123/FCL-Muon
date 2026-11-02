import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from research.adamw_combined.holdout_firewall_draft import build_firewalled_manifest
from research.adamw_combined.guarded_prefix_draft import import_guarded_prefix


class GuardedPrefixTest(unittest.TestCase):
    def test_derivation_validates_before_delegating(self):
        manifest = {'cpt': 1, 'labels': {'train': [0, 0, 1, 1, 1, 1]},
                    'partitions': {'train': [[[0, 1]], [[2, 3, 4, 5]]], 'test': [[[0]], [[1]]]}}
        filtered, held = build_firewalled_manifest(manifest, [1], 7, .25)
        source = {'clock': {'started': 1, 'deadline': 2}, 'identity': {'source': 'x'},
                  'config': {'selection_mode': 'prefix', 'seed': 0, 'server_correction': False}}
        saved = copy.deepcopy(source)
        target = SimpleNamespace(identity={'source': 'x', 'guarded_partition': 'sample'},
              manifest=filtered, reserved=held, clock={'started': 3, 'deadline': 4},
              cfg={'selection_mode': 'prefix', 'seed': 0, 'server_correction': True,
                   'bilevel': {'mode': 'optimized'}}, s={}, save=lambda: None)
        def fake_import(pilot, fork):
            self.assertEqual(fork['identity'], target.identity)
            self.assertEqual(fork['clock'], target.clock)
            pilot.s['prefix_provenance'] = {}
        with patch('research.adamw_combined.guarded_prefix_draft.import_prefix', fake_import):
            import_guarded_prefix(target, source, manifest, saved['clock'])
            self.assertTrue(target.s['prefix_provenance']['guarded_partition'])
            self.assertEqual(source, saved)
            target.manifest['partitions']['train'][0][0] = [0]
            with self.assertRaisesRegex(ValueError, 'T1'):
                import_guarded_prefix(target, source, manifest, saved['clock'])
