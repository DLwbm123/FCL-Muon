"""Server program staged as a neutral entry filename."""

import ctypes
import os
import runpy
import signal


if __name__ == '__main__':
    parent = os.getppid()
    if ctypes.CDLL(None).prctl(1, signal.SIGKILL) != 0 or parent == 1 or os.getppid() != parent:
        raise RuntimeError('Supervisor is unavailable')
    action = os.environ['ACTION']
    if action == 'preflight':
        import unittest
        from research.analytic_comparison.baselines import mathematical_checks

        suite = unittest.defaultTestLoader.loadTestsFromName('research.stat_ablation.test_checks')
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():
            raise SystemExit(1)
        mathematical_checks()
        module = 'research.stat_ablation.preflight'
    elif action == 'rrs':
        module = 'research.stat_ablation.worker'
    else:
        if action != 'analytic':
            raise ValueError('Unknown action')
        module = 'research.analytic_comparison.baselines'
    runpy.run_module(module, run_name='__main__')
