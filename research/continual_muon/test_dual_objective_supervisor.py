import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.dual_objective_supervisor import main


class Child:
    pid = 918273
    returncode = None
    def poll(self):return self.returncode
    def wait(self, timeout=None):
        self.returncode = -15
        return self.returncode


class DualObjectiveSupervisorTest(unittest.TestCase):
    def test_budget_stops_independently_timed_worker(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d);report = root / 'reports';report.mkdir()
            now = time.time()
            (report / 'job.json').write_text(json.dumps({'started': now, 'deadline': now + 2400}))
            env = {'PROJECT_ROOT': str(root), 'SOURCE_ROOT': str(root), 'CUDA_VISIBLE_DEVICES': '0',
                   'NEUTRAL_ENTRY': '/tmp/neutral.py'}
            with patch.dict(os.environ, env), patch('scripts.dual_objective_supervisor.allocated',
                    side_effect=[0, 1_000_000_000, 1_000_000_000]), \
                 patch('subprocess.check_output', return_value='20000\n'), \
                 patch('subprocess.Popen', return_value=Child()) as launch, patch('os.killpg') as kill:
                main()
                self.assertEqual(launch.call_args.args[0][0], '/usr/bin/timeout')
                self.assertEqual(launch.call_args.kwargs['env']['ENTRY_FILE'], 'scripts/dual_objective_diagnostic.py')
                kill.assert_called_once()
            self.assertEqual(json.loads((report / 'job.json').read_text())['status'], 'budget_or_signal_stop')


if __name__ == '__main__':
    unittest.main()
