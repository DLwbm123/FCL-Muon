import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.server_round_supervisor import main


class Child:
    pid = 918273
    returncode = None
    def poll(self):
        return self.returncode
    def wait(self, timeout=None):
        self.returncode = -15
        return self.returncode


class ServerRoundSupervisorTest(unittest.TestCase):
    def test_storage_stops_worker_group_with_independent_timeout(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d);report = root / 'reports';report.mkdir()
            now = time.time()
            (report / 'job.json').write_text(json.dumps({'started': now, 'deadline': now + 3600}))
            env = {'PROJECT_ROOT': str(root), 'SOURCE_ROOT': str(root), 'CUDA_VISIBLE_DEVICES': '0',
                   'NEUTRAL_ENTRY': '/tmp/neutral.py'}
            with patch.dict(os.environ, env), patch('scripts.server_round_supervisor.allocated',
                    side_effect=[0, 6_000_000_000, 6_000_000_000]), \
                 patch('subprocess.check_output', return_value='20000\n'), \
                 patch('subprocess.Popen', return_value=Child()) as launch, patch('os.killpg') as kill:
                main()
                self.assertEqual(launch.call_args.args[0][0], '/usr/bin/timeout')
                self.assertEqual(launch.call_args.kwargs['env']['CUBLAS_WORKSPACE_CONFIG'], ':4096:8')
                kill.assert_called_once()
            job = json.loads((report / 'job.json').read_text())
            self.assertEqual(job['status'], 'budget_or_signal_stop')


if __name__ == '__main__':
    unittest.main()
