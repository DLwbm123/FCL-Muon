import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.real_boundary_supervisor import main


class Child:
    pid = 918273
    returncode = None
    def __init__(self, mode):
        self.mode = mode
    def poll(self):
        if self.mode == 'done':
            self.returncode = 0
        return self.returncode
    def wait(self, timeout=None):
        self.returncode = -15
        return self.returncode


class SupervisorTest(unittest.TestCase):
    def run_supervisor(self, root, mode, sizes):
        report = root / 'reports'
        report.mkdir()
        now = time.time()
        (report / 'job.json').write_text(json.dumps({'started': now, 'deadline': now + 10800}))
        if mode == 'done':
            (report / 'RESULT.json').write_text(json.dumps({'status': 'completed'}))
        child = Child(mode)
        env = {'PROJECT_ROOT': str(root), 'SOURCE_ROOT': str(root), 'CUDA_VISIBLE_DEVICES': '0'}
        with patch.dict(os.environ, env), patch('scripts.real_boundary_supervisor.allocated', side_effect=sizes), \
             patch('subprocess.check_output', return_value='20000\n'), \
             patch('subprocess.Popen', return_value=child), patch('os.killpg') as kill:
            main()
            return json.loads((report / 'job.json').read_text()), kill

    def test_success_requires_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            job, _ = self.run_supervisor(Path(d), 'done', [0, 0])
            self.assertEqual(job['status'], 'completed')
            self.assertEqual(job['worker_exit_code'], 0)

    def test_storage_stop_terminates_group(self):
        with tempfile.TemporaryDirectory() as d:
            job, kill = self.run_supervisor(Path(d), 'running', [0, 6_000_000_000, 6_000_000_000])
            self.assertEqual(job['status'], 'budget_or_signal_stop')
            kill.assert_called_once()


if __name__ == '__main__':
    unittest.main()
