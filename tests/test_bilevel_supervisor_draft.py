"""Synthetic supervisor checks; no GPU or experiment processes are launched."""
import json
import os
import time
import pytest
from scripts import bilevel_supervisor_draft as supervisor


def prepare(tmp_path, monkeypatch):
    root = tmp_path / 'root'
    report = root / 'reports'
    report.mkdir(parents=True)
    now = time.time()
    (report / 'job.json').write_text(json.dumps({'started': now, 'deadline': now + 28800}))
    monkeypatch.setenv('PROJECT_ROOT', str(root))
    monkeypatch.setenv('SOURCE_ROOT', str(tmp_path / 'source'))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setenv('WORKER_ENTRY', '/neutral/entry.py')
    return root, report


def test_refuses_full_root_before_worker(tmp_path, monkeypatch):
    root, report = prepare(tmp_path, monkeypatch)
    monkeypatch.setattr(supervisor, 'allocated', lambda path: 10_000_000_000)
    with pytest.raises(ValueError, match='write-stop'):
        supervisor.main()
    assert json.loads((report / 'job.json').read_text())['status'] == 'supervisor_failure'
    assert not (report / 'worker.log').exists()


def test_worker_completion_and_duplicate_claim_refused(tmp_path, monkeypatch):
    root, report = prepare(tmp_path, monkeypatch)
    monkeypatch.setattr(supervisor, 'allocated', lambda path: 0)
    monkeypatch.setattr(supervisor.subprocess, 'check_output', lambda *a, **k: '40000')
    launched = []
    class Child:
        pid = 123
        returncode = 0
        def poll(self): return 0
    def launch(args, **kwargs):
        launched.append((args, kwargs['env']['ENTRY_FILE']))
        (report / 'RESULT.json').write_text(json.dumps({'status': 'completed'}))
        return Child()
    monkeypatch.setattr(supervisor.subprocess, 'Popen', launch)
    supervisor.main()
    assert json.loads((report / 'job.json').read_text())['status'] == 'completed'
    assert launched[0][0][-1] == '/neutral/entry.py'
    assert len(launched) == 1
    with pytest.raises(FileExistsError):
        supervisor.main()
    assert len(launched) == 1


def test_write_limit_stops_running_worker_group(tmp_path, monkeypatch):
    root, report = prepare(tmp_path, monkeypatch)
    sizes = iter((0, 10_000_000_000, 10_000_000_000))
    monkeypatch.setattr(supervisor, 'allocated', lambda path: next(sizes))
    monkeypatch.setattr(supervisor.subprocess, 'check_output', lambda *a, **k: '40000')
    killed = []
    monkeypatch.setattr(supervisor.os, 'killpg', lambda pid, sig: killed.append((pid, sig)))
    class Child:
        pid = 123
        returncode = None
        def poll(self): return self.returncode
        def wait(self, timeout=None):
            self.returncode = -15
            return self.returncode
    monkeypatch.setattr(supervisor.subprocess, 'Popen', lambda *a, **k: Child())
    supervisor.main()
    job = json.loads((report / 'job.json').read_text())
    assert job['status'] == 'budget_or_signal_stop'
    assert job['worker_exit_code'] == -15
    assert len(killed) == 1 and killed[0][0] == 123
