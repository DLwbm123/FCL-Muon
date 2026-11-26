"""Prepare a private server freeze and start one detached supervisor."""

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path.cwd()
REFERENCE = Path(os.environ['REFERENCE_ROOT'])
PYTHON = sys.executable
REPORTS = ROOT / 'src/reports'
PUBLIC = ROOT / 'protocol.json'


def write_json(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')


def register():
    f = json.loads((REFERENCE / 'src/reports/FROZEN_PROTOCOL.json').read_text())
    base = json.loads(PUBLIC.read_text())
    f = {key: f[key] for key in ('config', 'weights', 'asset_stat', 'environment', 'manifest_sha256')}
    f.update(base)
    adapter = ROOT / 'reference/vision_transformer_adapter.py'
    stat = adapter.stat()
    f['asset_stat'][str(adapter)] = [stat.st_size, stat.st_mtime_ns, stat.st_ino]
    paths = [p for parent in ('src/fcl', 'src/research') for p in (ROOT / parent).rglob('*.py')]
    paths.extend(ROOT / name for name in ('protocol.json', 'entry.py', 'supervisor.py'))
    f['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    f['code_identity'] = os.environ['SOURCE_REVISION']
    f['status'] = 'frozen_before_cpu_checks_or_data_access'
    f['started'] = time.time()
    f['deadline'] = f['started'] + f['budget']['wall_seconds']
    write_json(REPORTS / 'FROZEN_PROTOCOL.json', f)
    return f


def main():
    f = register()
    env = dict(os.environ, STUDY_ROOT=str(ROOT), PYTHONPATH=str(ROOT / 'src') + ':' + str(ROOT / 'src/deps'),
               PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TMPDIR=str(ROOT / 'tmp'),
               XDG_CACHE_HOME=str(ROOT / 'cache'), TORCH_HOME=str(ROOT / 'cache'))
    remaining = math.floor(f['deadline'] - time.time() - 35)
    with (ROOT / 'controller.log').open('x') as log:
        controller = subprocess.Popen(
            ['timeout', '--signal=TERM', '--kill-after=30s', str(remaining), PYTHON,
             str(ROOT / 'supervisor.py')],
            env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
        )
    receipt = dict(timeout_pid=controller.pid, started=f['started'], deadline=f['deadline'])
    write_json(ROOT / 'launch.json', receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
