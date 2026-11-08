"""Launch a detached hard timeout and neutral supervisor exactly once."""
import json
import math
import os
import subprocess
import time
from pathlib import Path

root=Path(__file__).resolve().parent
if (root/'launch.json').exists():raise RuntimeError('launch already recorded')
f=json.loads((root/'src/reports/FROZEN_PROTOCOL.json').read_text())
remaining=math.floor(f['deadline']-time.time()-35)
if remaining<600:raise RuntimeError('insufficient remaining clock')
with (root/'controller.log').open('x') as log:
    p=subprocess.Popen(['timeout','--signal=TERM','--kill-after=30s',str(remaining),
                        '/data/bmw/envs/n3/bin/python',str(root/'controller.py')],
                       stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                       env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
(root/'launch.json').write_text(json.dumps({'outer_pid':p.pid,'launched':time.time(),
    'original_started':f['started'],'original_deadline':f['deadline'],'timeout_seconds':remaining},indent=2))
print((root/'launch.json').read_text())
