"""Precondition-only launch: wait for official CIFAR archive, test, then queue."""
import os,runpy,time
from pathlib import Path
import subprocess,sys
root=Path(os.environ['PROJECT_ROOT']);os.chdir(root)
# Preparation failures must remain visible; no blind retry of costly training.
for _ in range(120):
    if (root/'data/cifar-100-python/train').exists():break
    archive=root/'data/cifar-100-python.tar.gz'
    if archive.exists() and archive.stat().st_size==169001437:
        from torchvision.datasets import CIFAR100
        # torchvision performs its built-in known MD5 validation and extraction.
        CIFAR100(root/'data',train=True,download=True)
        break
    time.sleep(20)
else:raise RuntimeError('Official CIFAR preparation incomplete; queue not started')
env=dict(os.environ,ENTRY_MODULE='pytest');env.pop('ENTRY_FILE',None)
code=subprocess.call([sys.executable,'/data/bmw/workers/e17.py','-q','tests'],env=env)
if code:raise RuntimeError('Reference gate failed')
runpy.run_path(str(root/'scripts/queue.py'),run_name='__main__')
