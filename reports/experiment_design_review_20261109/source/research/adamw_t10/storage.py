"""Serialize allocation scans with atomic writes, including checkpoint rename."""
import contextlib
import fcntl
import json
import os
from pathlib import Path
from fcl.checkpoint import atomic_save


@contextlib.contextmanager
def lock(root):
    with (Path(root)/'.storage.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def allocated(root):
    total = 0
    for base, dirs, files in os.walk(root, followlinks=False):
        for name in files:
            path = Path(base)/name
            stat = path.lstat()
            total += stat.st_blocks*512
    return total


def write(root, path, value):
    path = Path(path)
    with lock(root):
        temp = path.with_suffix('.tmp')
        with temp.open('w') as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temp, path)


def save(root, path, value):
    with lock(root):
        if allocated(root)+1_500_000_000 >= 18_000_000_000:
            raise RuntimeError('storage reserve reached; existing checkpoint retained')
        atomic_save(value, path)
