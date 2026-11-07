"""Exclusive, fail-closed ledger primitive for the unregistered diagnostic."""
import fcntl
import json
import math
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def locked_ledger(directory, started, deadline):
    """Hold one supervisor lock and preserve the original clock and charges.

    An active ledger requires an explicit recovery audit; it is never resumed
    automatically, because its last persisted GPU charge may be incomplete.
    """
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or
           not math.isfinite(x) for x in (started, deadline)):
        raise ValueError('invalid clock')
    if not 0 < deadline - started <= 86400:
        raise ValueError('invalid original clock')
    directory = Path(directory)
    if not directory.is_absolute() or not directory.is_dir():
        raise ValueError('ledger directory must exist and be absolute')
    with open(directory / 'supervisor.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = directory / 'job.json'

        def persist(job):
            if job.get('started') != started or job.get('deadline') != deadline:
                raise ValueError('original clock mismatch')
            data = json.dumps(job, allow_nan=False, sort_keys=True).encode() + b'\n'
            fd, name = tempfile.mkstemp(prefix='.job-', dir=directory)
            try:
                with os.fdopen(fd, 'wb') as output:
                    output.write(data)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(name, path)
                parent = os.open(directory, os.O_RDONLY)
                try:
                    os.fsync(parent)
                finally:
                    os.close(parent)
            finally:
                if os.path.exists(name):
                    os.unlink(name)

        if path.exists():
            with path.open() as source:
                job = json.load(source)
            if job.get('started') != started or job.get('deadline') != deadline:
                raise ValueError('original clock mismatch')
            used = job.get('gpu_seconds_used')
            if (isinstance(used, bool) or not isinstance(used, (int, float)) or
                    not math.isfinite(used) or used < 0 or job.get('active') is not False):
                raise ValueError('ledger requires recovery audit')
        else:
            job = dict(started=started, deadline=deadline, gpu_seconds_used=0.0,
                       active=False)
            persist(job)
        yield job, persist
