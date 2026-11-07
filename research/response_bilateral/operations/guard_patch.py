"""Narrow infrastructure amendment: account for a vanished atomic ledger temp."""
import os
import stat
from pathlib import Path


def allocated(root):
    root = Path(root)
    root_stat = root.lstat()
    if not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError('budget root must be a real directory')
    def fail_walk(error):
        raise error
    total = root_stat.st_blocks * 512
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=fail_walk):
        for name in dirs + files:
            path = Path(directory)/name
            try:
                total += path.lstat().st_blocks * 512
            except FileNotFoundError:
                # persist(job) atomically renames these tiny files. A vanished
                # entry is charged conservatively, never silently undercounted.
                if path.parent != root/'src/reports' or not name.startswith('.job-'):
                    raise
                total += 65536
    return total


def install():
    from research.block_response_diagnostic import boundary_budget
    boundary_budget.allocated = allocated
