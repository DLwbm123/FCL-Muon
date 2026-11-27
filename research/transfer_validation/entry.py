"""Neutral entry that receives the lane from the supervisor environment."""

import ctypes
import os
import runpy
import signal


if __name__=='__main__':
    parent=os.getppid()
    if ctypes.CDLL(None).prctl(1,signal.SIGKILL)!=0 or parent==1 or os.getppid()!=parent:
        raise RuntimeError('Supervisor unavailable')
    modules={
        'check':'research.transfer_validation.checks',
        'replay':'research.analytic_comparison.baselines',
        'audit':'research.transfer_validation.oracle',
        'fit':'research.stat_ablation.worker',
        'frozen':'research.transfer_validation.frozen_stage',
    }
    runpy.run_module(modules[os.environ['OPERATION']],run_name='__main__')
