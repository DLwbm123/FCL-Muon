"""Continuation entry; keeps the completed three-task reports immutable."""
import os,time
from pathlib import Path
from scripts.boundary_train import main,write

if __name__=='__main__':
    try:main(config_name='boundary_ten_seed0.json',report_subdir='reports/ten',continuation=True)
    except BaseException as exc:
        write(Path(os.environ['PROJECT_ROOT'])/'reports/ten/FAILURE.json',{'time':time.time(),'error':repr(exc)})
        raise
