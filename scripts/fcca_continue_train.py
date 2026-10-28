"""User-authorized T4-T10 continuation, preserving the three-task result."""
import os,time
from pathlib import Path
from scripts.fcca_train import main,write

if __name__=='__main__':
    try:main(config_name='fcca_ten_seed0.json',report_subdir='reports/ten',continuation=True)
    except BaseException as exc:
        write(Path(os.environ['PROJECT_ROOT'])/'reports/ten/FAILURE.json',{'time':time.time(),'error':repr(exc)})
        raise
