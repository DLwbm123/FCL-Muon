"""Neutral process entry; paths/config passed via environment, never argv."""
import os,runpy,sys
from pathlib import Path
root=Path(os.environ['PROJECT_ROOT'])
os.chdir(root)
sys.path.insert(0,str(root))
if os.environ.get('ENTRY_MODULE'):
    runpy.run_module(os.environ['ENTRY_MODULE'],run_name='__main__')
else:
    runpy.run_path(str(root/os.environ['ENTRY_FILE']),run_name='__main__')
