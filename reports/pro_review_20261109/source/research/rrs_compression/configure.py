"""One-time local registration helper; executed under a neutral filename."""
import hashlib
import json
import os
import textwrap
from pathlib import Path

root=Path(__file__).resolve().parent
settings={'STUDY_ROOT':str(root),'PYTHONPATH':str(root/'src')+':'+str(root/'src/deps'),
          'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8',
          'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TMPDIR':str(root/'tmp'),
          'XDG_CACHE_HOME':str(root/'cache'),'TORCH_HOME':str(root/'cache')}
for name in ('tmp','cache'):(root/name).mkdir(exist_ok=True)
(root/'settings.json').write_text(json.dumps(settings,indent=2))
common="""import os,json,sys,ctypes,signal,runpy
from pathlib import Path
base=Path(__file__).resolve().parent
settings=json.loads((base/'settings.json').read_text());os.environ.update(settings)
sys.path[:0]=settings['PYTHONPATH'].split(':')
parent=os.getppid();ctypes.CDLL(None).prctl(1,signal.SIGKILL)
if os.getppid()!=parent or parent==1:raise RuntimeError('parent died before protection')
"""
(root/'controller.py').write_text(common+"runpy.run_module('research.adamw_t10.controller',run_name='__main__')\n")
(root/'worker.py').write_text(common+"if __name__=='__main__':\n"+textwrap.indent("""action=os.environ['ACTION']
if action=='checks':
 import unittest
 from research.adamw_t10.storage import write
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('research.rrs_compression.test_checks'))
 write(base,base/'src/reports/checks.json',{'status':'passed' if result.wasSuccessful() else 'failed','tests':result.testsRun})
 raise SystemExit(0 if result.wasSuccessful() else 1)
runpy.run_module('research.rrs_compression.'+('worker' if action=='train' else 'preflight'),run_name='__main__')
""",' '))
print('neutral entries configured')
