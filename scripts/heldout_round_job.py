"""Record the diagnostic worker return code without depending on SSH."""
import json,os,subprocess,sys,time
from pathlib import Path
from research.fed_spectra.lifecycle import write

root=Path(os.environ['PROJECT_ROOT'])
origin=json.loads((root/'reports/ORIGIN.json').read_text())
assert time.time()<origin['started']+6900
assert json.loads((root/'reports/CPU_TESTS.json').read_text())['passed']
assert not (root/'reports/job.json').exists()
env=dict(os.environ,ENTRY_FILE='scripts/heldout_round_diagnostic.py')
with (root/'reports/worker.log').open('x') as log:
    p=subprocess.Popen([sys.executable,'/data/bmw/workers/e17.py'],env=env,
                       stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
    state={'status':'running','started':origin['started'],'supervisor_pid':os.getpid(),'worker_pid':p.pid}
    write(root/'reports/job.json',state)
    code=p.wait()
    state.update(status='completed' if code==0 else 'failed',return_code=code,ended=time.time())
    write(root/'reports/job.json',state)
    sys.exit(code)
