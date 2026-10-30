"""Bounded return-code receipt for the fixed compression diagnostic."""
import json,os,subprocess,sys,time
from pathlib import Path
from research.fed_spectra.lifecycle import write
root=Path(os.environ['PROJECT_ROOT']);report=root/'reports'
origin=json.loads((report/'ORIGIN.json').read_text())
assert time.time()<origin['started']+2400 and not (report/'job.json').exists()
assert json.loads((report/'CPU_TESTS.json').read_text())['passed']
with (report/'worker.log').open('x') as log:
    p=subprocess.Popen([sys.executable,'/data/bmw/workers/e17.py'],env=dict(os.environ,ENTRY_FILE='scripts/rank8_transport_diagnostic.py'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    state={'status':'running','started':origin['started'],'supervisor_pid':os.getpid(),'worker_pid':p.pid}
    write(report/'job.json',state)
    code=p.wait();state.update(status='completed' if code==0 else 'failed',return_code=code,ended=time.time())
    write(report/'job.json',state);sys.exit(code)
