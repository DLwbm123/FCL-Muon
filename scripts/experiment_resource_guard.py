"""Independent watchdog for an already running group; never resets its budget."""
import json,os,signal,time
from pathlib import Path

root=Path(os.environ['PROJECT_ROOT'])
job=json.loads((root/'reports/job.json').read_text())
deadline=job['deadline'];limit=job['storage_limit_bytes']
groups=[v['pid'] for v in job['lanes']];known={}

def identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
        return int(fields[19]),int(fields[2]) # process start ticks and process group
    except (FileNotFoundError,ProcessLookupError):return None

def alive(group):
    return any(g==group and identity(pid)==value for pid,(g,value) in known.items())

def allocated():
    total=0
    for p in root.rglob('*'):
        try:
            if p.is_file() and not p.is_symlink():total+=p.stat().st_blocks*512
        except FileNotFoundError:pass # concurrent atomic checkpoint replacement
    return total

def observe_processes():
    candidates=list(groups)
    for path in (root/'reports').glob('lane_seed*.json'):
        state=json.loads(path.read_text());candidates.extend(v['pid'] for v in state['runs'])
    for pid in candidates:
        value=identity(pid)
        if value and value[1] in groups and pid not in known:known[pid]=(value[1],value)

receipt=root/'reports/RESOURCE_GUARD.json'
observe_processes()
receipt.write_text(json.dumps({'status':'active','pid':os.getpid(),'deadline':deadline,'storage_limit_bytes':limit,'groups':groups},indent=2))
while True:
    observe_processes()
    if not any(alive(g) for g in groups):
        receipt.write_text(json.dumps({'status':'exited_after_workers','ended':time.time(),'deadline':deadline}));break
    size=allocated();reason='deadline' if time.time()>=deadline else 'storage' if size>=limit else None
    if reason:
        for group in groups:
            if alive(group):
                try:os.killpg(group,signal.SIGTERM)
                except ProcessLookupError:pass
        time.sleep(15)
        for group in groups:
            if alive(group):
                try:os.killpg(group,signal.SIGKILL)
                except ProcessLookupError:pass
        receipt.write_text(json.dumps({'status':'budget_stop','reason':reason,'ended':time.time(),'allocated_bytes':size,'deadline':deadline}));break
    time.sleep(min(15,max(.1,deadline-time.time())))
