"""Durable serial queue. Each config owns an original clock, preserved on retry."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from scripts.gates import qualify
root=Path(os.environ['PROJECT_ROOT']);os.chdir(root)
queue=json.loads((root/'queue.json').read_text())
statepath=root/'reports/queue-state.json'
states=json.loads(statepath.read_text()) if statepath.exists() else {}

def record():
    tmp=statepath.with_suffix('.tmp');tmp.write_text(json.dumps(states,indent=2));tmp.replace(statepath)

for item in queue:
    cfgpath=item['config'];cfg=json.loads((root/cfgpath).read_text());out=root/cfg['output']
    s=states.setdefault(cfgpath,{'status':'not_started'})
    if (out/'COMPLETE.json').exists():
        qualify(root,cfg);s['status']='completed';record();continue
    if s['status'] in ['failed','budget_exhausted','blocked']:break
    if cfg['dataset']=='seq-imagenetr' and not (root/'data/imagenet-r').exists():
        s.update(status='blocked',reason='ImageNet-R images not prepared; frozen YAML required');record();break
    if not (root/'weights.json').exists():
        s.update(status='blocked',reason='Required IN21K weights unavailable');record();break
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','-i',os.environ.get('CUDA_VISIBLE_DEVICES','6')],text=True).strip())
    if free<12000:s.update(status='blocked',reason='Less than 12 GB free GPU memory');record();break
    started=s.setdefault('started',time.time());remaining=started+85500-time.time()
    if remaining<=0:s['status']='budget_exhausted';record();break
    env=dict(os.environ,ENTRY_MODULE='fcl.train',RUN_CONFIG=str(root/cfgpath))
    env.pop('ENTRY_FILE',None)
    if (out/'latest.pt').exists():env['RESUME']=str(out/'latest.pt')
    else:env.pop('RESUME',None)
    out.mkdir(parents=True,exist_ok=True)
    s.update(status='running',gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),deadline=started+86400);record()
    with (out/'console.log').open('a') as log:
        child=subprocess.Popen([sys.executable,'/data/bmw/workers/e17.py'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        s['pid']=child.pid;record()
        deadline=time.time()+remaining
        while child.poll() is None:
            # Conservative 45GB guard reserves 5GB for atomic checkpoints/grace.
            allocated=sum(p.stat().st_blocks*512 for p in root.rglob('*') if p.is_file() and not p.is_symlink())
            if time.time()>=deadline or allocated>=45_000_000_000:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=60)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                s.update(status='budget_exhausted',allocated_bytes=allocated);break
            time.sleep(20)
    s['exit_code']=child.returncode;s['ended']=time.time()
    if (out/'COMPLETE.json').exists():
        try:qualify(root,cfg);s['status']='completed'
        except Exception as e:s.update(status='failed',reason='qualification: '+repr(e))
    elif s['status']!='budget_exhausted':s['status']='failed'
    record()
    if s['status']!='completed':break
    # Do not select parameters/checkpoints based on test metrics. Finite completion
    # and contract tests qualify the next stage; monitors review scientific diagnostics.
