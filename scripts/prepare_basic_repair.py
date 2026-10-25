"""Build an isolated, pinned faithful trainer with RNG-neutral snapshots only."""
import json
import subprocess
from pathlib import Path

target = Path('runs/basic_repair_deployment')
target.mkdir(parents=True, exist_ok=False)
names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', '00b6508', 'fcl'], text=True).splitlines()
for name in names:
    p = target/name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(subprocess.check_output(['git', 'show', '00b6508:'+name]))
p = target/'fcl/train.py'
s = p.read_text()
needle = "        atomic_save(state,out/'latest.pt')"
assert s.count(needle) == 1
s = s.replace(needle, needle+"\n        link=out/f'boundary_t{task}_r{round_idx}.pt'\n        if not link.exists():os.link(out/'latest.pt',link)")
needle = "                state['m']=[v.cpu() for v in momentum]"
assert s.count(needle) == 1
s = s.replace(needle, "                atomic_save({'B':[b.detach().cpu().clone() for b in bs], 'head':model.rows[-1].detach().cpu().clone(), 'task':t, 'round':r, 'client':k},out/f'client_t{t}_r{r}_k{k}.pt')\n"+needle)
p.write_text(s)
cfg = json.loads(Path('configs/pilot_mulora_fl.json').read_text())
cfg.update(tasks=1, output='runs/t1_original')
(target/'configs').mkdir()
(target/'configs/t1.json').write_text(json.dumps(cfg, indent=2))
(target/'reports').mkdir()
(target/'reports/DEPLOYMENT.json').write_text(json.dumps({'parent':'00b6508','changes':['Retain atomic round snapshots via hard links','Save detached CPU B/head endpoints after each client; no RNG calls'],'training_algorithm_changes':False}, indent=2))
print(target)
