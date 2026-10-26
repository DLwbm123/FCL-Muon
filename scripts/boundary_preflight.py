"""Small deterministic CPU qualification; no scientific training or clock reset."""
import json,time
from pathlib import Path
import pytest,torch
start=time.time();torch.set_num_threads(2)
code=pytest.main(['-q','tests/test_boundary_repair.py','tests/test_aggregation_utility.py'])
path=Path('reports/SYNTHETIC_PREFLIGHT.json')
if path.exists():path.rename(path.with_name('SYNTHETIC_PREFLIGHT_PREVIOUS_'+str(int(start))+'.json'))
path.write_text(json.dumps({'exit_code':int(code),'seconds':time.time()-start,'synthetic_cpu_only':True},indent=2))
raise SystemExit(code)
