import json,time
from pathlib import Path
import pytest,torch
started=time.time();torch.set_num_threads(2)
code=pytest.main(['-q','tests/test_boundary_repair.py','tests/test_aggregation_utility.py'])
p=Path('reports/ten/SYNTHETIC_PREFLIGHT.json')
if p.exists():p.rename(p.with_name('SYNTHETIC_PREFLIGHT_PREVIOUS_'+str(int(started))+'.json'))
p.write_text(json.dumps({'exit_code':int(code),'seconds':time.time()-started,'synthetic_CPU_only':True},indent=2))
raise SystemExit(code)
