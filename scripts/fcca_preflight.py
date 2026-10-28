"""Synthetic CPU-only qualification, never a scientific experiment restart."""
import json,time
from pathlib import Path
import pytest,torch
start=time.time();torch.set_num_threads(2)
code=pytest.main(['-q','tests/test_fcca.py'])
p=Path('reports/SYNTHETIC_PREFLIGHT.json')
if p.exists():p.rename(p.with_name('SYNTHETIC_PREFLIGHT_PREVIOUS_'+str(int(start))+'.json'))
p.write_text(json.dumps({'exit_code':int(code),'seconds':time.time()-start,'synthetic_cpu_only':True},indent=2))
raise SystemExit(code)
