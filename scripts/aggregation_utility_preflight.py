"""Synthetic CPU checks only, before the independent scientific clock starts."""
import json
import time
from pathlib import Path
import pytest
import torch
start=time.time();torch.set_num_threads(2)
code=pytest.main(['-q','tests/test_aggregation_utility.py'])
p=Path('reports/PREFLIGHT.json')
if p.exists():
    archive=p.with_name('PREFLIGHT_PREVIOUS_'+str(int(start))+'.json')
    p.rename(archive)
p.write_text(json.dumps({'exit_code':int(code),'seconds':time.time()-start,'synthetic_cpu_only':True},indent=2))
raise SystemExit(code)
