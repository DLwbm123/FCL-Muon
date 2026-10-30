import os,sys
from pathlib import Path
import pytest
from research.fed_spectra.lifecycle import write
root=Path(os.environ['PROJECT_ROOT'])
code=pytest.main(['-q','--basetemp',str(root/'cpu_tmp'),'tests/test_rank8_gate.py','tests/test_rank8_transport.py','tests/test_all_seen_gate.py','tests/test_matrix_gate.py','tests/test_matrix_tangent.py','tests/test_preflight_precision.py'])
write(root/'reports/CPU_TESTS.json',{'passed':code==0,'exit_code':int(code),'scope':'rank8 client wire oracle, exact rejection, recovery, cost, config and matrix/all-seen regression'})
sys.exit(code)
