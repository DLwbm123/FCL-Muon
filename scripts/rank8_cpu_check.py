import os,sys
from pathlib import Path
import pytest
from research.fed_spectra.lifecycle import write
root=Path(os.environ['PROJECT_ROOT'])
code=pytest.main(['-q','--basetemp',str(root/'cpu_tmp'),'tests/test_rank8_transport.py','tests/test_matrix_tangent.py','tests/test_heldout_round.py'])
write(root/'reports/CPU_TESTS.json',{'passed':code==0,'exit_code':int(code),'scope':'rank8 wire oracle, bad-seed conditions, matrix operator and validation guards'})
sys.exit(code)
