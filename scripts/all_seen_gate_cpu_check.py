"""Run related lifecycle and operator checks before the matrix gate study."""
import os,sys
import pytest
from pathlib import Path
from research.fed_spectra.lifecycle import write
root=Path(os.environ['PROJECT_ROOT'])
code=pytest.main(['-q','--basetemp',str(root/'cpu_tmp'),'tests/test_all_seen_gate.py','tests/test_all_seen_diagnostic.py','tests/test_matrix_gate.py','tests/test_matrix_tangent.py','tests/test_tangent_gate.py','tests/test_spectral_tangent.py','tests/test_direction_gate.py','tests/test_loss_gate.py','tests/test_alignment_diagnostic.py'])
write(root/'reports/CPU_TESTS.json',{'passed':code==0,'exit_code':int(code),'scope':'all-seen protected gate, exact rejection, client gradient oracle, recovery, matrix and legacy regression'})
sys.exit(code)
