"""Bounded CPU checks before the matrix diagnostic."""
import os,sys
import pytest
from pathlib import Path
from research.fed_spectra.lifecycle import write
root=Path(os.environ['PROJECT_ROOT'])
code=pytest.main(['-q','--basetemp',str(root/'cpu_tmp'),'tests/test_heldout_round.py','tests/test_all_seen_diagnostic.py','tests/test_matrix_tangent.py','tests/test_decision_diagnostic.py','tests/test_competition_diagnostic.py','tests/test_alignment_diagnostic.py'])
write(root/'reports/CPU_TESTS.json',{'passed':code==0,'exit_code':int(code),'scope':'fixed split, leakage and gradient guards, frozen candidate conditions, geometry and paired isolation'})
sys.exit(code)
