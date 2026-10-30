"""Bounded CPU checks before the matrix diagnostic."""
import os,sys
import pytest
from pathlib import Path
from research.fed_spectra.lifecycle import write
root=Path(os.environ['PROJECT_ROOT'])
code=pytest.main(['-q','--basetemp',str(root/'cpu_tmp'),'tests/test_all_seen_diagnostic.py','tests/test_matrix_tangent.py','tests/test_decision_diagnostic.py','tests/test_competition_diagnostic.py','tests/test_alignment_diagnostic.py'])
write(root/'reports/CPU_TESTS.json',{'passed':code==0,'exit_code':int(code),'scope':'all-seen gradient identity, strict decision conditions, matrix budget and isolated paired statistics'})
sys.exit(code)
