"""Audited infrastructure shim around the unmodified registered worker."""
import copy
import json
import os
import runpy
from pathlib import Path
from guard_patch import install

install()
from research.response_bilateral import runtime

root = Path(os.environ['PROJECT_ROOT'])
if (os.environ.get('ACTION'), os.environ.get('SEED'), os.environ.get('ARM'),
        os.environ.get('TARGET_TASK')) == ('train', '4', 'E3', '5'):
    original_write = runtime.write
    audit = json.loads((root/'reports/STORAGE_RECOVERY_AMENDMENT_1.json').read_text())
    previous = json.loads((root/'reports/train_s4_E3_T4.json').read_text())
    def write(path, data):
        if Path(path).name == 'train_s4_E3_T5.json' and data.get('status') == 'completed':
            original_write(root/'reports/train_s4_E3_T5.resume_invocation.json', data)
            data = copy.deepcopy(data)
            data['resumed_invocation_training_visits'] = data['additional_training_visits']
            data['additional_training_visits'] += audit['checkpoint']['T5_visits']
            data['additional_communication'] = {
                k: v-previous['communication'].get(k, 0)
                for k, v in data['communication'].items()}
            data['infrastructure_recovery'] = {
                'amendment': 'STORAGE_RECOVERY_AMENDMENT_1.json',
                'wall_seconds_scope': 'successful resumed invocation only; failed worker already charged in original global GPU ledger',
                'loader_accesses_scope': 'resumed invocation only; failed-attempt exact loader totals unavailable',
                'discarded_client_replay_visits': audit['discarded_client']['visited'],
                'discarded_client_replay_steps': audit['discarded_client']['steps'],
                'failed_attempt_evidence': 'failure_storage_race_1',
                'science_unchanged': True}
        return original_write(path, data)
    runtime.write = write

runpy.run_path(os.environ['ORIGINAL_ENTRY_FILE'], run_name='__main__')
