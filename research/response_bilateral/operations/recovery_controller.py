"""Single audited infrastructure recovery under the original clock and ledger."""
import hashlib
import json
import os
from pathlib import Path
from guard_patch import install

BASE = Path(__file__).resolve().parent


def adapt(source):
    changes = [
        ("if old['status'] not in ('development_checked',):",
         "if old['status'] != 'train_s4_E3_T5_failed' or old.get('active') is not False:"),
        ("job.update(status='registering', active=False); persist(job)",
         "job.update(status='audited_storage_recovery', active=False); persist(job)"),
        ("def run(name, entry, values=None, limit=None):\n",
         "def run(name, entry, values=None, limit=None):\n"
         "            if name+'_exit' in job:\n"
         "                if job[name+'_exit'] == 0 and stage_receipt(root, name, values):\n"
         "                    return True\n"
         "                if name != 'train_s4_E3_T5' or job[name+'_exit'] != 1:\n"
         "                    raise ValueError('unapproved failed or incomplete stage')\n"),
        ("if not enough(group_seconds, space=2500000000):",
         "if f'prefix_s{seed}_exit' not in job and not enough(group_seconds, space=2500000000):"),
        ("with (reports/(name+'.log')).open('x') as log:",
         "env['ORIGINAL_ENTRY_FILE'] = env['ENTRY_FILE']\n"
         "            env['ENTRY_FILE'] = str(root.parent/'stage_entry.py')\n"
         "            suffix = '.recovery1.log' if name == 'train_s4_E3_T5' else '.log'\n"
         "            with (reports/(name+suffix)).open('x') as log:"),
        ("status='waiting_for_idle_gpu'", "status='waiting_for_gpu_memory'")]
    for old, new in changes:
        if source.count(old) != 1:
            raise ValueError('original controller source contract changed')
        source = source.replace(old, new)
    compile(source, '<audited-storage-recovery>', 'exec')
    return source


def main():
    settings = json.loads((BASE/'settings.json').read_text())
    os.environ.update(settings)
    import sys
    sys.path[:0] = settings['PYTHONPATH'].split(os.pathsep)
    install()
    from resource_controller import choose_memory_gpu, stage_receipt
    root = Path(settings['PROJECT_ROOT'])
    r = root/'reports'
    frozen = json.loads((r/'FROZEN_PROTOCOL.json').read_text())
    audit = json.loads((r/'STORAGE_RECOVERY_AMENDMENT_1.json').read_text())
    job = json.loads((r/'job.json').read_text())
    if job != audit['failure_job']:
        raise ValueError('failure ledger changed since audit')
    for name, expected in audit['shim_sha256'].items():
        if hashlib.sha256((BASE/name).read_bytes()).hexdigest() != expected:
            raise ValueError('recovery shim changed')
    path = root/'research/response_bilateral/supervisor.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != frozen['source_sha256']['research/response_bilateral/supervisor.py']:
        raise ValueError('frozen controller changed')
    namespace = {'__name__': 'audited_storage_controller', '__file__': str(path), 'stage_receipt': stage_receipt}
    exec(compile(adapt(path.read_text()), str(path), 'exec'), namespace)
    namespace['idle_gpu'] = choose_memory_gpu
    namespace['freeze'] = lambda _: frozen
    namespace['main']()


if __name__ == '__main__':
    main()
