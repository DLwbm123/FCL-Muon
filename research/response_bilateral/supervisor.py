"""One locked ledger, independent group termination, no automatic failed retry."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from research.block_response_diagnostic.boundary_ledger import locked_ledger
from research.block_response_diagnostic.boundary_budget import allocated, run_guarded
from research.response_bilateral.runtime import roots, write
from research.response_bilateral.freeze import freeze
from research.response_bilateral.report import report, decisions


ARMS = ('E0', 'E3', 'E1', 'E2')


def idle_gpu():
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True).splitlines()
    for row in rows:
        index, used, util = (int(x.strip()) for x in row.split(','))
        if used < 100 and util == 0: return str(index)
    return None


def main():
    root, source, prior = roots()
    reports = root/'reports'
    old = json.loads((reports/'job.json').read_text())
    with locked_ledger(reports, old['started'], old['deadline']) as (job, persist):
        if old['status'] not in ('development_checked',):
            raise ValueError('not an explicitly checked development ledger; audit recovery first')
        job.update(status='registering', active=False); persist(job)
        f = freeze(job)
        def enough(seconds=0, space=700000000):
            return (job['gpu_seconds_used']+seconds < 85500 and
                    time.time()+seconds+300 < job['deadline'] and allocated(root.parent)+space < 18000000000)
        def run(name, entry, values=None, limit=None):
            if not enough(): return False
            device = idle_gpu()
            while device is None and enough():
                job.update(status='waiting_for_idle_gpu', pending=name, checked=time.time()); persist(job)
                time.sleep(15); device = idle_gpu()
            if device is None: return False
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=device, SUPERVISOR_PID=str(os.getpid()),
                       ENTRY_FILE=str(root/entry), **(values or {}))
            job.update(status=name+'_running', device=device, pending=name); persist(job)
            with (reports/(name+'.log')).open('x') as log:
                code, reason = run_guarded([sys.executable, str(root.parent/'worker.py')], env, log,
                    root.parent, job, persist, gpu_cap=86400, space_stop=18500000000,
                    interval=2, margin=120, max_worker_seconds=limit)
            job[name+'_exit'] = code
            if code != 0 or reason:
                job.update(status=name+'_failed', stop_reason=reason); persist(job)
                report(root)
                raise RuntimeError('frozen stage failed; retain clock/evidence, no automatic retry')
            job.update(status=name+'_completed', checked=time.time()); persist(job)
            return True
        if not run('preflight', 'research/response_bilateral/preflight.py', limit=2400):
            job.update(status='budget_before_preflight'); persist(job); return
        if json.loads((reports/'preflight_result.json').read_text())['status'] != 'passed':
            raise ValueError('no passing GPU preflight receipt')
        # Own synthetic checkpoint files only; validation receipts and logs stay.
        for folder in ('synthetic', 'synthetic_initial'):
            path = root/'runs'/folder
            if path.exists():
                for p in path.glob('*.pt'): p.unlink()
        estimates = []
        def task(seed, arm, target, horizon):
            name = f'train_s{seed}_{arm}_T{target}'
            ok = run(name, 'research/response_bilateral/worker.py',
                dict(ACTION='train', SEED=str(seed), ARM=arm, TARGET_TASK=str(target), HORIZON=str(horizon)))
            if ok:
                r = json.loads((reports/(name+'.json')).read_text())
                estimates.append(r['wall_seconds'])
                report(root)
            return ok
        def stop(reason):
            records, rows, complete = report(root)
            write(reports/'decision.json', {'status': reason, 'decision': decisions(records, complete),
                                          'budget_snapshot': dict(job)})
            job.update(status=reason, active=False, checked=time.time()); persist(job)
        # T2 actions first, so the prespecified H1 rule precedes task expansion.
        for arm in ARMS:
            if estimates and not enough(1.4*max(estimates)):
                stop('budget_partial_development'); return
            if not task(0, arm, 2, 3): stop('budget_partial_development'); return
        near = {}
        for arm in ('E2', 'E3'):
            r = json.loads((reports/f'train_s0_{arm}_T2.json').read_text())
            entries = [v for v in r['T2_activity'] if v['norm_before'] > 0]
            near[arm] = sum(v['relative_correction'] < 1e-6 for v in entries)/len(entries) if entries else 1.
        write(reports/'H1_activity.json', {'near_identity_fraction': near, 'threshold': .90})
        if all(v >= .9 for v in near.values()): stop('H1_near_identity_stop'); return
        # Before each complete matrix, use real measured task cost with fixed
        # 40% margin. Scores never enter this admission or the arm order.
        if not enough(1.4*max(estimates)*4, space=2000000000):
            stop('budget_before_T3_matrix'); return
        for arm in ARMS:
            if not task(0, arm, 3, 3): stop('budget_partial_T3'); return
        for seed in (3, 4, 5):
            task_estimate = 1.4*max(estimates)
            group_seconds = 17*task_estimate  # one T1 plus 4 arms x T2..T5
            if not enough(group_seconds, space=2500000000):
                stop('budget_before_next_complete_T5_group'); return
            if not run(f'prefix_s{seed}', 'research/response_bilateral/worker.py',
                       dict(ACTION='prefix', SEED=str(seed), ARM='E0', HORIZON='5')):
                stop('budget_prefix_incomplete'); return
            for arm in ARMS:
                for target in (2, 3, 4, 5):
                    if not task(seed, arm, target, 5): stop('budget_partial_T5'); return
        records, _, complete = report(root)
        decision = decisions(records, complete)
        write(reports/'decision.json', decision)
        winner = decision.get('winner')
        # E2's magnitude attribution needs the explicitly conditional control.
        if winner == 'E2' and enough(1.4*max(estimates)*12, space=1500000000):
            for seed in (3, 4, 5):
                for target in (2, 3, 4, 5):
                    if not task(seed, 'E1E2', target, 5): stop('budget_conditional_control'); return
        # Long sequences require a fresh complete E0/winner pair admission;
        # no best-seed selection. Start at seed3, using registered T5 forks.
        if winner and enough(1.4*max(estimates)*10, space=1000000000):
            # Own T5 checkpoints are forked, not overwritten. Each fork writes
            # an extension receipt before any T6 data/model evaluation.
            for arm in ('E0', winner):
                for target in (6, 7, 8, 9, 10):
                    if not task(3, arm, target, 10): stop('budget_partial_T10'); return
        stop('completed_bounded_study')


if __name__ == '__main__': main()
