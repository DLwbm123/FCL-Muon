"""One registered task at a time. All scheduling and budgets are external."""
import copy
import json
import os
import time
from pathlib import Path
import torch
from research.response_bilateral.runtime import setup, roots, validate, build, import_t1, digest, write


def main():
    setup()
    frozen, job = validate()
    root, source, prior = roots()
    action = os.environ['ACTION']
    seed, arm, horizon = int(os.environ['SEED']), os.environ['ARM'], int(os.environ['HORIZON'])
    clock = {k: job[k] for k in ('started', 'deadline')}
    start = time.monotonic()
    p = build(seed, arm, horizon, clock)
    if action == 'prefix':
        p.cfg['route']['mode'] = 'prefix'; p.cfg['selection_mode'] = 'prefix'
        p.cfg['common_intervention'] = False
        p.out = root/f'runs/prefix{seed}'; p.out.mkdir(exist_ok=False)
    elif action == 'train':
        if (p.out/'latest.pt').exists():
            p.restore()
        elif horizon == 10:
            path = root/f'runs/s{seed}_{arm}_T5/latest.pt'
            ck = torch.load(path, map_location='cpu', weights_only=False)
            expected = copy.deepcopy(ck['config'])
            expected.update(stop_after_task=10, output=p.cfg['output'])
            if (expected != p.cfg or ck['clock'] != clock or ck['identity'] != p.identity
                    or (ck['state']['task'], ck['state']['phase']) != (5, 'task_start')
                    or ck['state']['merges'] != list(range(5)) or ck['state']['history'].committed != set(range(5))):
                raise ValueError('incompatible registered T5 extension')
            write(root/'reports'/f'FROZEN_EXTENSION_s{seed}_{arm}.json',
                  {'source_sha256': digest(path), 'from_config': ck['config'], 'to_config': p.cfg,
                   'original_clock': clock, 'rule': 'frozen protocol winner, fixed seed3, no new hyperparameters'})
            for _ in range(ck['head_count']): p.model.add_head(p.cfg['cpt'])
            p.model.load_state_dict(ck['model'], strict=True); p.s = copy.deepcopy(ck['state'])
            for n, a in p.model.adapters.items():
                a.merged = ck['merged_flags'][n]
                if not a.merged: raise ValueError('unmerged extension')
            from fcl.checkpoint import restore_rng
            restore_rng(ck['rng']); p.save()
            del ck
        else:
            path = prior/'runs/smoke_adamw/latest.pt' if seed == 0 else root/f'runs/prefix{seed}/latest.pt'
            ck = torch.load(path, map_location='cpu', weights_only=False)
            import_t1(p, ck, {'source': str(path), 'source_clock': ck['clock'],
                             'source_config': ck['config'], 'source_identity': ck['identity'],
                             'shared_T1_once': True, 'prefix_sha256': digest(path)})
            del ck
    else:
        raise ValueError('unknown worker action')
    target_task = 1 if action == 'prefix' else int(os.environ['TARGET_TASK'])
    original_visits = sum(v['visited'] for v in p.s['training_records'])
    original_comm = copy.deepcopy(p.s['communication'])
    current_t = p.s['task']
    round_start = time.monotonic()
    reload_checked = False
    while p.s['task'] < target_task:
        old_phase, old_round = p.s['phase'], p.s['round']
        if not p.step(): raise ValueError('unexpected early sequence completion')
        if old_phase == 'ready_evaluation':
            last = copy.deepcopy(p.s['stages'][-1])
            p.restore()
            if p.evaluate()['confusion'] != last['confusion']:
                raise ValueError('terminal reloaded confusion mismatch')
            reload_checked = True
        if old_phase == 'local_done':
            torch.cuda.synchronize()
            duration = time.monotonic()-round_start
            if old_round == 0:
                write(root/'reports'/f'profile_s{seed}_{arm}_T{current_t+1}.json',
                      {'seed': seed, 'arm': arm, 'task': current_t, 'first_round_including_setup_seconds': duration,
                       'formal_epochs': p.cfg['epochs'], 'clients': p.cfg['clients'], 'not_reduced': True})
            round_start = time.monotonic()
    if p.s['phase'] != 'task_start': raise ValueError('not a committed task boundary')
    if not reload_checked: raise ValueError('missing terminal readback')
    torch.cuda.synchronize()
    elapsed = time.monotonic()-start
    record = {'status': 'completed', 'action': action, 'seed': seed, 'arm': arm,
              'task': p.s['task'], 'stop_after_task': horizon, 'total_tasks': 10,
              'wall_seconds': elapsed, 'stages': p.s['stages'], 'matrix': p.s['matrix'],
              'AIA': sum(v['all_seen'] for v in p.s['stages'])/len(p.s['stages']),
              'T2_activity': p.s.get('bilateral_T2_activity', []),
              'additional_training_visits': sum(v['visited'] for v in p.s['training_records'])-original_visits,
              'additional_training_steps': sum(v['steps'] for v in p.s['training_records']
                                                if v['task'] >= current_t),
              'additional_communication': {k: v-original_comm.get(k, 0) for k, v in p.s['communication'].items()},
              'communication': p.s['communication'], 'loader_accesses': p.accesses,
              'route_records': p.s.get('route_records', []),
              'terminal_reload_confusion_matches': True,
              'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
              'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(),
              'prefix_provenance': p.s.get('prefix_provenance')}
    write(root/'reports'/f'{action}_s{seed}_{arm}_T{p.s["task"]}.json', record)
    print(json.dumps({'completed': action, 'seed': seed, 'arm': arm, 'task': p.s['task'], 'seconds': elapsed}), flush=True)


if __name__ == '__main__': main()
