"""Unregistered guarded response runner draft. Do not launch before registration.

Requires an independently frozen public protocol and a tested resource supervisor.
"""
import copy
import json
import os
import time
from pathlib import Path

import torch
import timm
from torchvision.datasets import CIFAR100
from fcl.data import loader, transforms
from fcl.model import MuLoRAFL
from research.adamw_combined.bilevel_pilot_draft import GuardedBilevelPilot
from research.adamw_combined.guarded_prefix_draft import import_guarded_prefix

PARAMETERS = {'bilevel': {'split_seed': 42, 'client_cap': 2, 'trial_epochs': 1, 'step': .01},
              'holdout_seed': 17, 'holdout_fraction': .1}


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def run(root, data, prior, parameters):
    """Run paired arms only after the caller has enforced all launch gates."""
    root, data, prior = map(Path, (root, data, prior))
    job = json.loads((root / 'reports/job.json').read_text())
    clock = {k: job[k] for k in ('started', 'deadline')}
    if (clock['deadline'] - clock['started'] != 28800
            or not clock['started'] <= time.time() < clock['deadline']):
        raise TimeoutError('invalid or expired experiment clock')
    if parameters != PARAMETERS:
        raise ValueError('parameters differ from proposed frozen configuration')
    oldjob = json.loads((prior / 'reports/job.json').read_text())
    oldclock = {k: oldjob[k] for k in ('started', 'deadline')}
    prefix = torch.load(prior / 'runs/prefix/latest.pt', map_location='cpu', weights_only=False)
    state = prefix['state']
    if (prefix['clock'] != oldclock or prefix['config']['selection_mode'] != 'prefix'
            or (state['task'], state['round'], state['phase']) != (1, 0, 'task_start')
            or state['history'].committed != {0} or state['merges'] != [0]
            or len(state['training_records']) != 50
            or sum(r['visited'] for r in state['training_records']) != 125000
            or prefix['head_count'] != 1 or not all(prefix['merged_flags'].values())
            or state['pending'] or len(prefix['rng']['cuda']) != 1):
        raise ValueError('source prefix lifecycle mismatch')
    if any(torch.count_nonzero(v) for k, v in prefix['model'].items() if k.endswith(('.Bq', '.Bv'))):
        raise ValueError('source merged adapters not zero')
    expected = {'seed': 0, 'dataset': 'seq-cifar100', 'alpha': .05, 'clients': 10,
                'tasks': 3, 'rounds': 5, 'epochs': 5, 'batch': 16, 'rank': 32,
                'lr': .0002, 'aggregation': 'regmean'}
    if any(prefix['config'].get(k) != v for k, v in expected.items()):
        raise ValueError('source scientific configuration mismatch')
    manifest = json.loads((data / prefix['config']['manifest']).read_text())
    weights = json.loads((data / 'weights.json').read_text())
    if (manifest['identity'] != {k: expected[k] for k in ('dataset', 'seed', 'alpha', 'clients')}
            or prefix['identity']['weight_identity'] != weights['sha256']):
        raise ValueError('source metadata identity mismatch')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    # Existing CIFAR reader loads the full physical files; no claim of file-level isolation.
    raw = [CIFAR100(data / 'data', train=v, download=False) for v in (True, False)]
    if [list(x.targets) for x in raw] != [manifest['labels']['train'], manifest['labels']['test']]:
        raise ValueError('dataset labels mismatch')
    train_tf, test_tf = transforms()
    summaries = {}
    for mode in ('optimized', 'fixed_rrs'):
        cfg = copy.deepcopy(prefix['config'])
        cfg.update(output='runs/' + mode, selection_mode='prefix', server_correction=True)
        cfg['bilevel'] = dict(parameters['bilevel'], mode=mode)
        backbone = timm.create_model(cfg['backbone'], pretrained=False, num_classes=0)
        model = MuLoRAFL(backbone, 100, cfg['rank'], cfg['cosine_scale']).cuda()
        pilot = GuardedBilevelPilot(model, raw, manifest, loader, train_tf.cuda(), test_tf.cuda(), cfg,
                    root / 'runs' / mode, prefix['identity'], clock, holdout_tasks=[1, 2],
                    holdout_seed=parameters['holdout_seed'], holdout_fraction=parameters['holdout_fraction'])
        import_guarded_prefix(pilot, prefix, manifest, oldclock)
        if not torch.equal(torch.cuda.get_rng_state(), prefix['rng']['cuda'][0]):
            raise ValueError('source CUDA RNG mismatch')
        if not all(torch.equal(v, pilot.model.state_dict()[k].cpu()) for k, v in prefix['model'].items()):
            raise ValueError('source model mismatch')
        write(root / 'reports' / ('SOURCE_CHECK_' + mode + '.json'),
              {'passed': True, 'source_model_equal': True, 'source_cuda_rng_equal': True})
        while pilot.step():
            write(root / 'reports/PROGRESS.json', {'arm': mode, 'time': time.time(),
                  'task': pilot.s['task'], 'round': pilot.s['round'], 'phase': pilot.s['phase'],
                  'train_records': len(pilot.s['training_records'])})
        expected_new = cfg['rounds'] * cfg['epochs'] * sum(
            len(pilot.ids('train', t)) for t in (1, 2))
        records = pilot.s['training_records']
        if (len(records) != 150 or len({(r['task'], r['round'], r['client']) for r in records}) != 150
                or sum(r['visited'] for r in records if r['task'] > 0) != expected_new
                or pilot.s['history'].committed != {0, 1, 2}
                or len(pilot.s['selection_records']) != 2
                or len(pilot.s.get('combined_server_records', [])) != 10):
            raise ValueError('branch lifecycle or training visits mismatch')
        terminal = {k: v.detach().cpu().clone() for k, v in pilot.model.state_dict().items()}
        pilot.restore()
        if (pilot.s['phase'] != 'complete' or any(
                not torch.equal(v, pilot.model.state_dict()[k].cpu()) for k, v in terminal.items())):
            raise ValueError('terminal checkpoint reload mismatch')
        del terminal
        completed_responses = sum(c['completed'] for r in pilot.s['selection_records'] for c in r.get('calls', []))
        nclients = cfg['clients']
        basis_bytes = sum(a.A.numel() * a.A.element_size() for a in pilot.model.adapters.values())
        b_bytes = sum(b.numel() * b.element_size() for b in pilot.model.b_parameters())
        head_bytes = pilot.model.rows[-1].numel() * pilot.model.rows[-1].element_size()
        gram_bytes = pilot.model.backbone.num_features ** 2 * 8
        hypothetical = {'completed_responses': completed_responses,
            'basis_broadcast_bytes': completed_responses * nclients * basis_bytes,
            'B_upload_bytes': completed_responses * nclients * b_bytes,
            'head_upload_bytes': completed_responses * nclients * head_bytes,
            'fp64_gram_upload_bytes': completed_responses * nclients * gram_bytes,
            'meaning': 'hypothetical forward tensors for complete responses; no actual network or backward protocol measured'}
        summaries[mode] = {'stages': copy.deepcopy(pilot.s['stages']),
                           'selection': copy.deepcopy(pilot.s['selection_records']),
                           'server': copy.deepcopy(pilot.s['combined_server_records']),
                           'training_visits': expected_new,
                           'communication': copy.deepcopy(pilot.s['communication']),
                           'response_communication_hypothetical': hypothetical,
                           'holdout_reads': pilot.raw[0].holdout_reads,
                           'training_reads': pilot.raw[0].training_reads}
        write(root / 'reports' / ('PRIVATE_' + mode + '.json'), summaries[mode])
        del pilot, model, backbone
        torch.cuda.empty_cache()
    metrics = {name: {k: s['stages'][2][k] for k in ('all_seen', 'old_accuracy', 'new_accuracy', 'forgetting')}
               for name, s in summaries.items()}
    main, control = metrics['optimized'], metrics['fixed_rrs']
    checks = {'net_correct_at_least_6': round((main['all_seen']-control['all_seen'])*3000) >= 6,
              'old_not_lower': main['old_accuracy'] >= control['old_accuracy'],
              'new_loss_at_most_10': round((control['new_accuracy']-main['new_accuracy'])*1000) <= 10,
              'forgetting_not_higher': main['forgetting'] <= control['forgetting']+1e-12,
              'basis_changed': any(r['accepted_layers'] > 0 for r in summaries['optimized']['selection']),
              'server_accepted': any(r['accepted'] for r in summaries['optimized']['server'])}
    write(root / 'reports/RESULT.json', {'status': 'completed', 'metrics': metrics, 'checks': checks,
          'joint_passed': all(checks.values()), 'elapsed_seconds': time.time() - clock['started'],
          'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'source_test_previously_seen': True,
          'network_measured': False, 'test_images_basis': '2000 T2 plus 3000 T3 per arm'})


def main():
    root = Path(os.environ['PROJECT_ROOT'])
    job = json.loads((root / 'reports/job.json').read_text())
    registration = json.loads((root / 'reports/REGISTRATION.json').read_text())
    commit = registration.get('protocol_commit', '')
    if (len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit)
            or registration.get('public_protocol_verified') is not True
            or registration.get('parameters') != job.get('parameters')):
        raise ValueError('unverified registration or differing parameters')
    torch.cuda.set_per_process_memory_fraction(.45, 0)
    run(root, os.environ['SOURCE_ROOT'], os.environ['PRIOR_ROOT'], job['parameters'])


if __name__ == '__main__':
    main()
