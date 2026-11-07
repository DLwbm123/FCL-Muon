"""Register the complete protocol before any new real-data response/test."""
import copy
import json
import os
import time
from pathlib import Path
import torch
import timm
import kornia
import fcl.data as data
from research.response_bilateral.runtime import digest, roots, write


def freeze(job):
    root, source, prior = roots()
    protocol = json.loads((root/'research/response_bilateral/PROTOCOL.json').read_text())
    old = json.loads((prior/'reports/FROZEN.json').read_text())
    weights = json.loads((source/'weights.json').read_text())
    if digest(weights['path']) != old['weight_sha256'] or weights['sha256'] != old['weight_sha256']:
        raise ValueError('original pretrained source mismatch')
    ckpath = prior/'runs/smoke_adamw/latest.pt'
    ck = torch.load(ckpath, map_location='cpu', weights_only=False)
    route = {k: v for k, v in ck['config']['route'].items() if k not in ('mode', 'optimizer')}
    if route != old['route']: raise ValueError('original route drift')
    cfg = copy.deepcopy(ck['config'])
    cfg.update(maximum_seconds=86400, storage_limit_bytes=19000000000, write_stop_bytes=18500000000)
    base = json.loads((source/'manifests/seq-cifar100_n10_a0.05_s0.json').read_text())
    if digest(source/'manifests/seq-cifar100_n10_a0.05_s0.json') != old['manifest_sha256']['0']:
        raise ValueError('original manifest changed')
    if base['class_order'] != list(range(100)) or len(base['partitions']['train']) != 10:
        raise ValueError('expected fixed ten-task class order')
    dest = root/'manifests'; dest.mkdir(exist_ok=True)
    data.UPSTREAM = source/'sources/fed-mammoth'
    upstream = data.UPSTREAM/'_datasets/_utils.py'
    manifest_sha = {}
    for seed in (0, 3, 4, 5):
        m = copy.deepcopy(base)
        if seed:
            m['identity']['seed'] = seed
            m['partitions'] = data.upstream_split(m['labels']['train'], m['labels']['test'], 10, 10, .05, seed)
        for split in ('train', 'test'):
            ids = [i for task in m['partitions'][split] for group in task for i in group]
            if sorted(ids) != list(range(len(m['labels'][split]))): raise ValueError('invalid full partition')
            if len(m['partitions'][split]) != 10: raise ValueError('truncated task partition')
            for t, groups in enumerate(m['partitions'][split]):
                if len(groups) != 10 or any(not t*10 <= m['labels'][split][i] < (t+1)*10 for g in groups for i in g):
                    raise ValueError('task/client labels invalid')
        path = dest/f'seed{seed}.json'
        if path.exists(): raise ValueError('existing manifest requires audit')
        write(path, m); manifest_sha[str(seed)] = digest(path)
    sources = {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*.py'))
               if '__pycache__' not in p.parts and not p.name.startswith('._')}
    for name in ('PROTOCOL.json', 'METHOD.md', 'SOURCE_ADAPTATION.md', 'TASKBOOK.txt'):
        p = root/'research/response_bilateral'/name; sources[str(p.relative_to(root))] = digest(p)
    import hashlib
    code_identity = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    assets = [Path(weights['path']), ckpath, upstream]
    record = {'status': 'frozen_before_new_real_data', 'registered_at': time.time(),
        **{k: job[k] for k in ('started', 'deadline')}, 'protocol': protocol,
        'anchor_git': {'implementation_checkout_HEAD': '0b11d1a366715ce22da4762b83c4cf0bae6dd391',
                       'route_compare_HEAD': 'd12175f573a9619f04528a9f3a66405b1dfc98b6',
                       'dirty_state': 'existing untracked directories preserved; source snapshot is authoritative'},
        'code_identity': code_identity, 'source_sha256': sources,
        'base_config': cfg, 'route_mode': 'selected', 'optimizer': 'adamw', 'route': route,
        'weights': weights, 'prefix_sha256': digest(ckpath), 'prefix_source_config': ck['config'],
        'prefix_source_identity': ck['identity'], 'prefix_source_clock': ck['clock'],
        'manifest_sha256': manifest_sha, 'split_generator_sha256': digest(upstream),
        'asset_stat': {str(p): [p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ino] for p in assets},
        'environment': {'torch': str(torch.__version__), 'timm': str(timm.__version__),
                        'kornia': str(kornia.__version__), 'cuda': torch.version.cuda},
        'bootstrap_sha256': {n: digest(root.parent/n) for n in ('settings.json', 'driver.py', 'worker.py')},
        'source_original_frozen_sha256': digest(prior/'reports/FROZEN.json'),
        'frozen_before_scores': True, 'private_delivery': True}
    if (root/'reports/FROZEN_PROTOCOL.json').exists(): raise ValueError('protocol already registered')
    write(root/'reports/FROZEN_PROTOCOL.json', record)
    return record
