"""Real prefix GPU fork followed by the original synthetic response preflight."""
import copy
import gc
import json
import os
from pathlib import Path
import timm
import torch
from fcl.model import MuLoRAFL
from research.aggregation_utility.lifecycle import import_prefix
from research.route_compare_20261102.long_sequence_prefix import derive_prefix
from research.route_compare_20261102.pilot import RoutePilot
from research.route_compare_20261102.preflight import main as synthetic_preflight


def main():
    root = Path(os.environ['PROJECT_ROOT'])
    prior = Path(os.environ['PRIOR_ROOT'])
    frozen = json.loads((root / 'reports/FROZEN_FIVE_TASK.json').read_text())
    job = json.loads((root / 'reports/job.json').read_text())
    if frozen['status'] != 'frozen_before_new_test' or any(job[k] != frozen[k] for k in ('started', 'deadline')):
        raise ValueError('preflight clock or registration mismatch')
    oldjob = json.loads((prior / 'reports/job.json').read_text())
    oldclock = {k: oldjob[k] for k in ('started', 'deadline')}
    ck = torch.load(prior / 'runs/smoke_adamw/latest.pt', map_location='cpu', weights_only=False)
    cfg = copy.deepcopy(ck['config'])
    cfg.update(tasks=5, selection_mode='selected', output='runs/preflight_source')
    cfg['route']['mode'] = 'selected'
    model = MuLoRAFL(timm.create_model(cfg['backbone'], pretrained=False, num_classes=0),
                     100, cfg['rank'], cfg['cosine_scale']).cuda()
    clock = {k: job[k] for k in ('started', 'deadline')}
    p = RoutePilot(model, [None, None], {}, None, None, None, cfg, root / cfg['output'], ck['identity'], clock)
    fork = derive_prefix(ck, cfg, ck['identity'], oldclock, clock)
    import_prefix(p, fork)
    if len(ck['rng']['cuda']) != 1 or not torch.equal(torch.cuda.get_rng_state(), ck['rng']['cuda'][0]):
        raise ValueError('CUDA RNG restoration failed')
    if not all(a.merged and not torch.count_nonzero(a.Bq) and not torch.count_nonzero(a.Bv)
               for a in p.model.adapters.values()):
        raise ValueError('prefix not merged and zeroed')
    print(json.dumps({'prefix_gpu_strict_load': True, 'cuda_rng_restored': True,
                      'merged_B_zero': True, 'real_dataset_images': 0}), flush=True)
    del p, model, ck, fork
    gc.collect()
    torch.cuda.empty_cache()
    synthetic_preflight()


if __name__ == '__main__':
    main()
