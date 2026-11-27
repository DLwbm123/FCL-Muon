"""Run both statistics schedules on identical two-task real-ViT inputs."""

import copy
import gc

import numpy as np
import torch

from fcl.checkpoint import rng_state
from research.adamw_t10.storage import write
from research.analytic_comparison.baselines import mathematical_checks
from research.stat_ablation.runtime import build, setup, validate


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(x, b[k]) for k, x in a.items())
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def main():
    setup()
    mathematical_checks()
    root, freeze = validate()
    clock = {k: freeze[k] for k in ('started', 'deadline')}
    generator = torch.Generator().manual_seed(772009)
    data = [(torch.randn(3,224,224,generator=generator), torch.tensor(i)) for i in range(20)]
    parts = [[list(range(task*10+k,(task+1)*10,2)) for k in range(2)] for task in range(2)]
    manifests = {'labels':{s:list(range(20)) for s in ('train','test')},
                 'partitions':{s:parts for s in ('train','test')}}

    def loader(dataset, indices, batch, shuffle=False):
        order = torch.randperm(len(indices)).tolist() if shuffle else range(len(indices))
        ordered = [dataset[indices[i]] for i in order]
        return [tuple(torch.stack(v) for v in zip(*ordered[j:j+batch]))
                for j in range(0,len(ordered),batch)]

    results = []
    for end_only in (False,True):
        pilot = build(clock, synthetic=([data,data],manifests,loader))
        pilot.cfg = copy.deepcopy(pilot.cfg)
        pilot.cfg.update(clients=2,epochs=1,rounds=3,batch=2,tasks=2,diagnostic_tasks=[],
                         task_end_statistics=end_only)
        pilot.out = root / 'src/runs' / ('check_end' if end_only else 'check_all')
        pilot.out.mkdir(exist_ok=False)
        while pilot.s['phase']!='complete':
            pilot.step()
        pilot.restore()
        pilot.s['task']=1
        if not equal(pilot.evaluate(False),pilot.s['stages'][-1]):
            raise ArithmeticError('merged synthetic official metrics changed')
        state = {key:copy.deepcopy(pilot.s[key]) for key in ('training_records','stat_branches','stages')}
        state.update(model={k:x.cpu().clone() for k,x in pilot.model.state_dict().items()},
                     history=copy.deepcopy(pilot.s['history'].raw),rng=rng_state())
        cost=copy.deepcopy(pilot.s['branch_statistics_upload'])
        results.append((state,cost))
        del pilot
        gc.collect()
        torch.cuda.empty_cache()
    a,b=results
    if any(not equal(a[0][key],b[0][key]) for key in a[0]):
        raise ArithmeticError('two-task schedules differ in model, history, RNG, training or predictions')
    if any(a[1][k]!=3*b[1][k] for k in a[1]):
        raise ArithmeticError('statistics traffic does not match three-to-one cadence')
    write(root,root/'src/reports/COMPLETE.json',dict(
        status='completed', real_viT_two_tasks_three_rounds=True,
        model_rng_training_history_branches_and_official_predictions_equal=True,
        shared_old_first_tier_preflight_and_cpu_checks='retained as prior evidence',
        statistics_upload_every_round=a[1],statistics_upload_task_end_only=b[1],
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved()))
    print('Two-task three-round equivalence passed', flush=True)


if __name__=='__main__':
    main()
