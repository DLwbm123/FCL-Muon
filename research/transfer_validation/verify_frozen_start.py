"""Validate the saved checkpoints and run the independent T1 frozen control."""
import json
import os
from pathlib import Path

from research.adamw_t10.storage import write
from research.stat_ablation.runtime import build, setup, validate
from research.transfer_validation import frozen_stage


def read(path):
    return json.loads(path.read_text())


def compare(pilot, source, stage):
    checkpoint = source/'src/runs/rrs_shared'/('T1.pt' if stage==1 else 'latest.pt')
    endpoint = frozen_stage.restore_model(pilot, checkpoint)
    if endpoint['task']!=stage or endpoint['merges']!=list(range(stage)):
        raise ValueError('Checkpoint does not represent the requested committed stage')
    receipt = read(source/f'src/reports/rrs_shared/T{stage}.json')
    if receipt['status']!='evaluated_and_reload_verified' or receipt['task']!=stage:
        raise ValueError('The formal stage reload receipt is missing')
    before = receipt['stages'][-1]['branch_readouts']
    if before!=endpoint['stages'][-1]['branch_readouts']:
        raise ValueError('Stored stage results differ from the official readout')
    pilot.s = endpoint
    pilot.s['task'] = stage-1
    pilot.s['conditional_readouts'] = {}
    pilot.s['evaluator_old_anchor'] = None
    return before,pilot.evaluate(False)['branch_readouts']


def main():
    setup()
    root,freeze = validate()
    source = Path(os.environ['INPUT_CHECKPOINT']).parents[3]
    if freeze['config']['seed']!=1 or not source.is_dir():
        raise ValueError('Expected the frozen seed 1 checkpoint')
    clock = {k:freeze[k] for k in ('started','deadline')}
    before,after = compare(build(clock),source,1)
    if before!=after:
        raise ValueError('T1 merged checkpoint predictions differ from the saved stage')
    write(root,root/'T1_GATE.json',dict(status='passed',strict_checkpoint_identity=True,
        stage=1,manifest_sha256=freeze['manifest_sha256'],
        raw_and_centered_all_branch_readouts_equal=True,raw_accuracy=after['full']['all_seen'],
        test_samples=sum(map(sum,after['full']['confusion']))))
    print('T1 checkpoint verified',flush=True)
    before,after = compare(build(clock),source,10)
    delta={}
    for branch in before:
        delta[branch]={}
        for variant in ('raw','centered'):
            left,right=(before[branch],after[branch]) if variant=='raw' else (
                before[branch]['centered'],after[branch]['centered'])
            u,v=left['confusion'],right['confusion']
            if [sum(row) for row in u]!=[sum(row) for row in v]:
                raise ValueError('Evaluation sample coverage changed')
            cells=[dict(label=i,predicted=j,change=b-a) for i,(ur,vr) in enumerate(zip(u,v))
                   for j,(a,b) in enumerate(zip(ur,vr)) if a!=b]
            delta[branch][variant]=dict(confusion_equal=not cells,changed_cells=cells,
                per_class_correct_equal=all(u[i][i]==v[i][i] for i in range(len(u))),
                accuracy_delta_pp=100*(right['all_seen']-left['all_seen']),
                min_reassigned_samples=sum(abs(c['change']) for c in cells)//2)
    write(root,root/'T10_DIAGNOSTIC.json',dict(merged_checkpoint_reevaluated=True,
        previous_original_gate='failed',changes=delta))
    print('T10 diagnostic recorded',flush=True)
    frozen_stage.main()


if __name__=='__main__':
    main()
