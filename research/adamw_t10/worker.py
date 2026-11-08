"""Full ten-task run with readback at each evaluated checkpoint."""
import copy
import json
import time
import torch
from .runtime import setup, build, validate
from .storage import write


def main():
    setup();root,f=validate();job=json.loads((root/'src/reports/job.json').read_text())
    p=build({k:job[k] for k in ('started','deadline')})
    if (p.out/'latest.pt').exists():raise RuntimeError('fresh launch expected; explicit recovery audit required')
    if hasattr(p,'response_server_operator') or 'bilateral' in p.cfg:raise ValueError('proximal operator present')
    start=time.monotonic()
    reference=json.loads((root/'src/reports/LORM_REFERENCE.json').read_text())
    while p.s['phase']!='complete':
        old=p.s['phase'];task=p.s['task'];p.step()
        if old=='ready_evaluation':
            expected=copy.deepcopy(p.s['stages'][-1]);p.restore()
            actual=p.evaluate(save_artifacts=False)
            if actual!=expected:raise ArithmeticError('raw/centered checkpoint readback mismatch')
            stages=p.s['stages']
            record={'status':'evaluated_and_reload_verified','task':task+1,
                    'raw_AIA':sum(s['all_seen'] for s in stages)/len(stages),
                    'centered_AIA':sum(s['centered']['all_seen'] for s in stages)/len(stages),
                    'stages':stages,'matrix':p.s['matrix'],'centered_matrix':[s['centered']['per_task'] for s in stages],
                    'training_visits':sum(s['visited'] for s in p.s['training_records']),
                    'training_steps':sum(s['steps'] for s in p.s['training_records']),
                    'client_rounds':len(p.s['training_records']),'communication':p.s['communication'],
                    'route_records':p.s.get('route_records',[]),'loader_accesses_this_worker':p.accesses,
                    'wall_seconds_this_worker':time.monotonic()-start,
                    'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),
                    'terminal_reload_both_readouts_match':True,'no_proximal_module':True}
            record['historical_LoRM_same_seed_comparison']={}
            for read in ('raw','centered'):
                ours=record[read+'_AIA']*100
                history=reference['readouts'][read]['stage_accuracy'][:task+1]
                record['historical_LoRM_same_seed_comparison'][read]={
                    'tasks':task+1,'current_AIA_percent':ours,'LoRM_AIA_percent':sum(history)/len(history),
                    'difference_pp':ours-sum(history)/len(history),
                    'whole_scheme_historical_comparison_not_single_mechanism_ablation':True}
            write(root,root/f'src/reports/T{task+1}.json',record)
            print(json.dumps({'task':task+1,'raw_AIA':record['raw_AIA'],
                              'centered_AIA':record['centered_AIA'],'readback_verified':True}),flush=True)
    if (p.s['merges']!=list(range(10)) or p.s['history'].committed!=set(range(10))
            or len(p.s['training_records'])!=500
            or sum(s['visited'] for s in p.s['training_records'])!=1_250_000):
        raise ValueError('incomplete formal ten-task protocol')
    p.restore()
    write(root,root/'src/reports/COMPLETE.json',{
        'status':'completed','tasks':10,'merges':p.s['merges'],
        'committed_tasks':sorted(p.s['history'].committed),'all_10_readbacks_verified':True,
        'wall_seconds':time.monotonic()-start,'final_checkpoint_phase':p.s['phase']})


if __name__=='__main__':main()
