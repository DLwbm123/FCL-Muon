"""Fresh T1-T10; changed RRS wire codec, frozen ordinary AdamW route/head."""
import copy
import json
import time
import torch
from research.adamw_t10.runtime import setup,build,validate
from research.adamw_t10.storage import write


def main():
    setup();root,f=validate();reports=root/'src/reports';job=json.loads((reports/'job.json').read_text())
    p=build({k:job[k] for k in ('started','deadline')})
    if (p.out/'latest.pt').exists():raise RuntimeError('fresh launch expected; audit any recovery')
    if hasattr(p,'response_server_operator') or 'bilateral' in p.cfg:raise ValueError('proximal operator present')
    baseline=json.loads((reports/'BASELINE_T10.json').read_text())
    dense_cost=json.loads((reports/'BASELINE_VALIDATION.json').read_text())['communication']
    start=time.monotonic()
    while p.s['phase']!='complete':
        old=p.s['phase'];task=p.s['task'];p.step()
        if old=='ready_evaluation':
            expected=copy.deepcopy(p.s['stages'][-1]);p.restore()
            if p.evaluate(False)!=expected:raise ArithmeticError('raw/centered checkpoint readback mismatch')
            stages=p.s['stages'];records=p.s['rrs_compression_records']
            record={'status':'evaluated_and_reload_verified','task':task+1,
                'raw_AIA':sum(s['all_seen'] for s in stages)/len(stages),
                'centered_AIA':sum(s['centered']['all_seen'] for s in stages)/len(stages),
                'stages':stages,'matrix':p.s['matrix'],'centered_matrix':[s['centered']['per_task'] for s in stages],
                'training_visits':sum(s['visited'] for s in p.s['training_records']),
                'training_steps':sum(s['steps'] for s in p.s['training_records']),
                'client_rounds':len(p.s['training_records']),'communication':p.s['communication'],
                'communication_phase':'evaluated before this task-end transmission; use COMPLETE for final totals',
                'route_records':p.s.get('route_records',[]),'rrs_compression_records':records,
                'loader_accesses_this_worker':p.accesses,'wall_seconds_this_worker':time.monotonic()-start,
                'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),
                'terminal_reload_both_readouts_match':True,'no_proximal_module':True,
                'wire_upload_bytes':sum(r['wire_bytes'] for r in records),
                'dense_reference_statistic_bytes':sum(r['dense_reference_tensor_bytes'] for r in records)}
            for read in ('raw','centered'):
                baseline_stages=[s if read=='raw' else s['centered'] for s in baseline['stages'][:task+1]]
                record['baseline_'+read+'_AIA']=sum(s['all_seen'] for s in baseline_stages)/len(baseline_stages)
                record[read+'_AIA_difference_pp']=(record[read+'_AIA']-record['baseline_'+read+'_AIA'])*100
            write(root,reports/f'T{task+1}.json',record)
            print(json.dumps({k:record[k] for k in ('task','raw_AIA','raw_AIA_difference_pp','wire_upload_bytes')}),flush=True)
    if (p.s['merges']!=list(range(10)) or p.s['history'].committed!=set(range(10))
            or len(p.s['training_records'])!=500 or sum(v['visited'] for v in p.s['training_records'])!=1250000):
        raise ValueError('incomplete formal ten-task protocol')
    p.restore();p.s['task']=9
    merged=p.evaluate(False);p.s['task']=10
    expected=p.s['stages'][-1]
    validation={'status':'passed','checkpoint_phase':'complete','all_tasks_merged_committed':True,'final_readback_images':10000,
        'raw_confusion_equal':merged['confusion']==expected['confusion'],
        'centered_confusion_equal':merged['centered']['confusion']==expected['centered']['confusion'],
        'merged_raw_accuracy':merged['all_seen'],'merged_centered_accuracy':merged['centered']['all_seen'],
        'communication':p.s['communication'],'recorded_real_dataset_train_visits':sum(v['visited'] for v in p.s['training_records']),
        'compressed_records':len(p.s['rrs_compression_records'])}
    if 'RRS_raw_upload' in p.s['communication'] or validation['compressed_records']!=1200:raise ValueError('incomplete compression')
    if (sum(r['wire_bytes'] for r in p.s['rrs_compression_records'])!=p.s['communication']['RRS_compressed_upload']):
        raise ValueError('encoded bytes mismatch')
    write(root,reports/'FINAL_VALIDATION.json',validation)
    raw_aia=sum(s['all_seen'] for s in p.s['stages'])/10
    complete={'status':'completed','tasks':10,'merges':p.s['merges'],'committed_tasks':sorted(p.s['history'].committed),
        'all_10_readbacks_verified':True,'wall_seconds':time.monotonic()-start,'final_checkpoint_phase':'complete',
        'raw_AIA':raw_aia,'baseline_raw_AIA':baseline['raw_AIA'],'raw_AIA_difference_pp':(raw_aia-baseline['raw_AIA'])*100,
        'final_communication':p.s['communication'],'baseline_final_communication':dense_cost,
        'RRS_wire_reduction_fraction':1-p.s['communication']['RRS_compressed_upload']/dense_cost['RRS_raw_upload'],
        'communication_limits':'formal logical payloads only; full distributed selector/outer-derivative and transport traffic not measured',
        'loader_accesses_this_worker':p.accesses}
    write(root,reports/'COMPLETE.json',complete)


if __name__=='__main__':main()
