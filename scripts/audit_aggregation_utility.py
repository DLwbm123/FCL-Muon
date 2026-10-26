"""One-time CPU audit of a completed arm; never performs another GPU evaluation."""
import json
import os
import time
from pathlib import Path
import torch


def main():
    root=Path(os.environ['PROJECT_ROOT']);mode=os.environ['AUDIT_MODE']
    assert mode in {'rrs','central','federated'}
    out=root/'runs'/mode;target=out/'FINAL_CHECK.json'
    if target.exists():
        result=json.loads(target.read_text());assert result['passed']
        print(json.dumps({'mode':mode,'existing_verified_audit':True,'checked':result['checked']}));return
    torch.set_num_threads(4)
    cfg=json.loads((root/'configs/aggregation_utility_three_task_seed0.json').read_text())
    job=json.loads((root/'reports/job.json').read_text());clock={k:job[k] for k in ['started','deadline']}
    q=json.loads((out/'QUALIFICATION.json').read_text());done=json.loads((out/'COMPLETE.json').read_text())
    c=torch.load(out/'latest.pt',map_location='cpu',weights_only=False);s=c['state']
    assert c['config']==dict(cfg,selection_mode=mode,output='runs/'+mode) and c['clock']==clock==q['clock']
    assert c['head_count']==3 and (s['task'],s['round'],s['phase'])==(3,0,'complete')
    assert s['merges']==[0,1,2] and s['history'].committed=={0,1,2} and all(c['merged_flags'].values())
    bs={k:v for k,v in c['model'].items() if k.endswith('.Bq') or k.endswith('.Bv')}
    assert len(bs)==24 and all(torch.count_nonzero(v)==0 for v in bs.values())
    records=s['training_records']
    assert len(records)==150 and len({(v['task'],v['round'],v['client']) for v in records})==150
    assert sum(v['visited'] for v in records)==375000 and sum(v['samples'] for v in s['stats_records'])==90000
    assert len(s['certificates'])==15 and all(v['deployed_fp32']['passed'] for v in s['certificates'])
    assert q['qualified'] and done['qualified']
    assert q['full_3000_checkpoint_reload_raw_predictions_equal'] and q['full_3000_checkpoint_reload_centered_predictions_equal']
    assert json.loads((out/'metrics.json').read_text())['stages']==s['stages']
    assert s['selection_records']==q['selection_records'] and len(s['selection_records'])==2
    manifest=json.loads((root/cfg['manifest']).read_text());selection_audits=[]
    for rec in s['selection_records']:
        t=rec['task'];private=torch.load(out/f'selection_t{t+1}_private.pt',map_location='cpu',weights_only=False)
        assert private['task']==t and private['mode']==mode and private['records']==rec['layers']
        assert rec['rng_restored'] and rec['formal_head_unchanged']
        assert rec['fit_samples']+rec['check_samples']==5000
        assert sum(rec['prototype_counts'])==rec['fit_samples']
        counts=[0]*cfg['cpt']
        for k,(fit,check) in enumerate(private['splits']):
            assert set(fit).isdisjoint(check) and sorted(fit+check)==sorted(manifest['partitions']['train'][t][k])
            assert len(fit)==rec['fit_client_counts'][k] and len(check)==rec['check_client_counts'][k]
            for i in fit:counts[manifest['labels']['train'][i]-t*cfg['cpt']]+=1
        assert counts==rec['prototype_counts']
        accepted=0;fallback=0
        for name,layer in rec['layers'].items():
            selected=private['selected_bases'][name]
            if layer['accepted']:
                assert all(layer['checks'].values()) and layer['single_check_no_research']
                assert torch.isfinite(selected).all()
                error=torch.linalg.norm(selected@selected.T-torch.eye(cfg['rank']))
                assert error<cfg['selection']['orthogonal_tolerance']
                accepted+=1
            else:
                assert torch.equal(selected,private['original_bases'][name]);fallback+=1
            if t==2:assert torch.equal(selected,c['model'][f'backbone.blocks.{name}.attn.qkv.A'])
        assert accepted==rec['accepted_layers'] and accepted+fallback==12
        selection_audits.append({'task':t+1,'accepted_layers':accepted,'exact_fallback_layers':fallback,
                                 'split_and_fit_prototype_counts_pass':True,'accepted_certificates_pass':True,
                                 'final_saved_A_matches_selection':True if t==2 else None})
        del private
    result={'checked':time.time(),'passed':True,'mode':mode,'clock':clock,'config_exact':True,
            'task':3,'round':0,'phase':'complete','heads':3,'merges':[0,1,2],'history':[0,1,2],
            'zero_B_count':len(bs),'unique_client_rounds':150,'training_visits_including_prefix':375000,
            'base_stats_visits':90000,'head_certificates':15,'selection_audits':selection_audits,
            'full3000_reload_raw_centered_equal':True,'maximum_reload_logit_difference':q['maximum_reload_logit_difference'],
            'completed':done,'timing_and_peak_scope':q['timing_and_peak_scope'],
            'new_branch_seconds':q['new_branch_seconds'],'peak_gpu_allocated_bytes':q['peak_gpu_allocated_bytes']}
    target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
