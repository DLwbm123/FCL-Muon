"""One current-task round with validation excluded from every training path."""
import gc,json,os,signal,time
from pathlib import Path
import torch
from torchvision.datasets import CIFAR100
from fcl.checkpoint import restore_rng,atomic_save
from fcl.data import loader,transforms
from scripts.spectra_train import initialize_rng,new_model
from scripts.actual_alignment_diagnostic import DiagnosticPilot
from research.fed_spectra.alignment_diagnostic import measure,select_probes
from research.fed_spectra.competition_diagnostic import measure_competition
from research.fed_spectra.matrix_tangent import tangent_candidate
from research.fed_spectra.all_seen_diagnostic import combined_gradient,evaluate_candidate,local_conditions
from research.fed_spectra.decision_diagnostic import measure_decision_pair
from research.fed_spectra.heldout_round import split_clients,GuardedLoader
from research.fed_spectra.lifecycle import write


def run_seed(root,seed,source,clock):
    initialize_rng(seed)
    x=torch.load(source/'runs/control/task3.pt',map_location='cpu',weights_only=False)
    assert x['schema']=='cosine-head-regmean-pilot-1' and x['config']['seed']==seed
    assert x['state']['task']==3 and x['state']['phase']=='task_start' and x['head_count']==3
    assert x['config']['actual_update']['operator']['eta']==0 and len(x['state']['training_records'])==150
    cfg=json.loads(json.dumps(x['config']));cfg['actual_update']['operator']['eta']=.1
    cfg.update(output=f'runs/seed{seed}',maximum_seconds=6900,storage_limit_bytes=12000000000,write_stop_bytes=10000000000)
    manifest=json.loads((source/cfg['manifest']).read_text())
    assert manifest['identity']=={k:cfg[k] for k in ['dataset','seed','alpha','clients']}
    manifest={'identity':manifest['identity'],'partitions':{'train':manifest['partitions']['train']},'labels':{'train':manifest['labels']['train']}}
    original=manifest['partitions']['train'][3];train,validation=split_clients(original,seed)
    assert all(train) and cfg['epochs']==5 and cfg['clients']==10
    train_ids=sum(train,[]);validation_ids=sum(validation,[])
    assert not set(train_ids+validation_ids)&set(i for task in manifest['partitions']['train'][:3] for g in task for i in g)
    manifest['partitions']['train'][3]=train
    counts={'seed':seed,'train_counts':list(map(len,train)),'validation_counts':list(map(len,validation)),
        'expected_new_training_visits':5*len(train_ids),'validation_weight_coverage':sum(len(g) for g,v in zip(train,validation) if v)/len(train_ids)}
    frozen=json.loads((root/f'reports/SPLIT_seed{seed}.json').read_text());assert counts==frozen
    raw=CIFAR100(source/'data',train=True,download=False);assert list(raw.targets)==manifest['labels']['train']
    guard=GuardedLoader(loader,raw,train_ids,validation_ids)
    train_tf,test_tf=transforms();train_tf=train_tf.cuda();test_tf=test_tf.cuda()
    model=new_model(cfg,json.loads((source/'weights.json').read_text()))
    for _ in range(3):model.add_head(cfg['cpt'])
    model.load_state_dict(x['model'],strict=True)
    for name,a in model.adapters.items():a.merged=x['merged_flags'][name]
    assert all(a.merged for a in model.adapters.values())
    p=DiagnosticPilot(model,[raw,None],manifest,guard,train_tf,test_tf,cfg,root/cfg['output'],{'seed':seed,'source_tasks':3,'split':'fixed_current_80_20'},clock)
    p.storage_root=root;p.s=x['state'];restore_rng(x['rng']);del x
    write(p.out/'manifest_private.json',manifest);write(p.out/'split_private.json',{'training':train,'validation':validation})
    probes=select_probes(train,seed);assert not set(sum(probes,[]))&set(validation_ids)
    visits0=sum(v['visited'] for v in p.s['training_records']);teacher0=p.s['anchor_forward_images'];stats0=len(p.s['stats_records']);spectra0=len(p.s['spectra_probes'])
    while p.s['phase']!='local_done':
        assert p.s['task']==3 and p.s['round']==0;p.step()
        write(root/'reports/PROGRESS.json',{'seed':seed,'phase':p.s['phase'],'new_client_rounds':len(p.s['training_records'])-150,'new_training_visits':sum(v['visited'] for v in p.s['training_records'])-visits0,'time':time.time()})
    visits=sum(v['visited'] for v in p.s['training_records'])-visits0
    assert len(p.s['training_records'])==160 and visits==counts['expected_new_training_visits']
    assert guard.validation_access_images==guard.validation_model_forward_images==guard.training_validation_accesses==0
    baseline=[v.clone() for v in p.s['agg_B']];origin=[b.detach().cpu().clone() for b in model.b_parameters()]
    p.fuse();positive=[b.detach().cpu().clone() for b in model.b_parameters()]
    assert p.s['task']==3 and p.s['round']==1
    # Create the anchor before installing a student-only hook; deepcopy must not copy loader/data guards.
    p.get_teacher();hook=model.register_forward_pre_hook(guard.model_guard)
    training_stage_loader_access_images=guard.training_access_images
    heads=[h.detach().clone() for h in model.rows]
    def deploy(values):
        with torch.no_grad():
            for b,v in zip(model.b_parameters(),values):b.copy_(v)
    deploy(baseline);fit,grad=measure(p,probes,True);fit_c,h=measure_competition(p,probes,True)
    old_pairs=[tangent_candidate(o,b,v,g) for o,b,v,g in zip(origin,baseline,positive,grad)]
    new_pairs=[tangent_candidate(o,b,v,g) for o,b,v,g in zip(origin,baseline,positive,combined_gradient(grad,h))]
    original_candidate=[v[0] for v in old_pairs];new_candidate=[v[0] for v in new_pairs];del grad,h
    assert all(abs(a[1]['reference_angle']-b[1]['reference_angle'])<=1e-12 for a,b in zip(old_pairs,new_pairs))
    guard.allow_validation=True
    deploy(baseline);val,val_c=measure(p,validation)[0],measure_competition(p,validation)[0]
    deploy(original_candidate);fit_old,fit_old_c=measure(p,probes)[0],measure_competition(p,probes)[0]
    val_old,val_old_c=measure(p,validation)[0],measure_competition(p,validation)[0]
    paired_original={'selection':measure_decision_pair(p,probes,baseline,original_candidate),'confirmation':measure_decision_pair(p,validation,baseline,original_candidate)}
    refs={'selection':(fit,fit_c,fit_old,fit_old_c),'confirmation':(val,val_c,val_old,val_old_c)}
    for label,pair in paired_original.items():
        _,base_comp,_,candidate_comp=refs[label]
        for point,reference in [('baseline',base_comp),('candidate',candidate_comp)]:
            assert max(abs(pair[f'{point}_metrics'][k]-reference['weighted'][k]/pair['coverage']) for k in pair[f'{point}_metrics'])<=2e-6
    new=evaluate_candidate(p,probes,validation,baseline,new_candidate,refs)
    assert all(torch.equal(a.detach().cpu(),b) for a,b in zip(model.b_parameters(),baseline)) and all(torch.equal(a,b) for a,b in zip(heads,model.rows))
    assert guard.validation_grad_violations==guard.training_validation_accesses==0
    assert guard.validation_model_forward_images==10*len(validation_ids)
    original_evals=[fit,val,fit_old,val_old];comp_evals=[fit_c,val_c,fit_old_c,val_old_c]
    allcost={'student_forward_images':sum(v['student_forward_images'] for v in original_evals+comp_evals)+sum(v['student_forward_images'] for v in paired_original.values())+new['extra_student_forward_images'],
      'teacher_forward_images':sum(v['teacher_forward_images'] for v in original_evals)+new['extra_teacher_forward_images'],
      'student_backward_images':fit['backward_images']+fit_c['student_backward_images']}
    new_stats=p.s['stats_records'][stats0:];new_spectra=p.s['spectra_probes'][spectra0:]
    row={'seed':seed,'status':'COMPLETED','source_tasks_reused':3,'new_training_visits':visits,'new_client_rounds':10,'test_images_used':0,
      'split_counts':counts,'new_training_teacher_forward_images':p.s['anchor_forward_images']-teacher0,
      'new_stats_forward_images':sum(v['samples'] for v in new_stats),'new_stats_records':new_stats,'new_spectra_probe_records':new_spectra,'candidate_probe_cost':allcost,
      'selection_baseline':fit,'selection_baseline_competition':fit_c,'validation_baseline':val,'validation_baseline_competition':val_c,
      'selection_original_candidate':fit_old,'selection_original_candidate_competition':fit_old_c,'validation_original_candidate':val_old,'validation_original_candidate_competition':val_old_c,
      'original_candidate_paired':paired_original,'original_candidate_operators':[v[1] for v in old_pairs],
      'all_seen_candidate':new,'all_seen_candidate_operators':[v[1] for v in new_pairs],
      'engineering_passed':True,'heads_unchanged':True,'baseline_restored':True,'selection_confirmation_disjoint':True,
      'validation_training_or_stats_accesses':guard.training_validation_accesses,'validation_grad_violations':guard.validation_grad_violations,
      'validation_student_forward_images_guard':guard.validation_model_forward_images,'loader_training_side_access_images':guard.training_access_images,
      'loader_validation_access_images':guard.validation_access_images,'training_stage_loader_access_images':training_stage_loader_access_images,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'elapsed_seconds_from_common_start':time.time()-clock['started']}
    # Preserve one endpoint only, for audit/recovery; no candidate continues training.
    p.budget();atomic_save({'origin_B':origin,'baseline_B':baseline,'positive_B':positive,'head':heads[-1].cpu()},p.out/'pair_private.pt')
    write(root/f'reports/seed{seed}.json',row)
    hook.remove();del p,model,raw,train_tf,test_tf;gc.collect();torch.cuda.empty_cache()
    return row


def main():
    root=Path(os.environ['PROJECT_ROOT']);origin=json.loads((root/'reports/ORIGIN.json').read_text());clock={'started':origin['started'],'deadline':origin['started']+6900}
    assert json.loads((root/'reports/CPU_TESTS.json').read_text())['passed'] and not (root/'reports/RESULT.json').exists()
    def stop(*_):raise TimeoutError('Original shared deadline reached')
    signal.signal(signal.SIGTERM,stop);rows=[]
    for seed,source in enumerate(json.loads((root/'sources_private.json').read_text())):
        torch.cuda.reset_peak_memory_stats();rows.append(run_seed(root,seed,Path(source),clock))
    # Reuse exactly the frozen numeric conditions; engineering proof comes from this new shared endpoint.
    conditions=local_conditions([dict(r,matrix_candidate_reproduced=r['engineering_passed']) for r in rows])
    write(root/'reports/RESULT.json',{'status':'COMPLETED','seeds':[0,1,2],'new_training_visits':sum(r['new_training_visits'] for r in rows),
        'test_images_used':0,'conditions':conditions,'heldout_local_decision_support':all(conditions.values()),'per_seed':rows,
        'elapsed_seconds':time.time()-clock['started'],
        'limitations':['Validation inputs excluded from this run training/statistics/gradients; prior exploratory exposure and pretraining overlap are not ruled out.','One shared T4 round, not full T5 or old-class safety. No automatic full training expansion.']})

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        write(Path(os.environ['PROJECT_ROOT'])/'reports/FAILURE.json',{'error':repr(exc),'time':time.time()});raise
