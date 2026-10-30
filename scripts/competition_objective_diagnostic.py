"""Reuse common endpoints for a cross-task objective diagnostic; never train."""
import gc,json,os,signal,time
from pathlib import Path
import torch
from torchvision.datasets import CIFAR100
from fcl.checkpoint import restore_rng
from fcl.data import loader,transforms
from scripts.spectra_train import initialize_rng,new_model
from scripts.actual_alignment_diagnostic import DiagnosticPilot
from research.fed_spectra.alignment_diagnostic import measure,select_probes
from research.fed_spectra.direction_diagnostic import confirmation_probes
from research.fed_spectra.matrix_tangent import tangent_candidate,gradient_coverage
from research.fed_spectra.competition_diagnostic import measure_competition,gradient_alignment
from research.fed_spectra.lifecycle import write


class ReadOnlyPilot(DiagnosticPilot):
    def step(self):raise AssertionError('No new training allowed')
    def save(self):raise AssertionError('No checkpoint copies allowed')
    def budget(self):
        if time.time()>=self.clock['deadline']:raise TimeoutError('Original diagnostic clock exhausted')
        if time.time()-getattr(self,'storage_checked',0)>10:
            self.storage_checked=time.time();used=0
            for directory,_,files in os.walk(self.storage_root,followlinks=False):
                for name in files:
                    path=Path(directory)/name
                    if not path.is_symlink():used+=path.stat().st_blocks*512
            if used>=1_000_000_000:raise RuntimeError('Diagnostic write stop')


def run(root,seed,source,old,old_direction,old_tangent,old_matrix,clock,decision_audit=False,all_seen_candidate=False):
    initialize_rng(seed)
    x=torch.load(old/f'runs/seed{seed}/latest.pt',map_location='cpu',weights_only=False)
    pair=torch.load(old/f'runs/seed{seed}/pair_private.pt',map_location='cpu',weights_only=False)
    assert x['schema']=='cosine-head-regmean-pilot-1' and x['config']['seed']==seed
    assert x['state']['task']==3 and x['state']['phase']=='local_done' and x['state']['round']==0 and x['head_count']==4
    assert len(x['state']['training_records'])==160
    cfg=json.loads(json.dumps(x['config']));cfg['output']=f'private/seed{seed}'
    manifest=json.loads((source/cfg['manifest']).read_text())
    assert manifest['identity']=={k:cfg[k] for k in ['dataset','seed','alpha','clients']}
    manifest={'identity':manifest['identity'],'partitions':{'train':manifest['partitions']['train']},'labels':{'train':manifest['labels']['train']}}
    raw=CIFAR100(source/'data',train=True,download=False);assert list(raw.targets)==manifest['labels']['train']
    train_tf,test_tf=transforms();train_tf=train_tf.cuda();test_tf=test_tf.cuda()
    model=new_model(cfg,json.loads((source/'weights.json').read_text()))
    for _ in range(4):model.add_head(cfg['cpt'])
    model.load_state_dict(x['model'],strict=True)
    for name,a in model.adapters.items():a.merged=x['merged_flags'][name]
    assert all(not a.merged for a in model.adapters.values())
    p=ReadOnlyPilot(model,[raw,None],manifest,loader,train_tf,test_tf,cfg,root/cfg['output'],{'seed':seed},clock)
    p.storage_root=root;p.s=x['state'];restore_rng(x['rng']);del x
    assert all(torch.equal(a.detach().cpu(),b) for a,b in zip(model.b_parameters(),pair['origin_B']))
    baseline,positive=pair['baseline_B'],pair['corrected_B']
    assert all(torch.equal(a,b) for a,b in zip(p.s['agg_B'],baseline))
    with torch.no_grad():model.rows[-1].copy_(pair['head'])
    heads=[v.detach().clone() for v in model.rows]
    groups=manifest['partitions']['train'][3];selection=select_probes(groups,seed)
    confirm=confirmation_probes(groups,seed,selection)
    assert all(not set(a)&set(b) for a,b in zip(selection,confirm))
    coverage=sum(len(g) for g,c in zip(groups,confirm) if c)/sum(map(len,groups))
    if coverage==0:raise ValueError('No confirmation coverage')
    def deploy(values):
        with torch.no_grad():
            for b,v in zip(model.b_parameters(),values):b.copy_(v)
    deploy(baseline);fit_base,grad=measure(p,selection,True)
    fit_comp,fit_h=measure_competition(p,selection,True)
    generated=[tangent_candidate(o,b,v,g) for o,b,v,g in zip(pair['origin_B'],baseline,positive,grad)]
    selected=[v[0] for v in generated];operators=[v[1] for v in generated]
    all_seen_values=all_seen_operators=None
    if all_seen_candidate:
        from research.fed_spectra.all_seen_diagnostic import combined_gradient
        all_generated=[tangent_candidate(o,b,v,g) for o,b,v,g in zip(pair['origin_B'],baseline,positive,combined_gradient(grad,fit_h))]
        all_seen_values=[v[0] for v in all_generated];all_seen_operators=[v[1] for v in all_generated]
        assert all(abs(a['reference_angle']-b['reference_angle'])<=1e-12 for a,b in zip(operators,all_seen_operators))
    fit_alignment=gradient_alignment(grad,fit_h,baseline,selected);del fit_h
    deploy(selected);fit_candidate,_=measure(p,selection)
    fit_comp_candidate,_=measure_competition(p,selection)
    previous=json.loads((old/f'reports/seed{seed}.json').read_text())
    discrepancies={name:max(abs(value['weighted'][k]-previous[oldname]['weighted'][k]) for k in ('ce','kl','loss','accuracy'))
        for name,value,oldname in [('baseline',fit_base,'baseline')]}
    assert max(discrepancies.values())<=1e-6,discrepancies
    deploy(baseline);check_base,check_grad=measure(p,confirm,True)
    coverage_records=[gradient_coverage(o,b,g,c,v) for o,b,g,c,v in zip(pair['origin_B'],baseline,grad,check_grad,selected)]
    check_comp,check_h=measure_competition(p,confirm,True)
    check_alignment=gradient_alignment(check_grad,check_h,baseline,selected)
    del grad,check_grad,check_h
    prior=json.loads((old_direction/f'reports/seed{seed}.json').read_text())
    confirmation_reproduction=max(abs(check_base['weighted'][k]-prior['confirmation_baseline']['weighted'][k]) for k in check_base['weighted'])
    assert confirmation_reproduction<=1e-6
    assert abs(coverage-prior['confirmation_weight_coverage'])<=1e-12
    tangent_prior=json.loads((old_tangent/f'reports/seed{seed}.json').read_text())
    assert max(abs(fit_base['weighted'][k]-tangent_prior['selection_baseline']['weighted'][k]) for k in fit_base['weighted'])<=1e-6
    assert max(abs(check_base['weighted'][k]-tangent_prior['confirmation_baseline']['weighted'][k]) for k in check_base['weighted'])<=1e-6
    deploy(selected);check_selected,_=measure(p,confirm)
    check_comp_candidate,_=measure_competition(p,confirm)
    matrix_prior=json.loads((old_matrix/f'reports/seed{seed}.json').read_text())
    assert max(abs(fit_candidate['weighted'][k]-matrix_prior['selection_candidate']['weighted'][k]) for k in fit_candidate['weighted'])<=1e-6
    assert max(abs(check_selected['weighted'][k]-matrix_prior['confirmation_selected']['weighted'][k]) for k in check_selected['weighted'])<=1e-6
    assert all(torch.equal(a,b) for a,b in zip(heads,model.rows))
    difference={k:(check_selected['weighted'][k]-check_base['weighted'][k])/coverage for k in check_base['weighted']}
    deploy(baseline);assert all(torch.equal(a.detach().cpu(),b) for a,b in zip(model.b_parameters(),baseline))
    decision_rows=None
    if decision_audit:
        from research.fed_spectra.decision_diagnostic import measure_decision_pair
        decision_rows={}
        for name,probe,base,chosen in [('selection',selection,fit_comp,fit_comp_candidate),('confirmation',confirm,check_comp,check_comp_candidate)]:
            item=measure_decision_pair(p,probe,baseline,selected)
            for point,reference in [('baseline',base),('candidate',chosen)]:
                assert max(abs(item[f'{point}_metrics'][k]-reference['weighted'][k]/item['coverage']) for k in item[f'{point}_metrics'])<=2e-6
            t=item['transition_fraction']
            assert abs(sum(t[0])-base['weighted']['all_seen_accuracy']/item['coverage'])<1e-7
            assert abs(sum(v[0] for v in t)-chosen['weighted']['all_seen_accuracy']/item['coverage'])<1e-7
            decision_rows[name]=item
        assert all(torch.equal(a.detach().cpu(),b) for a,b in zip(model.b_parameters(),baseline))
        assert all(torch.equal(a,b) for a,b in zip(heads,model.rows))
    nonempty=sum(bool(g) for g in groups);bbytes=sum(b.numel()*4 for b in model.b_parameters())
    evaluations=[fit_base,fit_candidate,check_base,check_selected]
    row={'seed':seed,'status':'COMPLETED','task':4,'round':1,'new_training_visits':0,'test_images_used':0,
      'operators':operators,'gradient_coverage':coverage_records,
      'competition_selection_baseline':fit_comp,'competition_selection_candidate':fit_comp_candidate,
      'competition_confirmation_baseline':check_comp,'competition_confirmation_candidate':check_comp_candidate,
      'competition_selection_difference':{k:fit_comp_candidate['weighted'][k]-fit_comp['weighted'][k] for k in fit_comp['weighted']},
      'competition_confirmation_difference':{k:(check_comp_candidate['weighted'][k]-check_comp['weighted'][k])/coverage for k in check_comp['weighted']},
      'selection_gradient_alignment':fit_alignment,'confirmation_gradient_alignment':check_alignment,
      'competition_extra_student_forward_images':sum(v['student_forward_images'] for v in [fit_comp,fit_comp_candidate,check_comp,check_comp_candidate]),
      'competition_extra_student_backward_images':fit_comp['student_backward_images']+check_comp['student_backward_images'],
      'matrix_candidate_reproduced':True,'confirmation_weight_coverage':coverage,
      'confirmation_baseline_reproduction_max_difference':confirmation_reproduction,
      'selection_loss_difference':fit_candidate['weighted']['loss']-fit_base['weighted']['loss'],
      'old_gate_would_accept':fit_candidate['weighted']['loss']-fit_base['weighted']['loss'] < -1e-6,
      'confirmation_difference_vs_frozen_tangent':{k:difference[k]-tangent_prior['confirmation_normalized_difference'][k] for k in difference},
      'selection_baseline':fit_base,'selection_candidate':fit_candidate,
      'confirmation_baseline':check_base,'confirmation_selected':check_selected,'confirmation_normalized_difference':difference,
      'selection_confirmation_disjoint':True,'baseline_restored':True,'heads_unchanged':True,
      'reproduction_max_abs_difference':discrepancies,
      'student_forward_images':sum(v['student_forward_images'] for v in evaluations),
      'teacher_forward_images':sum(v['teacher_forward_images'] for v in evaluations),
      'student_backward_images':fit_base['backward_images']+check_base['backward_images'],
      'confirmation_forward_images':sum(v['student_forward_images']+v['teacher_forward_images'] for v in [check_base,check_selected]),
      'hypothetical_full_gradient_FP32_upload_bytes_per_probe':nonempty*bbytes,
      'communication_scope':'training-only single-process diagnostic; distributed execution not implemented',
      'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'elapsed_from_common_start':time.time()-clock['started']}
    if all_seen_candidate:
        from research.fed_spectra.all_seen_diagnostic import evaluate_candidate
        references={'selection':(fit_base,fit_comp,fit_candidate,fit_comp_candidate),'confirmation':(check_base,check_comp,check_selected,check_comp_candidate)}
        row['all_seen_candidate']=evaluate_candidate(p,selection,confirm,baseline,all_seen_values,references)
        row['all_seen_candidate']['operators']=all_seen_operators
        assert all(torch.equal(a.detach().cpu(),b) for a,b in zip(model.b_parameters(),baseline))
        assert all(torch.equal(a,b) for a,b in zip(heads,model.rows))
    if decision_rows is not None:
        row['decision_audit']=decision_rows
        row['decision_extra_student_forward_images']=sum(v['student_forward_images'] for v in decision_rows.values())
    row['peak_gpu_allocated_bytes']=torch.cuda.max_memory_allocated()
    row['elapsed_from_common_start']=time.time()-clock['started']
    write(root/f'reports/seed{seed}.json',row)
    del p,model,raw,train_tf,test_tf,pair;gc.collect();torch.cuda.empty_cache()
    return row


def main():
    root=Path(os.environ['PROJECT_ROOT']);origin=json.loads((root/'reports/ORIGIN.json').read_text())
    clock={'started':origin['started'],'deadline':origin['started']+2400}
    assert json.loads((root/'reports/CPU_TESTS.json').read_text())['passed']
    assert not (root/'reports/RESULT.json').exists()
    def stop(*_):raise TimeoutError('External timeout or signal')
    signal.signal(signal.SIGTERM,stop)
    private=json.loads((root/'sources_private.json').read_text());rows=[]
    for seed,source in enumerate(private['controls']):
        torch.cuda.reset_peak_memory_stats()
        write(root/'reports/PROGRESS.json',{'seed':seed,'phase':'loading','time':time.time()})
        rows.append(run(root,seed,Path(source),Path(private['diagnostic']),Path(private['direction']),Path(private['tangent']),Path(private['matrix']),clock,decision_audit=os.environ.get('DECISION_AUDIT')=='1',all_seen_candidate=os.environ.get('ALL_SEEN_CANDIDATE')=='1'))
    values=[r['confirmation_normalized_difference']['loss'] for r in rows]
    comp=[r['competition_confirmation_difference']['competition_ce'] for r in rows]
    all_ce=[r['competition_confirmation_difference']['all_seen_ce'] for r in rows]
    conditions={'coverage_each_ge_95percent':all(r['confirmation_weight_coverage']>=.95 for r in rows),
        'original_objective_improves_all':all(v< -1e-6 for v in values),
        'competition_worsens_at_least_two':sum(v>1e-6 for v in comp)>=2,
        'mean_competition_worsens':sum(comp)/3>1e-6,'engineering_checks':True}
    if os.environ.get('ALL_SEEN_CANDIDATE')=='1':
        from research.fed_spectra.all_seen_diagnostic import local_conditions
        new_conditions=local_conditions(rows)
        write(root/'reports/ALL_SEEN_RESULT.json',{'conditions':new_conditions,'local_decision_support':all(new_conditions.values()),'new_training_visits':0,'test_images_used':0,'seeds':[0,1,2]})
    write(root/'reports/RESULT.json',{'status':'COMPLETED','seeds':[0,1,2],'new_training_visits':0,'test_images_used':0,
        'objective_conflict_supported_at_these_endpoints':all(conditions.values()),'conditions':conditions,'confirmation_loss_differences':values,
        'mean_confirmation_loss_difference':sum(values)/3,'confirmation_competition_differences':comp,'confirmation_all_seen_ce_differences':all_ce,'elapsed_seconds':time.time()-clock['started'],'per_seed':rows,
        'limitations':['Both probe sets were used in local training; not a held-out generalization test.',
        'One shared T4 round per seed, not a five-task or old-class improvement.',
        'The candidate is reproduced unchanged from the prior matrix diagnostic. Competition gradients explain only, never choose a new candidate. Current-task training inputs cannot measure old-class accuracy or establish a cause for T5 seed differences.']})

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        write(Path(os.environ['PROJECT_ROOT'])/'reports/FAILURE.json',{'error':repr(exc),'time':time.time()});raise
