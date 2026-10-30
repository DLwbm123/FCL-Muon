"""One fixed all-seen objective candidate, evaluated without any training."""
import torch
from research.fed_spectra.alignment_diagnostic import measure
from research.fed_spectra.competition_diagnostic import measure_competition
from research.fed_spectra.decision_diagnostic import measure_decision_pair


def combined_gradient(original,competition):
    if len(original)!=len(competition):raise ValueError('Gradient lengths differ')
    out=[]
    for g,h in zip(original,competition):
        if g.shape!=h.shape:raise ValueError('Gradient shapes differ')
        v=g.double()+h.double()
        if not torch.isfinite(v).all():raise ArithmeticError('Nonfinite combined gradient')
        out.append(v)
    return out


def evaluate_candidate(p,selection,confirmation,baseline,candidate,references):
    results={}
    try:
        for name,probe in [('selection',selection),('confirmation',confirmation)]:
            with torch.no_grad():
                for b,v in zip(p.model.b_parameters(),candidate):b.copy_(v)
            objective,_=measure(p,probe)
            competition,_=measure_competition(p,probe)
            paired=measure_decision_pair(p,probe,baseline,candidate)
            coverage=paired['coverage'];base,base_c,original,original_c=references[name]
            assert coverage>=.95
            assert max(abs(paired['candidate_metrics'][k]-competition['weighted'][k]/coverage) for k in paired['candidate_metrics'])<=2e-6
            assert max(abs(paired['baseline_metrics'][k]-base_c['weighted'][k]/coverage) for k in paired['baseline_metrics'])<=2e-6
            t=paired['transition_fraction']
            assert abs(sum(t[0])-base_c['weighted']['all_seen_accuracy']/coverage)<1e-7
            assert abs(sum(v[0] for v in t)-competition['weighted']['all_seen_accuracy']/coverage)<1e-7
            base_all=(base['weighted']['loss']+base_c['weighted']['competition_ce'])/coverage
            original_all=(original['weighted']['loss']+original_c['weighted']['competition_ce'])/coverage
            new_all=(objective['weighted']['loss']+competition['weighted']['competition_ce'])/coverage
            results[name]={'objective':objective,'competition':competition,'paired_decisions':paired,
                'combined_objective_baseline':base_all,'combined_objective_original_candidate':original_all,'combined_objective_new_candidate':new_all,
                'combined_delta_vs_baseline':new_all-base_all,'combined_delta_vs_original':new_all-original_all,
                'original_objective_delta_vs_baseline':(objective['weighted']['loss']-base['weighted']['loss'])/coverage,
                'all_seen_accuracy_delta_vs_baseline':(competition['weighted']['all_seen_accuracy']-base_c['weighted']['all_seen_accuracy'])/coverage,
                'all_seen_accuracy_delta_vs_original':(competition['weighted']['all_seen_accuracy']-original_c['weighted']['all_seen_accuracy'])/coverage}
        return {'probes':results,
            'extra_student_forward_images':sum(x['objective']['student_forward_images']+x['competition']['student_forward_images']+x['paired_decisions']['student_forward_images'] for x in results.values()),
            'extra_teacher_forward_images':sum(x['objective']['teacher_forward_images'] for x in results.values()),
            'extra_student_backward_images':0}
    finally:
        with torch.no_grad():
            for b,v in zip(p.model.b_parameters(),baseline):b.copy_(v)


def local_conditions(rows):
    if len(rows)!=3 or sorted(r['seed'] for r in rows)!=[0,1,2]:raise ValueError('All three registered seeds required')
    points=[r['all_seen_candidate']['probes']['confirmation'] for r in rows]
    accuracy=[p['all_seen_accuracy_delta_vs_baseline'] for p in points]
    return {'coverage_each_ge_95percent':all(p['paired_decisions']['coverage']>=.95 for p in points),
        'combined_objective_improves_baseline_all':all(p['combined_delta_vs_baseline']< -1e-6 for p in points),
        'combined_objective_improves_original_all':all(p['combined_delta_vs_original']< -1e-6 for p in points),
        'original_objective_not_worse_all':all(p['original_objective_delta_vs_baseline']<=1e-6 for p in points),
        'all_seen_accuracy_not_worse_all':all(a>=-1e-10 for a in accuracy),
        'mean_all_seen_accuracy_strictly_improves':sum(accuracy)/3>1e-10,
        'engineering_checks':all(r['matrix_candidate_reproduced'] and r['heads_unchanged'] and r['baseline_restored'] and r['selection_confirmation_disjoint'] for r in rows)}
