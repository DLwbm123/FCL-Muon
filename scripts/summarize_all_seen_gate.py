"""Audit all three completed protected all-seen runs without training or selection."""
import json
from pathlib import Path
from summarize_actual_repetition import describe
from summarize_matrix_gate import FIELDS

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/all_seen_gate_20261030'
REFERENCE_FOLDERS={'ungated':'actual_update_repeat_20261029','previous_gate':'actual_loss_gate_20261029',
    'oriented':'actual_oriented_gate_20261030','spectral':'actual_tangent_gate_20261030','matrix':'actual_matrix_gate_20261030'}


def main():
    seeds={};references={k:{} for k in REFERENCE_FOLDERS};audit=[];gates={}
    for seed in range(3):
        key=str(seed);folder=OUT/f'seed{seed}'
        report=json.loads((folder/'FINAL_REPORT.json').read_text());seeds[key]=report
        preflight=json.loads((folder/'REAL_PREFLIGHT.json').read_text())
        assert preflight['passed'] and preflight['all_seen_objective_preflight']
        assert report['status']=='COMPLETED_QUALIFIED' and report['seed']==seed
        for name,path in REFERENCE_FOLDERS.items():
            references[name][key]=json.loads((ROOT/f'reports/{path}/seed{seed}/FINAL_REPORT.json').read_text())['arms']['global']
        for task in (3,5):
            q=json.loads((folder/f'global/QUALIFICATION_T{task}.json').read_text())
            assert q['qualified'] and q['config']['seed']==seed and q['client_rounds']==task*50
            assert q['config']['all_seen_gate']=={'current_loss_tolerance':1e-6}
            assert q['training_visits']==task*125000 and q['reload_images']==task*1000
            assert q['reload_raw_predictions_equal'] and q['reload_centered_predictions_equal']
            records=q['training_gate_records'];assert len(records)==len(q['actual_update_rounds'])==task*5
            clients=sum(len(v['baseline']['clients']) for v in records)
            assert q['logical_communication_bytes']['matrix_full_gradient_upload']==clients*24*768*32*4
            assert q['logical_communication_bytes']['all_seen_guard_scalar_upload']==clients*2*8
            assert q['extra_probe_backward_images']==q['gate_backward_images']==sum(v['backward_images'] for v in records)>0
            assert q['gate_student_forward_images']==sum(v['student_forward_images'] for v in records)
            assert q['gate_teacher_forward_images']==sum(v['teacher_forward_images'] for v in records)
            assert q['extra_probe_forward_images']==q['gate_student_forward_images']+q['gate_teacher_forward_images']
            assert q['gate_current_guard_rejected_rounds']==sum(not v['all_seen_objective']['current_guard_pass'] for v in records)
            for g,actual in zip(records,q['actual_update_rounds']):
                assert (g['task'],g['round'])==(actual['task'],actual['round'])
                assert g['baseline']['objective']==g['candidate']['objective']=='all_seen_ce_plus_old_kl_with_current_loss_diagnostic'
                delta=g['candidate']['weighted']['loss']-g['baseline']['weighted']['loss']
                cur=g['candidate']['weighted']['current_loss']-g['baseline']['weighted']['current_loss']
                assert abs(g['loss_difference']-delta)<1e-12 and abs(g['all_seen_objective']['current_loss_difference']-cur)<1e-12
                assert g['all_seen_objective']['primary_pass']==(delta< -1e-6)
                assert g['all_seen_objective']['current_guard_pass']==(cur<=1e-6)
                assert g['accepted']==(delta< -1e-6 and cur<=1e-6)
                ops=g['matrix_tangent']['operators'];assert len(ops)==24
                for op in ops:
                    assert 'mode_gradient' not in op and op['norm_relative_error']<=1e-10 and op['actual_FP32_norm_relative_error']<=5e-4
                    assert 0<=op['used_angle']<=op['reference_angle']+1e-12
                    if op['identity_reason'] is None:assert abs(op['used_angle']-op['reference_angle'])<=1e-12
                for name,parts in actual['layers'].items():
                    for i,v in enumerate(parts):
                        assert v['actual_FP32_norm_relative_error']<=5e-4
                        if g['accepted']:assert v==g['candidate_round']['layers'][name][i]
                        else:
                            assert v['identity_reason']=='training_loss_gate_rejected' and v['actual_FP32_relative_correction']==0
                            assert v['energy_after']==v['energy_before'] and v['singular_values_after']==v['singular_values_before']
            audit.append({'seed':seed,'task':task,'images':q['reload_images'],'maximum_logit_difference':q['maximum_reload_logit_difference'],'passed':True})
        gates[key]={}
        for task in range(1,6):
            records=[g for g in q['training_gate_records'] if g['task']==task]
            gates[key][str(task)]={'accepted':sum(g['accepted'] for g in records),'rounds':len(records),
                'primary_passes':sum(g['all_seen_objective']['primary_pass'] for g in records),
                'current_guard_rejections':sum(not g['all_seen_objective']['current_guard_pass'] for g in records),
                'all_seen_loss_delta':describe([g['loss_difference'] for g in records]),
                'original_loss_delta':describe([g['all_seen_objective']['current_loss_difference'] for g in records]),
                'identity_operators':sum(op['identity_reason'] is not None for g in records for op in g['matrix_tangent']['operators'])}
        for view in ('raw','centered'):
            for arm in ('control','global'):
                metrics=report['arms'][arm][view]
                for task,row in enumerate(metrics,1):
                    assert row['images']==task*1000 and row['W']+row['S']==row['errors']
                    assert abs(row['AIA_percent']-sum(v['overall_percent'] for v in metrics[:task])/task)<1e-10
            for task in range(1,6):
                pair=report['paired_vs_control'][str(task)][view]
                diff=report['arms']['global'][view][task-1]['overall_percent']-report['arms']['control'][view][task-1]['overall_percent']
                assert pair['corrected']-pair['destroyed']==pair['net_corrected'] and abs(diff-pair['net_corrected']*100/pair['images'])<1e-10
    references['control']={s:r['arms']['control'] for s,r in seeds.items()}
    differences={}
    for name,arms in references.items():
        differences[name]={view:{str(t):{k:describe([seeds[s]['arms']['global'][view][t-1][k]-arms[s][view][t-1][k] for s in seeds])
            for k in FIELDS if arms['0'][view][t-1][k] is not None} for t in range(1,6)} for view in ('raw','centered')}
    d=differences['control']['raw']['5']
    conditions={'mean_raw_gain_ge_0.2pp':d['overall_percent']['mean']>=.2-1e-10,
        'mean_AIA_nonnegative':d['AIA_percent']['mean']>=-1e-10,'mean_old_nonnegative':d['old_percent']['mean']>=-1e-10,
        'every_seed_new_ge_minus1pp':d['new_percent']['min']>=-1-1e-10,'mean_S_not_increased':d['S']['mean']<=0,
        'two_positive_seeds':sum(v>1e-10 for v in d['overall_percent']['values'])>=2,'all_qualified':len(audit)==6}
    for name in ('previous_gate','oriented','spectral','matrix'):
        conditions[f'mean_vs_{name}_raw_nonnegative']=differences[name]['raw']['5']['overall_percent']['mean']>=-1e-10
    cost={}
    for s,r in seeds.items():
        a=r['arms']['global'];cc=r['arms']['control']['cost']['logical_communication_bytes'];gc=a['cost']['logical_communication_bytes']
        cost[s]={'new_training_visits':a['cost']['training_visits'],'student_probe_forward_images':a['training_gate']['student_forward_images'],
            'teacher_probe_forward_images':a['training_gate']['teacher_forward_images'],'student_probe_backward_images':a['training_gate']['backward_images'],
            'additional_logical_communication_bytes':sum(gc.values())-sum(cc.values()),'additional_logical_communication_percent':100*(sum(gc.values())/sum(cc.values())-1),
            'all_seen_guard_scalar_upload_bytes':gc['all_seen_guard_scalar_upload']}
        preflight=json.loads((OUT/f'seed{s}/REAL_PREFLIGHT.json').read_text())
        for key in ('preflight_student_forward_images','preflight_student_backward_images','preflight_teacher_forward_images'):cost[s][key]=preflight[key]
    result={'status':'COMPLETED_QUALIFIED','registered_seeds':[0,1,2],'scientific_gate_passed':all(conditions.values()),'conditions':conditions,
        'per_seed':seeds,'reference_arms':references,'differences':differences,'gate_by_seed_task':gates,'qualification_audit':audit,'additional_cost':cost,
        'completion':json.loads((OUT/'COMPLETION.json').read_text()),
        'limitations':['Three exploratory seeds, one dataset and alpha, T5 only; no significance or T10 claim.',
        'Prior local validation had only three corrected images and prior exploratory exposure; it is not independent final evidence.',
        'Training probe guards are not old-image safety. Formal client training is unchanged; gradient messages and extra scalars are charged.',
        'Existing matched references reused. GPU wall times are not communication measurements or acceleration evidence.']}
    (OUT/'THREE_SEED_REPORT.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['全已见类目标与原目标保护门：三seed完整T5','',
        '结论：'+('通过全部冻结科学条件。仍限于本数据集/alpha/T5三seed探索，不作显著性或T10结论。' if result['scientific_gate_passed'] else '未通过全部冻结科学条件，保留阴性及所有权衡，不降低门槛或补seed。'),
        f"T5 raw对control均值{d['overall_percent']['mean']:+.6f} ± {d['overall_percent']['sample_std']:.6f}pp（样本SD）；各seed{d['overall_percent']['values']}。",
        '冻结条件：'+json.dumps(conditions,ensure_ascii=False),'',
        'seed | control raw | all-seen raw | Δold | Δnew | ΔAIA | 接受/25 | 原目标保护拒绝']
    for s,r in seeds.items():
        a=r['arms']['global'];v=r['T5_raw_differences_pp_except_S_count']
        lines.append(f"{s} | {r['arms']['control']['raw'][-1]['overall_percent']:.4f} | {a['raw'][-1]['overall_percent']:.4f} | {v['old_percent']:+.4f} | {v['new_percent']:+.4f} | {v['AIA_percent']:+.6f} | {a['training_gate']['accepted_rounds']} | {a['training_gate']['current_guard_rejected_rounds']}")
    lines+=['','全部任务/对照配对差均值±样本SD（pp）：']
    for name in differences:
        for view in ('raw','centered'):
            for t,row in differences[name][view].items():
                z=row['overall_percent'];lines.append(f"vs {name} {view} T{t}: {z['mean']:+.6f} ± {z['sample_std']:.6f}; {z['values']}")
    lines+=['','额外成本逐seed：'+json.dumps(cost,ensure_ascii=False),
        '总新训练1875000访问；6个T3/T5重载和两门/角度/实际谱/成本/配对审计通过。完成回执：'+json.dumps(result['completion'],ensure_ascii=False),
        '原五轮T5失败和旧局部诊断失败保持，不据本轮结果改写。完整raw/centered/old/new/AIA/遗忘/W+S/方向混淆、纠正破坏、两门及谱/成本见JSON；模型/ID/logits/梯度/完整日志留服务器。',
        '若本轮失败，关闭该同角矩阵候选家族继续试探；不得补seed、增角、降低条件重跑。后继主线必须是有独立依据的新机制假设。']
    (OUT/'THREE_SEED_REPORT_ZH.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'scientific_gate_passed':result['scientific_gate_passed'],'conditions':conditions,'T5':d,'cost':cost},indent=2))


if __name__=='__main__':main()
