"""Validate and summarize every registered gate seed without training or selection."""
import json, statistics as st
from pathlib import Path
from summarize_actual_repetition import describe

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/actual_loss_gate_20261029'
FIELDS=['overall_percent','old_percent','new_percent','AIA_percent','forgetting_pp','signed_forgetting_pp','W','S','old_to_new_percent','new_to_old_percent']

def main():
    seeds={}; audit=[]; gates={}; old={}
    for seed in range(3):
        folder=OUT/f'seed{seed}'
        r=json.loads((folder/'FINAL_REPORT.json').read_text())
        c=json.loads((folder/'COMPLETION.json').read_text())
        assert c['status']=='completed' and c['qualified'] and c['exit_codes']=={'global':0}
        assert r['status']=='COMPLETED_QUALIFIED' and r['seed']==seed and r['comparison_control_reused']
        seeds[str(seed)]=r
        source=ROOT/'reports/actual_update_global_20261029' if seed==0 else ROOT/f'reports/actual_update_repeat_20261029/seed{seed}'
        old[str(seed)]=json.loads((source/'FINAL_REPORT.json').read_text())['arms']['global']
        for task in (3,5):
            q=json.loads((folder/f'global/QUALIFICATION_T{task}.json').read_text())
            assert q['qualified'] and q['config']['seed']==seed and q['client_rounds']==task*50
            assert q['training_visits']==task*125000 and q['reload_images']==task*1000
            assert q['reload_raw_predictions_equal'] and q['reload_centered_predictions_equal']
            assert len(q['training_gate_records'])==len(q['actual_update_rounds'])==task*5
            assert q['extra_probe_backward_images']==0
            assert q['extra_probe_forward_images']==q['gate_student_forward_images']+q['gate_teacher_forward_images']
            assert q['gate_student_forward_images']==sum(v['student_forward_images'] for v in q['training_gate_records'])
            assert q['gate_teacher_forward_images']==sum(v['teacher_forward_images'] for v in q['training_gate_records'])
            for g,actual in zip(q['training_gate_records'],q['actual_update_rounds']):
                assert (g['task'],g['round'])==(actual['task'],actual['round'])
                assert abs(g['loss_difference']-(g['candidate']['weighted']['loss']-g['baseline']['weighted']['loss']))<1e-12
                assert g['accepted']==(g['loss_difference']< -1e-6)
                for name,layer in actual['layers'].items():
                    for i,v in enumerate(layer):
                        assert v['actual_FP32_norm_relative_error']<=5e-4
                        if g['accepted']:assert v==g['candidate_round']['layers'][name][i]
                        else:
                            assert v['identity_reason']=='training_loss_gate_rejected' and v['actual_FP32_relative_correction']==0
                            assert v['energy_after']==v['energy_before'] and v['singular_values_after']==v['singular_values_before']
            audit.append({'seed':seed,'task':task,'images':q['reload_images'],'max_logit_difference':q['maximum_reload_logit_difference'],'passed':True})
        gates[str(seed)]={}
        for t in range(1,6):
            records=[v for v in q['training_gate_records'] if v['task']==t]
            gates[str(seed)][str(t)]={'accepted':sum(v['accepted'] for v in records),'rounds':len(records),
                'loss_delta':describe([v['loss_difference'] for v in records]),
                'ce_delta':describe([v['candidate']['weighted']['ce']-v['baseline']['weighted']['ce'] for v in records]),
                'kl_delta':describe([v['candidate']['weighted']['kl']-v['baseline']['weighted']['kl'] for v in records])}
        for view in ('raw','centered'):
            for arm in ('control','global'):
                rows=r['arms'][arm][view]
                for t,row in enumerate(rows,1):
                    assert row['images']==t*1000 and row['W']+row['S']==row['errors']
                    assert abs(row['AIA_percent']-st.mean(v['overall_percent'] for v in rows[:t]))<1e-10
            for t in range(1,6):
                p=r['paired_vs_control'][str(t)][view]
                d=r['arms']['global'][view][t-1]['overall_percent']-r['arms']['control'][view][t-1]['overall_percent']
                assert p['corrected']-p['destroyed']==p['net_corrected'] and abs(d-p['net_corrected']*100/p['images'])<1e-10
    differences={}
    for reference in ('control','ungated'):
        differences[reference]={}
        for view in ('raw','centered'):
            differences[reference][view]={}
            for t in range(1,6):
                refs={s:r['arms']['control'] if reference=='control' else old[s] for s,r in seeds.items()}
                differences[reference][view][str(t)]={k:describe([r['arms']['global'][view][t-1][k]-refs[s][view][t-1][k] for s,r in seeds.items()]) for k in FIELDS if refs['0'][view][t-1][k] is not None}
    d=differences['control']['raw']['5']
    conditions={'mean_raw_gain_ge_0.2pp':d['overall_percent']['mean']>=.2-1e-10,
      'mean_AIA_nonnegative':d['AIA_percent']['mean']>=-1e-10,'mean_old_nonnegative':d['old_percent']['mean']>=-1e-10,
      'every_seed_new_ge_minus1pp':d['new_percent']['min']>=-1-1e-10,'mean_S_not_increased':d['S']['mean']<=0,
      'two_positive_seeds':sum(v>1e-10 for v in d['overall_percent']['values'])>=2,
      'all_qualified':len(audit)==6,'mean_gated_vs_ungated_raw_nonnegative':differences['ungated']['raw']['5']['overall_percent']['mean']>=-1e-10}
    cost={}
    for s,r in seeds.items():
        a=r['arms']['global'];cc=r['arms']['control']['cost']['logical_communication_bytes'];gc=a['cost']['logical_communication_bytes']
        cost[s]={'new_training_visits':a['cost']['training_visits'],'student_probe_forward_images':a['training_gate']['student_forward_images'],
                 'teacher_probe_forward_images':a['training_gate']['teacher_forward_images'],
                 'additional_logical_communication_bytes':sum(gc.values())-sum(cc.values()),
                 'additional_logical_communication_percent':100*(sum(gc.values())/sum(cc.values())-1)}
    result={'status':'COMPLETED_QUALIFIED','scientific_gate_passed':all(conditions.values()),'conditions':conditions,
      'registered_seeds':[0,1,2],'per_seed':seeds,'ungated_arms':old,'differences':differences,'gate_by_seed_task':gates,
      'qualification_audit':audit,'additional_cost':cost,'completion':json.loads((OUT/'COMPLETION.json').read_text()),
      'limitations':['Three seeds, one dataset/alpha/T5, exploratory; no significance or T10 claim.',
      'Training-internal probes do not guarantee generalization or old-class safety.',
      'Control and ungated runs reused; their training is not new cost; different GPU wall times do not measure acceleration.',
      'All decisions and candidate versus deployed spectra are retained in per-seed qualifications.']}
    (OUT/'THREE_SEED_REPORT.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['训练目标接受门：三seed五任务最终报告','',
      '结论：工程资格全部通过，但未通过冻结科学门槛。原无门控失败结论不改写。',
      '三seed raw总体差均值为'+f"{d['overall_percent']['mean']:+.6f} ± {d['overall_percent']['sample_std']:.6f}pp（样本标准差），不足+.20pp；平均old为{d['old_percent']['mean']:+.6f}pp，seed1 new为{d['new_percent']['values'][1]:+.6f}pp。",
      '相对无门控raw平均变化'+f"{differences['ungated']['raw']['5']['overall_percent']['mean']:+.6f}pp，不能将微小相对改善包装为主线成功。",'',
      'seed | control raw | ungated raw | gated raw | Δold | Δnew | ΔAIA | 接受轮数/25']
    for s,r in seeds.items():
        v=r['T5_raw_differences_pp_except_S_count'];lines.append(f"{s} | {r['arms']['control']['raw'][-1]['overall_percent']:.4f} | {old[s]['raw'][-1]['overall_percent']:.4f} | {r['arms']['global']['raw'][-1]['overall_percent']:.4f} | {v['old_percent']:+.4f} | {v['new_percent']:+.4f} | {v['AIA_percent']:+.6f} | {r['arms']['global']['training_gate']['accepted_rounds']}/25")
    lines+=['','冻结条件：'+json.dumps(conditions,ensure_ascii=False),'','全部任务配对差均值±样本SD（pp；不挑seed）：']
    for ref in differences:
        for view in ('raw','centered'):
            for t,row in differences[ref][view].items():
                x=row['overall_percent'];lines.append(f"vs {ref} {view} T{t}: {x['mean']:+.6f} ± {x['sample_std']:.6f}; {x['values']}")
    lines+=['','门控接受次数，按T1–T5：']
    for s,items in gates.items():lines.append(s+': '+str([v['accepted'] for v in items.values()]))
    lines+=['','解释：拒绝规则按训练目标正确工作，57/75轮拒绝，仍没有形成跨seed稳定的总体/旧类收益。不能从低接受率推导放宽阈值会更好，也不能用测试值选符号或强度。',
      '下一步有限训练内诊断：复用既有共同T4首轮端点，检验谱修正方向的训练梯度定向能否在不重合的训练探针上保持下降。保持修正尺度、无需新增本地训练、不使用测试图。另立协议；诊断不等于泛化收益。',
      '', '额外成本（逐seed）：'+json.dumps(cost,ensure_ascii=False),
      '总新训练1875000访问，6个T3/T5里程碑重载验收；所有记录进程退出，共同总历时'+f"{result['completion']['elapsed_seconds']/3600:.4f}小时，最终根实际占用{result['completion']['allocated_bytes_at_exit']/1e9:.4f}GB。",
      '全部T1–T5 raw/centered/old/new/AIA/遗忘/W+S/方向混淆、同图配对和成本见THREE_SEED_REPORT.json；每轮训练损失与实际/候选谱记录见各seed资格文件。控制组复用成本不算本次新增，不根据GPU墙钟宣称加速。',
      '模型、teacher、逐图ID/logits和完整日志留服务器。没有显著性或T10结论。']
    (OUT/'THREE_SEED_REPORT_ZH.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'conditions':conditions,'T5':d,'cost':cost},indent=2))

if __name__=='__main__':main()
