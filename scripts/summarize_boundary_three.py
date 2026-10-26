"""Publishable aggregate-only report for the fully qualified B three-task stage."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/boundary_repair_20261027'
def read(p):return json.loads(p.read_text())

def normalize(stage):
    n=stage['task']+1;c=stage['confusion']
    raw={'per_task':[{'n':1000,'all_seen_accuracy':a,'restricted_accuracy':b,
                     'restricted_correct_but_other_task_wins':round((b-a)*1000)} for a,b in zip(stage['per_task'],stage['restricted'])],
         'old_to_current':stage['old_to_new_error'],'current_to_old':stage['new_to_old_error'],
         'task_group_confusion':[[sum(c[i][j] for i in range(a*10,a*10+10) for j in range(b*10,b*10+10)) for b in range(n)] for a in range(n)]}
    return {'raw':raw,'centered':stage['centered']}

def trajectory(stages):
    result={}
    for mode in ('raw','centered'):
        rows=[[v['all_seen_accuracy'] for v in s[mode]['per_task']] for s in stages];n=len(rows)
        overall=[sum(r)/len(r) for r in rows]
        result[mode]={'matrix_percent':[[100*a for a in r] for r in rows],
         'stage_overall_percent':[100*a for a in overall],
         'stage_old_percent':[100*sum(r[:-1])/t if t else None for t,r in enumerate(rows)],
         'stage_new_percent':[100*r[t] for t,r in enumerate(rows)],
         'final_overall_percent':100*overall[-1],'final_old_percent':100*sum(rows[-1][:-1])/(n-1),
         'final_new_percent':100*rows[-1][-1],'AIA_percent':100*sum(overall)/n,
         'mean_new_T2_T3_percent':100*sum(rows[t][t] for t in range(1,n))/(n-1),
         'forgetting_pp':100*sum(max(rows[j][i] for j in range(i,n))-rows[-1][i] for i in range(n-1))/(n-1),
         'restricted_matrix_percent':[[100*v['restricted_accuracy'] for v in s[mode]['per_task']] for s in stages],
         'restricted_correct_stolen_counts':[[v['restricted_correct_but_other_task_wins'] for v in s[mode]['per_task']] for s in stages],
         'task_group_confusion':[s[mode]['task_group_confusion'] for s in stages],
         'old_to_current_rate':[s[mode]['old_to_current'] for s in stages],
         'current_to_old_rate':[s[mode]['current_to_old'] for s in stages]}
    return result


def main():
    q=read(OUT/'B/QUALIFICATION.json');job=read(OUT/'B/job.json');assert q['qualified'] and job['qualified']
    assert q['client_rounds']==150 and q['training_visits']==375000 and q['reload_images']==3000
    assert q['reload_raw_predictions_equal'] and q['reload_centered_predictions_equal']
    b=trajectory([normalize(s) for s in q['stages']])
    lo=trajectory([read(ROOT/f'reports/ten_task_20261026/lorm_seed0/task{t}_fixed_rules.json') for t in (1,2,3)])
    old=read(ROOT/'reports/aggregation_utility_20261026/PRO_REPORT_DATA.json')['arms']['federated']
    paired=read(OUT/'B/B3_PAIRED_PREVIOUS.json');records=q['repair_records'];assert len(records)==100
    keys=['steps','images','mask_images','extra_backward_calls','extra_backward_images','diagnostic_forward_images',
          'diagnostic_forward_calls','momentum_help_ssb_hurt','triggers','nonzero_repairs','repair_step_wall_seconds','client_wall_seconds']
    mechanism={k:sum(v[k] for v in records) for k in keys};reasons={};sums={};maxima={}
    for r in records:
        for k,v in r['reasons'].items():reasons[k]=reasons.get(k,0)+v
        for k,v in r['sums'].items():sums[k]=sums.get(k,0)+v
        for k,v in r['maxima'].items():maxima[k]=max(maxima.get(k,0),v)
    active=mechanism['steps']-reasons.get('empty_mask',0)
    mechanism.update(reasons=reasons,reason_fraction_all_steps={k:v/mechanism['steps'] for k,v in reasons.items()},
        mask_fraction=mechanism['mask_images']/mechanism['images'],trigger_fraction_all_steps=mechanism['triggers']/mechanism['steps'],
        momentum_help_ssb_hurt_fraction_active=mechanism['momentum_help_ssb_hurt']/active,
        sums=sums,maxima=maxima,means_active={k:v/active for k,v in sums.items()},
        numerical_checks_passed=all(v['numerical_checks_passed'] for v in records),
        timing_note='repair_step_wall_seconds covers whole update, not independently isolated repair overhead')
    compare={mode:{'final_overall_pp':b[mode]['final_overall_percent']-lo[mode]['final_overall_percent'],
                   'net_correct':round((b[mode]['final_overall_percent']-lo[mode]['final_overall_percent'])*30),
                   'AIA_pp':b[mode]['AIA_percent']-lo[mode]['AIA_percent']} for mode in ('raw','centered')}
    decomposition={}
    for mode in ('raw','centered'):
        current=b[mode];restricted=round(current['restricted_matrix_percent'][-1][-1]*10)
        correct=round(current['matrix_percent'][-1][-1]*10)
        decomposition[mode]={'restricted_errors':1000-restricted,
            'restricted_correct_stolen_by_old':current['restricted_correct_stolen_counts'][-1][-1],
            'all_seen_errors':1000-correct}
        assert sum(decomposition[mode][k] for k in ('restricted_errors','restricted_correct_stolen_by_old'))==1000-correct
    cost={k:q[k] for k in ('training_visits','new_training_visits','stats_visits','reload_images','elapsed_seconds','peak_gpu_allocated_bytes','checkpoint_bytes','logical_communication_bytes')}
    cost.update(actual_new_client_rounds=100,actual_new_statistics_images=60000,
                new_stage_evaluation_images=5000,shared_T1_evaluation_images=1000,
                prototype_forward_images=sum(v['samples'] for v in q['probe_records'] if v['kind']=='prototype_fit'),
                selection_forward_backward_images=sum(v['samples'] for v in q['probe_records'] if v['kind']!='prototype_fit'),
                extra_training_backward_images=mechanism['extra_backward_images'],
                diagnostic_forward_images=mechanism['diagnostic_forward_images'],
                logical_communication_total_bytes=sum(q['logical_communication_bytes'].values()),
                allocated_bytes_at_T3_end=job['allocated_bytes'],real_preflight=read(OUT/'B/REAL_PREFLIGHT.json'))
    result={'status':'B_T3_QUALIFIED_AND_GATE_PASSED_B_T10_RUNNING_A_T10_RUNNING',
      'arms':{'B':b,'LoRM':lo,'previous_federated':old},'B_minus_LoRM':compare,'T3_new_class_decomposition':decomposition,
      'paired_B_vs_previous':paired,'repair_mechanism':mechanism,'repair_records':records,
      'finite_step_windows':q['finite_step_windows'],'cost':cost,
      'selection_accepted_layers':[v['accepted_layers'] for v in q['selection_records']],
      'qualification':{k:q[k] for k in ('qualified','client_rounds','training_visits','merges','history_committed','certificates_passed','reload_images','reload_raw_predictions_equal','reload_centered_predictions_equal','maximum_reload_logit_difference')},
      'gate':read(OUT/'B_ten/GATE.json'),
      'sources':{'B':'reports/boundary_repair_20261027/B/QUALIFICATION.json','LoRM':'reports/ten_task_20261026/lorm_seed0/task{1,2,3}_fixed_rules.json','previous':'reports/aggregation_utility_20261026/PRO_REPORT_DATA.json'},
      'limitations':['One seed; shared development evaluation set, no independent held-out confirmation or multi-seed inference.',
       'LoRM is standard rank16 alternating A/B AdamW linear-bias mixed-precision system; B rank32 SSB cosine Head-RegMean FP32. Full-system comparison, not causal attribution of the LoRM gap to repair.',
       'Three-task performance condition determines extension to ten tasks; disclose selection rather than treating extension as independent confirmation.',
       'First-order local correction does not guarantee finite-step CE/cross loss, old-class accuracy, or FedAvg response.',
       'Existing selected-A certificate covers original SSB proposal only; training correction has separate numerical diagnostics.',
       'Raw performance worsens versus previous federated method despite centered improvement; both retained.',
       'Wall time and GPU memory measured on shared hardware; isolated repair overhead not established.']}
    (OUT/'B_THREE_TASK_REPORT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    rows=['实验 B：历史加权跨任务边界修正，完整三任务报告','',
      f"B 已完成150客户端轮、375000次训练访问（复用T1的125000次，新增250000次），15次头聚合证书通过。实际最终checkpoint全部3000图重载后的raw与去均值预测均一致，最大logit差{q['maximum_reload_logit_difference']:.9g}。",
      f"主指标：B 去均值{b['centered']['final_overall_percent']:.3f}%，LoRM {lo['centered']['final_overall_percent']:.3f}%，高{compare['centered']['final_overall_pp']:.3f}个百分点，净多判对{compare['centered']['net_correct']}张。raw：B {b['raw']['final_overall_percent']:.3f}%，LoRM {lo['raw']['final_overall_percent']:.3f}%，高{compare['raw']['final_overall_pp']:.3f}个百分点。",
      '因此满足用户条件，已从B自身T3完整状态在GPU7继续T4-T10；保留原时钟、参数与三任务结果。A/B十任务均尚未完成。本报告只交付已验收的B三任务，不能替代十任务结论。','',
      '配置与方法：CIFAR100前30类、每任务10类，seed0、10客户端全参与、alpha=.05、每任务5轮、本地5epochs、batch16。候选池64、正式rank32、固定共享A训练B、cosine16、Head-RegMean alpha=.5。训练CE仅当前10类，旧头冻结。竞争损失只修正B，head AdamW和B EMA每步各推进一次，修正不写EMA。lambda=1、kappa=.25，全部Q/V共同投影和全局幅度限制。测试不使用真实任务ID。','',
      '比较口径：LoRM使用标准配置且为相同seed/划分/类别顺序、同阶段当时评测，不用T10模型回测前三任务；不新增LoRM训练。对LoRM是完整系统比较，差异不只边界修正，不能将全部优势归因于新模块。','']
    for mode in ('centered','raw'):
        rows+=['读出 '+mode,'指标 | B | 标准LoRM | 旧联邦效用方法']
        for key in ('final_overall_percent','final_old_percent','final_new_percent','AIA_percent','mean_new_T2_T3_percent','forgetting_pp'):
            rows.append(f"{key}: {b[mode][key]:.4f} | {lo[mode][key]:.4f} | {old[mode][key]:.4f}")
        for name,tr in [('B',b),('LoRM',lo),('旧联邦效用',old)]:
            rows.append(name+'完整准确率下三角矩阵（%）：'+json.dumps(tr[mode]['matrix_percent']))
            rows.append(name+'restricted矩阵（%）：'+json.dumps(tr[mode]['restricted_matrix_percent']))
            rows.append(name+'任务内正确但跨任务被抢走数：'+json.dumps(tr[mode]['restricted_correct_stolen_counts']))
        rows.append('B每阶段 overall/old/new（%）：'+json.dumps({k:b[mode][k] for k in ('stage_overall_percent','stage_old_percent','stage_new_percent')}))
        rows.append('B每阶段 old→new / new→old错误率：'+json.dumps([b[mode]['old_to_current_rate'],b[mode]['current_to_old_rate']]))
        rows.append('T3新1000图错误分解：'+json.dumps(decomposition[mode],ensure_ascii=False))
        rows.append('相同样本ID对旧联邦方法的T3配对：'+json.dumps(paired['3'][mode],ensure_ascii=False));rows.append('')
    rows+=['AIA是三个阶段总体准确率的均值。最终遗忘为旧任务i在学习后至当前的最大准确率减当前准确率，再对旧任务平均；遗忘低不单独代表旧类绝对准确率高。',
      '相对旧联邦方法，去均值只改变9张预测：纠正8张、破坏1张、净增加7张；新类纠正5张且无破坏。raw改变30张：纠正4张、破坏23张，净减少19张。不能把净增7张表述为只改变7张，也不能隐去raw退步。','',
      '训练期机制汇总：'+json.dumps(mechanism,ensure_ascii=False,indent=2),
      '逐任务/轮次/客户端完整100条诊断和6个固定窗口记录在JSON。窗口比较使用同一增强结果、固定mask和相同CE更新后的头；观察不额外推进优化器，恢复buffer/RNG。以下为窗口实际损失：']
    for w in q['finite_step_windows']:
        rows.append(json.dumps(w,ensure_ascii=False))
    rows+=['', '选择A阶段每任务接受层数：'+str(result['selection_accepted_layers']),
      '选A证书只针对原SSB基础提案。修正的CE一阶正交、历史幅度及竞争一阶诊断另行记录；截断后可能保留竞争冲突。FP32实际幅度轻微超出0.25仅在显式舍入误差包络内通过，完整数值见JSON，不称精确实数等式。',
      '成本：'+json.dumps(cost,ensure_ascii=False,indent=2),
      '额外反向图像与次数单独记录；repair_step_wall_seconds包含整步计算，不能作为隔离测得的额外修正开销。总运行时间含选A、训练、评测/重载，GPU共享负载使跨实验耗时不构成受控速度比较。','',
      '结论：三任务完整方案超过标准LoRM，优势主要来自旧类保持，B的T3新类准确率仍低于LoRM。修正确实触发并改变更新，去均值相对旧方案净增7张但raw净减19张。新模块收益较小且依赖读出，不能由此宣称普遍机制优势；需等待已授权的完整十任务结果。',
      '本次十任务扩展按同一开发评测集的T3性能条件决定，是明确披露的条件选择，不是独立留出验证。仅seed0，不能把层/客户端/小批次作为独立seed做显著性结论。',
      '数据边界：公开源码/配置/协议/聚合统计；模型权重、历史Gram、梯度、原型与逐图预测留在服务器私有目录。']
    (OUT/'B_THREE_TASK_REPORT_ZH.txt').write_text('\n'.join(rows)+'\n')
    print(json.dumps({'B_minus_LoRM':compare,'mechanism_steps':mechanism['steps'],'triggers':mechanism['triggers'],'reasons':reasons,'new_class':decomposition},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
