"""Full temporal ten-task reporting for a qualified A or B run; no new evaluations."""
import json,os
from pathlib import Path
from summarize_boundary_three import normalize,trajectory
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/boundary_repair_20261027'
def read(p):return json.loads(p.read_text())

def full_trajectory(stages):
    result=trajectory(stages)
    for mode in result:
        result[mode]['mean_new_T2_T10_percent']=result[mode].pop('mean_new_T2_T3_percent')
        result[mode]['stage_forgetting_pp']=[None if t==0 else sum(max(result[mode]['matrix_percent'][j][i] for j in range(i,t+1))-row[i] for i in range(t))/t for t,row in enumerate(result[mode]['matrix_percent'])]
    return result


def mechanism(records):
    if not records:return None
    totals={k:sum(r[k] for r in records) for k in ('steps','images','mask_images','extra_backward_calls','extra_backward_images',
        'diagnostic_forward_images','diagnostic_forward_calls','momentum_help_ssb_hurt','triggers','nonzero_repairs','repair_step_wall_seconds','client_wall_seconds')}
    reasons={};sums={};maxima={}
    for r in records:
        for k,v in r['reasons'].items():reasons[k]=reasons.get(k,0)+v
        for k,v in r['sums'].items():sums[k]=sums.get(k,0)+v
        for k,v in r['maxima'].items():maxima[k]=max(maxima.get(k,0),v)
    active=totals['steps']-reasons.get('empty_mask',0)
    totals.update(reasons=reasons,reason_fractions={k:v/totals['steps'] for k,v in reasons.items()},
      mask_fraction=totals['mask_images']/totals['images'],trigger_fraction=totals['triggers']/totals['steps'],
      momentum_help_ssb_hurt_fraction_active=totals['momentum_help_ssb_hurt']/active,
      sums=sums,maxima=maxima,means_active={k:v/active for k,v in sums.items()},
      numerical_checks_passed=all(r['numerical_checks_passed'] for r in records),
      timing_note='Whole update time, not separately measured repair overhead')
    return totals


def main(arm):
    assert arm in ('A','B');folder=OUT/('A' if arm=='A' else 'B_ten')
    q=read(folder/'QUALIFICATION.json');job=read(folder/'job.json');done=read(folder/'COMPLETE.json')
    assert q['qualified'] and done['qualified'] and job['status']=='completed' and job['exit_code']==0
    assert q['client_rounds']==500 and q['training_visits']==1250000 and q['reload_images']==10000
    assert q['merges']==list(range(10)) and q['history_committed']==list(range(10)) and q['certificates_passed']==50
    assert q['reload_raw_predictions_equal'] and q['reload_centered_predictions_equal'] and len(q['stages'])==10
    lo_folder=ROOT/'reports/ten_task_20261026/lorm_seed0';assert read(lo_folder/'FINAL_CHECK.json')['passed']
    tr=full_trajectory([normalize(s) for s in q['stages']])
    lo=full_trajectory([read(lo_folder/f'task{t}_fixed_rules.json') for t in range(1,11)])
    for rule in ('raw','centered'):
        assert len(tr[rule]['matrix_percent'])==10 and all(len(r)==i+1 for i,r in enumerate(tr[rule]['matrix_percent']))
    keys=('final_overall_percent','final_old_percent','final_new_percent','AIA_percent','mean_new_T2_T10_percent','forgetting_pp')
    comparisons={rule:{k:tr[rule][k]-lo[rule][k] for k in keys} for rule in ('raw','centered')}
    stage_comparison={}
    for rule in ('raw','centered'):
        stage_comparison[rule]=[{'task':t+1,'method_overall':tr[rule]['stage_overall_percent'][t],
          'LoRM_overall':lo[rule]['stage_overall_percent'][t],
          'difference_pp':tr[rule]['stage_overall_percent'][t]-lo[rule]['stage_overall_percent'][t],
          'net_correct':round((tr[rule]['stage_overall_percent'][t]-lo[rule]['stage_overall_percent'][t])*10*(t+1))} for t in range(10)]
    reused=3 if arm=='A' else 1
    probes=[r for r in q['probe_records'] if r['task']>=reused]
    cost={k:q[k] for k in ('training_visits','new_training_visits','stats_visits','elapsed_seconds','peak_gpu_allocated_bytes',
                        'checkpoint_bytes','logical_communication_bytes','reload_images')}
    cost.update(new_training_visits_scope='Since latest imported boundary; for B this is T4-T10 only',
        whole_experiment_new_training_visits=(10-reused)*125000,
        whole_experiment_reused_training_visits=reused*125000,
        actual_new_statistics_images=(10-reused)*30000,
        actual_new_stage_evaluation_images=sum(t*1000 for t in range(reused+1,11)),
        prior_stage_evaluation_images_reused=sum(t*1000 for t in range(1,reused+1)),
        prototype_forward_images=sum(r['samples'] for r in probes if r['kind']=='prototype_fit'),
        selection_forward_backward_images=sum(r['samples'] for r in probes if r['kind']!='prototype_fit'),
        logical_communication_total_bytes=sum(q['logical_communication_bytes'].values()),
        experiment_root_allocated_bytes_at_end=job['allocated_bytes'])
    if arm=='B':
        b3=read(OUT/'B/QUALIFICATION.json');cost.update(additional_T3_reload_images=3000,
          peak_gpu_allocated_bytes_all_B_segments=max(q['peak_gpu_allocated_bytes'],b3['peak_gpu_allocated_bytes']),
          timing_scope='Original B launch through ten-task reload, including T3 qualification and wait before continuation',
          real_preflight=read(OUT/'B/REAL_PREFLIGHT.json'))
    else:cost['timing_scope']='T4-T10 continuation and final reload; historical T1-T3 runtime is not included'
    diag=mechanism(q.get('repair_records',[]))
    report={'status':'COMPLETE_QUALIFIED_SINGLE_SEED_TEN_TASK','arm':arm,'primary_readout':'fixed task-block mean subtraction; raw retained',
      'method':tr,'LoRM':lo,'method_minus_LoRM_pp':comparisons,'stage_comparison':stage_comparison,
      'final_net_correct_vs_LoRM':{rule:round(comparisons[rule]['final_overall_percent']*100) for rule in comparisons},
      'objective_observed':{rule:comparisons[rule]['final_overall_percent']>0 for rule in comparisons},
      'qualification':{k:q[k] for k in ('qualified','client_rounds','training_visits','merges','history_committed','certificates_passed','reload_images','reload_raw_predictions_equal','reload_centered_predictions_equal','maximum_reload_logit_difference')},
      'selection_accepted_layers':[r['accepted_layers'] for r in q['selection_records']],
      'repair_mechanism':diag,'cost':cost,'clock':q['clock'],'source_config_sha256':q['config_sha256'],
      'sources':{'method':str(folder.relative_to(ROOT))+'/QUALIFICATION.json','LoRM':'reports/ten_task_20261026/lorm_seed0/task{1..10}_fixed_rules.json'},
      'limitations':['Single seed and shared development test set; no independent held-out or multi-seed significance.',
       'Complete-system comparison: standard LoRM rank16 alternating A/B AdamW linear-bias mixed precision versus rank32 SSB cosine Head-RegMean FP32; not isolated causal attribution.',
       'Every task uses its original temporal evaluation, not the final T10 model backtested on earlier stages.',
       'Shared GPU timing is not a controlled throughput benchmark; communication is logical tensor accounting, not measured network traffic.']}
    if arm=='B':
        report.update(conditional_extension=read(OUT/'B_ten/GATE.json'),repair_records=q['repair_records'],finite_step_windows=q['finite_step_windows'])
        report['limitations']+=['B horizon extended because T3 exceeded LoRM on the same development set; disclose this selection, not independent confirmation.',
        'Correction guarantees are local first-order approximations with FP32 rounding allowances, not finite-step/old-class/FedAvg guarantees. Original A-selection certificates only cover SSB base proposals.']
    (OUT/f'{arm}_TEN_TASK_REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    label='原联邦聚合效用 + 原SSB' if arm=='A' else '联邦聚合效用 + 历史加权边界修正'
    rows=[f'实验 {arm} 完整十任务报告：{label}','',
      f"主读出：{tr['centered']['final_overall_percent']:.3f}% vs 标准LoRM {lo['centered']['final_overall_percent']:.3f}%，差{comparisons['centered']['final_overall_percent']:+.3f}个百分点，净正确数差{report['final_net_correct_vs_LoRM']['centered']:+d}/10000。",
      f"raw：{tr['raw']['final_overall_percent']:.3f}% vs LoRM {lo['raw']['final_overall_percent']:.3f}%，差{comparisons['raw']['final_overall_percent']:+.3f}个百分点，净正确数差{report['final_net_correct_vs_LoRM']['raw']:+d}/10000。",
      '执行已合格：500客户端轮、125万训练访问、10个头/10次merge/10次history commit、50次头聚合证书。实际最终checkpoint全部10000图重载raw与去均值预测一致；最大logit差 '+str(q['maximum_reload_logit_difference'])+'。不声称logits逐位相同。',
      '固定的块去均值是预先规定的主读出，无真实任务ID。raw一并报告；若结论随读出变化，不能表述为所有读出均优于LoRM。','',
      '实验设置：CIFAR100完整T1–T10，每任务10类，seed0，10客户端全参与，Dirichlet alpha=.05，每任务5轮、本地5epochs、batch16。共享A的候选池64、正式rank32，当前任务CE训练B，原SSB五步Newton–Schulz与原动量，cosine16 head及Head-RegMean alpha=.5；FP32、确定性、math SDPA、TF32关闭。',
      '标准LoRM保持原官方方案：rank16、A/B交替更新、AdamW、linear+bias head、混合精度；相同数据划分、类别顺序和初始化来源。是完整系统比较，不把对LoRM的全部差异归因于选A或边界修正。',
      ('A从原已完成T3的真实checkpoint继承模型/头/history/RNG继续T4–T10；前三阶段指标直接继承当时记录，未重训。' if arm=='A' else 'B从原共享T1开始；T3验收且超过LoRM主指标后按用户授权，从B自己的T3状态继续T4–T10。使用同一原始时钟与参数，保留三任务记录；这是条件扩展，不是独立确认。'),'']
    for rule in ('centered','raw'):
        rows += ['读出 '+rule,'指标 | 本方法 | LoRM | 差（百分点）']
        for k in keys:rows.append(f'{k}: {tr[rule][k]:.4f} | {lo[rule][k]:.4f} | {comparisons[rule][k]:+.4f}')
        rows.append('逐阶段总体对比及净正确数：'+json.dumps(stage_comparison[rule],ensure_ascii=False))
        for name,values in [(arm,tr[rule]),('LoRM',lo[rule])]:
            rows.append(name+'完整下三角准确率矩阵（%）：')
            rows += ['T'+str(i+1)+': '+', '.join(f'{v:.3f}' for v in r) for i,r in enumerate(values['matrix_percent'])]
            for k in ('stage_overall_percent','stage_old_percent','stage_new_percent','stage_forgetting_pp','restricted_matrix_percent',
                      'restricted_correct_stolen_counts','old_to_current_rate','current_to_old_rate','task_group_confusion'):
                rows.append(name+' '+k+': '+json.dumps(values[k]))
        rows.append('')
    rows+=['AIA是十个阶段总体准确率的均值。最终遗忘：对每个旧任务，取学习后到当前的最好准确率减当前准确率，再对9个旧任务平均；逐阶段用同一定义。遗忘低不单独证明绝对旧类准确率高，应联合old accuracy与各a_tt判断。',
           '选择A各任务（T2–T10）接受层数：'+str(report['selection_accepted_layers']),
           '成本与资源（含复用/新增区分）：'+json.dumps(cost,ensure_ascii=False,indent=2)]
    if diag:
        rows+=['修正机制汇总：'+json.dumps(diag,ensure_ascii=False,indent=2),
               '按任务/轮/客户端的完整诊断及固定窗口真实损失保留在JSON。截断残余冲突如实报告。FP32实际位移的数值通过允许显式舍入包络，不宣称实数精确约束；局部一阶修正不保证有限步、旧类或FedAvg后的效果。',
               'repair_step_wall_seconds是整步耗时；额外反向/图像量单列，未独立测量纯修正时间开销。']
    rows+=['','结论范围：只支持本seed0、本固定协议和对应读出的观察，不构成多seed或独立留出集结论。原始/负结果不删改，不因本次测试结果调整超参数或追加实验。',
           '公开内容仅源码、配置、协议、聚合统计及报告；权重、Gram、梯度、原型和逐图输出保留私有。']
    (OUT/f'{arm}_TEN_TASK_REPORT_ZH.txt').write_text('\n'.join(rows)+'\n')
    print(json.dumps({'arm':arm,'comparison':comparisons,'net_correct':report['final_net_correct_vs_LoRM'],'objective':report['objective_observed']},indent=2))

if __name__=='__main__':main(os.environ.get('REPORT_ARM','A'))
