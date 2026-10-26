"""Final aggregate synthesis; includes small gains, adverse raw results and numeric limits."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/boundary_repair_20261027'
def read(name):return json.loads((OUT/name).read_text())

a=read('A_TEN_TASK_REPORT.json');b=read('B_TEN_TASK_REPORT.json');paired=read('B_ten/PAIRED_B_VS_A.json')
assert a['qualification']['qualified'] and b['qualification']['qualified']
methods={'A':a['method'],'B':b['method'],'LoRM':b['LoRM']};keys=['final_overall_percent','final_old_percent','final_new_percent','AIA_percent','mean_new_T2_T10_percent','forgetting_pp']
diff={rule:{k:b['method'][rule][k]-a['method'][rule][k] for k in keys} for rule in ('raw','centered')}
for t in range(1,11):
 for rule in ('raw','centered'):
  p=paired[str(t)][rule]['all'];assert p['net_correct']==p['corrected']-p['broken']
  assert p['changed_predictions']==p['corrected']+p['broken']+p['both_wrong_but_changed']
  delta=(b['method'][rule]['stage_overall_percent'][t-1]-a['method'][rule]['stage_overall_percent'][t-1])*10*t
  assert round(delta)==p['net_correct']
records=b['repair_records'];diag=b['repair_mechanism'];windows=b['finite_step_windows']
assert len(records)==450 and len(windows)==27 and diag['steps']==71400
window_rows=[{'task':w['task']+1,'round':w['round']+1,'step':w['step'],'reason':w['repair']['reason'],
 'scale':w['repair'].get('scale'),'ce_corrected_minus_base':w['corrected']['ce']-w['base']['ce'],
 'cross_corrected_minus_base':None if w['base']['cross'] is None else w['corrected']['cross']-w['base']['cross']} for w in windows]
active_windows=[w for w in window_rows if w['reason'] in ('full','capped') and w['scale']>0]
assert len(active_windows)==2
worst={key:[{'task':r['task']+1,'round':r['round']+1,'client_zero_based':r['client'],
 'maximum':r['maxima'][key],'client_round_max_rounding_Q_norm':r['maxima']['rounding_Q_norm']}
 for r in sorted(records,key=lambda r:r['maxima'].get(key,0),reverse=True)[:4]] for key in ('normalized_ce_residual','amplitude_ratio')}
audit={'envelope_checks_passed':diag['numerical_checks_passed'],'strict_FP32_bound_not_claimed':True,
 'maximum_normalized_CE_residual':diag['maxima']['normalized_ce_residual'],
 'maximum_actual_amplitude_ratio':diag['maxima']['amplitude_ratio'],'nominal_cap':.25,
 'maximum_rounding_Q_norm':diag['maxima']['rounding_Q_norm'],
 'worst_client_rounds':worst,'per_step_violation_count_unavailable':True,
 'finite_step_windows':window_rows,'nonzero_repair_windows':2,
 'finite_step_limitation':'Last-round windows coincide with zero B learning rate; most windows have no repair. No retrospective favorable-window selection or new training.'}
per_task=[]
for task in range(1,10):
 rs=[r for r in records if r['task']==task];reasons={}
 for r in rs:
  for k,v in r['reasons'].items():reasons[k]=reasons.get(k,0)+v
 per_task.append({'task':task+1,'client_rounds':len(rs),'steps':sum(r['steps'] for r in rs),
                  'triggers':sum(r['triggers'] for r in rs),'nonzero_repairs':sum(r['nonzero_repairs'] for r in rs),'reasons':reasons})
report={'status':'ALL_AUTHORIZED_EXPERIMENTS_COMPLETE_QUALIFIED_REPORTS_READY','seed':0,
 'methods':methods,'A_minus_LoRM_pp':a['method_minus_LoRM_pp'],'B_minus_LoRM_pp':b['method_minus_LoRM_pp'],
 'B_minus_A_pp':diff,'paired_B_vs_A_all_stages':paired,'repair_mechanism':diag,'repair_by_task':per_task,
 'numerical_and_finite_step_audit':audit,'cost':{'A':a['cost'],'B':b['cost']},
 'qualification':{'A':a['qualification'],'B':b['qualification']},
 'historical_B_T3':{'centered_percent':94.2666666667,'LoRM_centered_percent':90.8,'raw_percent':90.4666666667,'LoRM_raw_percent':88.7333333333,
 'gate':'Strict primary accuracy exceedance; source/clock preserved on GPU7 continuation','report':'B_THREE_TASK_REPORT_ZH.txt'},
 'answers':{'A_ten_above_LoRM_primary':True,'B_ten_above_LoRM_primary':True,'A_and_B_raw_below_LoRM':True,
 'incremental_B_gain_small':True,'multi_seed_or_independent_confirmation':False},
 'public_scope':'Code/config/protocol/aggregate metrics only; no private model/Gram/gradient/prototype/per-image artifacts.',
 'no_further_experiments':'Authorized A10 and conditional B10 complete; monitoring can end after public delivery verification.'}
(OUT/'FINAL_COMPARISON.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
rows=['FCL-Muon 实验 A/B 最终汇总：完整十任务交付','',
 '两条授权实验均已完成并通过实际checkpoint全10000图重载验收。A是联邦聚合效用选空间加原SSB；B在同一主线上增加历史加权跨任务边界修正。标准LoRM采用已有seed0完整十任务结果，没有重新训练或额外调参。','',
 '最终总体准确率（%） | 标准LoRM | A | B',
 '预定去均值主读出 | 84.150 | 85.890 | 86.160',
 'raw | 80.090 | 79.130 | 79.180',
 'A/B在预定主读出上分别高于LoRM 1.740/2.010个百分点；raw分别低0.960/0.910个百分点。不能声称所有读出都更优。',
 '相对A，B最终去均值增加0.270个百分点（27/10000张），raw增加0.050个百分点（5/10000张）。新增修正的净收益较小，不能将B相对LoRM的全部收益归因于这个模块。','',
 '完整逐任务表现与保持/学习的区分']
for rule in ('centered','raw'):
 rows+=['读出 '+rule,'指标 | LoRM | A | B | B-A（百分点）']
 for k in keys:rows.append(f"{k}: {methods['LoRM'][rule][k]:.4f} | {methods['A'][rule][k]:.4f} | {methods['B'][rule][k]:.4f} | {diff[rule][k]:+.4f}")
 rows+=['阶段 | LoRM总体 | A总体 | B总体 | B相对A配对净正确数']
 for t in range(10):rows.append(f"T{t+1}: {methods['LoRM'][rule]['stage_overall_percent'][t]:.3f} | {methods['A'][rule]['stage_overall_percent'][t]:.3f} | {methods['B'][rule]['stage_overall_percent'][t]:.3f} | {paired[str(t+1)][rule]['all']['net_correct']:+d}")
 rows.append('T10同样本ID配对：'+json.dumps(paired['10'][rule],ensure_ascii=False))
 rows.append('')
rows += ['B去均值T10新类93.4%，LoRM81.4%，但T2–T10新类初学准确率的均值为90.7%，仍低于LoRM91.1%。T10单阶段优势不能替代整个序列的新类学习评价。B去均值旧类85.356%，LoRM84.456%；最终遗忘5.967pp vs8.456pp。',
 'B相对A的去均值AIA仅增加约0.0434pp；raw AIA反而下降约0.3301pp。报告完整轨迹，不仅报告有利终点。AIA按全部十阶段等权平均；遗忘按旧任务从学习后至当前的最高准确率减当前准确率再平均。',
 'B相对A最终去均值预测改变101张：纠正52、破坏25、两者都错但类别改变24，净增27；其中旧类净增25、新类净增2。raw改变196张：纠正66、破坏61、两者都错但类别改变69，净增5。净正确数差不等于改变预测数。',
 '新类跨任务错误：T3时B去均值新类任务内错误30/1000、任务内正确被旧类抢走59、最终错误89；相对A新类纠正5张。T10详细分解如下（所有样本均按相同ID比较）：']
for name in ('A','B','LoRM'):
 for rule in ('centered','raw'):
  m=methods[name][rule];restricted=round(m['restricted_matrix_percent'][-1][-1]*10);correct=round(m['matrix_percent'][-1][-1]*10)
  stolen=m['restricted_correct_stolen_counts'][-1][-1]
  assert 1000-restricted+stolen==1000-correct
  rows.append(f'{name} {rule}: 任务内错误{1000-restricted}，任务内正确被旧类抢走{stolen}，最终错误{1000-correct}。')
rows += ['', '修正是否真正工作及其边界',
 f"T2–T10共{diag['steps']}步，mask覆盖当前训练图像的{100*diag['mask_fraction']:.3f}%。触发{diag['triggers']}次（{100*diag['trigger_fraction']:.3f}%），真正产生非零位移{diag['nonzero_repairs']}次；完整修正3311次，幅度截断6502次，无相对退化跳过。16次触发未部署非零修正；现有汇总未逐步拆分零幅度上限与舍入的原因。",
 '动量方向有利而SSB后不利只观测84次（有效竞争梯度步骤的约0.118%）；此频率较低，不能把所有触发都解释成SSB把好方向翻坏。模块触发依据还包含实际head更新的贡献。',
 f"名义全局幅度上限0.25；实际FP32最大幅度比{audit['maximum_actual_amplitude_ratio']:.8f}，最大归一化CE正交残差{audit['maximum_normalized_CE_residual']:.8f}（约4.35%）。所有预设舍入包络检查通过，但不等于严格的实际位移上限或精确正交。",
 '最坏相对残差/幅度比记录均在第5轮客户端，符合学习率趋近零时相对舍入影响放大的情形；每步分母未保存，无法逐事件证明原因或统计精确越界次数，不作更强数值保证。全程绝对Q度量舍入差最大约2.00e-8；这与相对偏差不是同一指标。',
 '截断后最大剩余正竞争一阶变化约0.009925，不能宣称每次冲突都完全抵消。原选A证书只针对原SSB基础提案，不认证新修正。',
 '27个预定有限步窗口仅2个部署非零修正，其实际CE与cross相对基础步均略降；其余窗口没有修正或零上限。末轮窗口取本地最后一步，B学习率为零，不能充分验证稳定阶段B修正的有限步效果。没有事后补选窗口或额外训练。',
 '理论闭式解与合成测试验证了实数/高精度全局投影、竞争一阶抵消及全局缩放；旧头梯度路径、CE-only head/EMA单步和状态恢复已实测。未验证多seed稳定增益、独立留出泛化或每个客户端FedAvg后的损失保证。','',
 '成本与执行',
 f"A本次新增训练875000次，复用T1–T3的375000次；B全实验新增1125000次，复用T1的125000次。各自总协议均1250000次。A本段历时{a['cost']['elapsed_seconds']/3600:.3f}小时；B从首次启动到T10重载历时{b['cost']['elapsed_seconds']/3600:.3f}小时，含三任务验收和续跑等待。不同复用范围/共享GPU负载，不作为受控速度对比。",
 f"B新增{diag['extra_backward_calls']}次竞争反向、覆盖{diag['extra_backward_images']}图；固定诊断额外{diag['diagnostic_forward_calls']}次前向、{diag['diagnostic_forward_images']}图。另有真实preflight四次前向/四次反向各16图。修正步骤总耗时含原步骤，不能当成单独测量的额外修正时间。",
 'B额外逻辑Q广播8847360字节；全序列逻辑通信A11838573600、B11847420960字节。逻辑通信不是实际网络流量。',
 f"A峰值GPU分配约{a['cost']['peak_gpu_allocated_bytes']/1e9:.3f}GB，B约{b['cost']['peak_gpu_allocated_bytes_all_B_segments']/1e9:.3f}GB；任务根目录最终实际分配A约{a['cost']['experiment_root_allocated_bytes_at_end']/1e9:.3f}GB、B约{b['cost']['experiment_root_allocated_bytes_at_end']/1e9:.3f}GB，低于预算。",
 '两者各500客户端轮、50部署证书，10头/10次merge/history，原clock未重置。最终实际checkpoint完整10000图raw与去均值预测重载一致；A最大logit差0.0005228519439697266，B0.0003650188446044922。这里只表示各自保存/重载预测可复现，不表示A和B预测相同。','',
 '结论与交付范围',
 '上一版本A十任务以及新版本B十任务，都达到预定去均值主指标高于标准LoRM的要求；raw均未超过LoRM。边界修正有实际触发和可核对的预测改变，十任务增益有限，AIA和raw轨迹不一致，数值与有限步证据有明确边界。',
 'B先根据同一开发评测集T3超过LoRM的条件才扩展T10；这不是独立验证集确认。所有结果仅seed0，不用层/客户端/小批次冒充独立seed，不作显著性结论。',
 '交付文件：A_TEN_TASK_REPORT_ZH.txt/JSON，B_THREE_TASK_REPORT_ZH.txt/JSON，B_TEN_TASK_REPORT_ZH.txt/JSON，本FINAL_COMPARISON_ZH.txt/JSON，完整配置/协议/来源哈希/资格证据及配对聚合统计。各单独报告和JSON包含完整raw/centered下三角矩阵、每阶段old/new、restricted及竞争错误，源码脚本可重建报告。',
 '仅发布源码/配置/协议/聚合结果。模型权重、Gram、梯度、原型与逐图数据保留在服务器，不公开。授权范围已完成，不再自动启动后续实验；公开访问核验后关闭本会话监测。']
(OUT/'FINAL_COMPARISON_ZH.txt').write_text('\n'.join(rows)+'\n')
(OUT/'B_ten/NUMERICAL_AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'B_minus_A':diff,'final_pair_centered':paired['10']['centered']['all'],'nonzero_windows':len(active_windows)},indent=2))
