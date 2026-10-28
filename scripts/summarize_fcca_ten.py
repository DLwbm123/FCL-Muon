"""Rebuild the aggregate-only ten-task FCCA report without model/private data."""
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from summarize_boundary_three import normalize, trajectory

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/fcca_ten_20261028'
def read(path):
    return json.loads(path.read_text())
def dump(value):
    return json.dumps(value, ensure_ascii=False)
def close(a, b, tolerance=1e-9):
    assert abs(a-b) <= tolerance, (a,b)

def main():
    q=read(OUT/'QUALIFICATION.json');job=read(OUT/'job.json');complete=read(OUT/'COMPLETE.json')
    q3=read(ROOT/'reports/fcca_three_20261028/QUALIFICATION.json')
    j3=read(ROOT/'reports/fcca_three_20261028/job.json')
    pair=read(OUT/'PAIRED_C_VS_A.json');cfg=read(ROOT/'configs/fcca_ten_seed0.json')['fcca']
    assert q['qualified'] and job['qualified'] and complete['qualified'] and job['exit_code']==0
    assert job['status']=='completed' and q['primary_readout']=='raw'
    assert q['client_rounds']==500 and q['training_visits']==1250000 and q['stats_visits']==300000
    assert q['continuation_new_client_rounds']==350 and q['new_training_visits']==q['continuation_new_training_visits']==875000
    assert q['full_C_new_training_visits']==1125000 and q['certificates_passed']==50
    assert q['merges']==q['history_committed']==list(range(10))
    assert q['reload_images']==10000 and q['reload_raw_predictions_equal'] and q['reload_centered_predictions_equal']
    assert q['clock']==q3['clock']=={k:job[k] for k in ('started','deadline')}
    assert job['started']==1793160824.1203914 and job['ended']<job['deadline']
    assert job['ended']-job['started']<86400 and job['allocated_bytes']<job['write_stop_bytes']<50000000000
    assert q['stages'][:3]==q3['stages'] and q['fcca_records'][:10]==q3['fcca_records']
    assert q['import_proof']['reused_client_rounds']==150 and q['import_proof']['reused_training_visits']==375000
    c=trajectory([normalize(s) for s in q['stages']])
    for m in c.values():m['mean_new_T2_T10_percent']=m.pop('mean_new_T2_T3_percent')
    methods=read(ROOT/'reports/boundary_repair_20261027/FINAL_COMPARISON.json')['methods']
    methods['C']=c;decomposition={};identity_count=0
    for name,method in methods.items():
        decomposition[name]={}
        for mode,m in method.items():
            assert len(m['matrix_percent'])==10
            stages=[]
            for t,row in enumerate(m['matrix_percent']):
                assert len(row)==t+1
                counts=[]
                for i,acc in enumerate(row):
                    w=1000-round(m['restricted_matrix_percent'][t][i]*10)
                    s=m['restricted_correct_stolen_counts'][t][i];error=1000-round(acc*10)
                    assert w+s==error and min(w,s,error)>=0;identity_count+=1
                    counts.append({'W':w,'S':s,'errors':error})
                conf=m['task_group_confusion'][t];assert all(sum(r)==1000 for r in conf)
                old_new=sum(conf[i][t] for i in range(t));new_old=sum(conf[t][:t])
                if t:
                    close(old_new/(t*1000),m['old_to_current_rate'][t])
                    close(new_old/1000,m['current_to_old_rate'][t])
                stages.append({'per_task':counts,'W':sum(r['W'] for r in counts),'S':sum(r['S'] for r in counts),
                               'errors':sum(r['errors'] for r in counts),'old_to_new_count':old_new,'new_to_old_count':new_old})
            decomposition[name][mode]=stages
    for t in range(1,11):
        for mode in ('raw','centered'):
            for group,r in pair[str(t)][mode].items():
                assert r['net_correct']==r['corrected']-r['broken']
                assert r['changed_predictions']==r['corrected']+r['broken']+r['both_wrong_but_changed']
            for field in pair[str(t)][mode]['all']:
                assert pair[str(t)][mode]['all'][field]==sum(pair[str(t)][mode][g][field] for g in ('old','new'))
            delta=c[mode]['stage_overall_percent'][t-1]-methods['A'][mode]['stage_overall_percent'][t-1]
            assert pair[str(t)][mode]['all']['net_correct']==round(delta*t*10)
    records=q['fcca_records'];assert len(records)==45
    assert {(r['task'],r['round']) for r in records}=={(t,r) for t in range(2,11) for r in range(1,6)}
    round_summary=[];reasons=Counter()
    for r in records:
        assert r['fixed_actual_HeadRegMean'] and r['check_used_once'] and not r['old_class_guarantee']
        assert r['solver']['success'] and r['solver']['feasible']
        p=r['sample_weights'];alpha=r['proposed_weights'];close(sum(alpha),1,cfg['weight_tolerance'])
        assert all(.5*x-cfg['weight_tolerance']<=y<=1.5*x+cfg['weight_tolerance'] for x,y in zip(p,alpha))
        fit=r['baseline_fit'];v=fit['global_raw_derivative'];matrix=fit['cross_client_derivative_matrix']
        for k,value in enumerate(v):close(value,sum(w*row[k] for w,row in zip(fit['fit_client_weights'],matrix)),1e-12)
        close(sum(value*(a-b) for value,a,b in zip(v,alpha,p)),r['solver']['predicted_fit_raw_ce_change'],1e-12)
        why=[];changes={}
        for split in ('fit','check'):
            base=r['baseline_'+split];cand=r['candidate_'+split]
            gain=cfg['minimum_relative_raw_gain']*max(base['raw_ce'],1e-12)
            within=cfg['within_ce_absolute_tolerance']+cfg['within_ce_relative_tolerance']*base['within_ce']
            if cand['raw_ce']>base['raw_ce']-gain:why.append(split+'_raw_gain')
            if cand['within_ce']>base['within_ce']+within:why.append(split+'_within_ce')
            changes[split]={'raw_ce':cand['raw_ce']-base['raw_ce'],'within_ce':cand['within_ce']-base['within_ce'],
                            'minimum_raw_decrease':gain,'within_ce_allowed_increase':within}
        if r['history_candidate_actual_FP32']>r['history_limit']:why.append('actual_FP32_history_guard')
        assert why==r['reasons'] and r['accepted']==(not why)
        assert r['deployed_weights']==(alpha if r['accepted'] else p)
        reasons.update(why)
        round_summary.append({'task':r['task'],'round':r['round'],'accepted':r['accepted'],'reasons':why,
                              'predicted_fit_raw_ce_change':r['solver']['predicted_fit_raw_ce_change'],
                              'actual_changes':changes,'history_actual_minus_limit':r['history_candidate_actual_FP32']-r['history_limit'],
                              'deployed_weight_L1_from_sample':sum(abs(x-y) for x,y in zip(r['deployed_weights'],p))})
    accepted=sum(r['accepted'] for r in records);assert accepted==q['accepted_aggregations']==19
    per_task=[{'task':t,'candidates':5,'accepted':sum(r['accepted'] for r in records if r['task']==t),
               'rejection_reason_counts_nonexclusive':dict(Counter(reason for r in records if r['task']==t for reason in r['reasons']))}
              for t in range(2,11)]
    finite={'candidate_rounds':45,'accepted':accepted,'rejected':45-accepted,
            'fit_raw_CE_decreased':sum(r['actual_changes']['fit']['raw_ce']<0 for r in round_summary),
            'check_raw_CE_decreased':sum(r['actual_changes']['check']['raw_ce']<0 for r in round_summary),
            'fit_and_check_raw_CE_decreased':sum(all(r['actual_changes'][s]['raw_ce']<0 for s in ('fit','check')) for r in round_summary),
            'reasons_nonexclusive':dict(reasons),'per_task':per_task}
    margins=[r['history_actual_minus_limit'] for r in round_summary if 'actual_FP32_history_guard' in r['reasons']]
    finite['rejected_actual_history_excess_range']=[min(margins),max(margins)]
    keys=['final_overall_percent','final_old_percent','final_new_percent','AIA_percent','mean_new_T2_T10_percent','forgetting_pp']
    differences={name:{mode:{k:c[mode][k]-m[mode][k] for k in keys} for mode in ('raw','centered')}
                 for name,m in methods.items() if name!='C'}
    a=methods['A'];gate={'T10_raw_strictly_above_A':c['raw']['final_overall_percent']>a['raw']['final_overall_percent']+1e-10,
                         'raw_AIA_not_below_A':c['raw']['AIA_percent']>=a['raw']['AIA_percent']-1e-10,
                         'T10_raw_old_not_below_A':c['raw']['final_old_percent']>=a['raw']['final_old_percent']-1e-10,
                         'T10_centered_not_below_A':c['centered']['final_overall_percent']>=a['centered']['final_overall_percent']-1e-10,
                         'some_candidate_accepted':accepted>0,'numerical_lifecycle_qualified':q['qualified']}
    comm=q['logical_communication_bytes'];comm3=q3['logical_communication_bytes']
    segment_comm={k:v-comm3.get(k,0) for k,v in comm.items()};assert all(v>=0 for v in segment_comm.values())
    counts=['extra_forward_images','extra_backward_images','extra_forward_calls','extra_backward_calls']
    for k in counts:assert q[k]==sum(r[k] for r in records)
    cost={'full_protocol_client_rounds':500,'full_protocol_training_visits':1250000,'shared_T1_training_visits':125000,
          'full_C_new_client_rounds':450,'full_C_new_training_visits':1125000,
          'T4_T10_new_client_rounds':350,'T4_T10_new_training_visits':875000,
          'full_protocol_statistics_images':300000,'full_C_new_statistics_images':270000,'T4_T10_new_statistics_images':210000,
          'full_protocol_stage_evaluation_images':55000,'T4_T10_new_stage_evaluation_images':49000,
          'T3_reload_images':3000,'T10_reload_images':10000,
          'fcca_full_C':{k:q[k] for k in counts},'fcca_T4_T10':{k:q[k]-q3[k] for k in counts},
          'real_preflight':read(ROOT/'reports/fcca_three_20261028/REAL_PREFLIGHT.json'),
          'logical_full_protocol_bytes':comm,'logical_T4_T10_bytes':segment_comm,
          'logical_full_protocol_total_bytes':sum(comm.values()),'logical_T4_T10_total_bytes':sum(segment_comm.values()),
          'logical_FCCA_full_C_bytes':sum(v for k,v in comm.items() if k.startswith('fcca_')),
          'logical_FCCA_T4_T10_bytes':sum(v for k,v in segment_comm.items() if k.startswith('fcca_')),
          'base_logical_full_protocol_bytes':sum(v for k,v in comm.items() if not k.startswith('fcca_')),
          'fcca_fuse_seconds_full_C':sum(r['seconds'] for r in records),'fcca_fuse_seconds_T4_T10':sum(r['seconds'] for r in records if r['task']>3),
          'total_original_clock_seconds':job['ended']-job['started'],'T4_T10_elapsed_seconds':job['ended']-job['continuation_started'],
          'T3_elapsed_seconds':j3['ended']-j3['started'],'between_segments_wait_seconds':job['continuation_started']-j3['ended'],
          'peak_gpu_allocated_bytes_all_C_segments':q['peak_gpu_allocated_bytes_all_C_segments'],
          'final_root_allocated_bytes':job['allocated_bytes'],'continuation_initial_root_allocated_bytes':job['baseline_allocated_bytes'],
          'T10_checkpoint_bytes':q['checkpoint_bytes'],'selector_probe_records':q['probe_records']}
    close(cost['total_original_clock_seconds'],cost['T3_elapsed_seconds']+cost['between_segments_wait_seconds']+cost['T4_T10_elapsed_seconds'],1e-5)
    for period,ts in [('full_C',range(1,10)),('T4_T10',range(3,10))]:
        probes=[r for r in q['probe_records'] if r['task'] in ts]
        cost['selector_'+period]={'prototype_forward_images':sum(r['samples'] for r in probes if r['kind']=='prototype_fit'),
                                 'selection_forward_backward_images':sum(r['samples'] for r in probes if r['kind']!='prototype_fit')}
    result={'status':'COMPLETE_QUALIFIED_SMALL_DESCRIPTIVE_GAIN','seed':0,'methods':methods,'C_minus_reference_pp':differences,
            'error_decomposition':decomposition,'paired_C_vs_A':pair,'round_summary':round_summary,'fcca_records':records,
            'finite_step_summary':finite,'scientific_conditions':gate,'descriptive_scientific_progress':all(gate.values()),'cost':cost,
            'qualification':{k:q[k] for k in ('qualified','client_rounds','training_visits','merges','history_committed','certificates_passed',
                                             'reload_images','reload_raw_predictions_equal','reload_centered_predictions_equal','maximum_reload_logit_difference')},
            'clock':q['clock'],'checks':{'W_plus_S_task_identities':identity_count,'paired_stage_readouts':20,'candidate_acceptance_recomputed':45,
                                       'cross_client_derivative_and_prediction_recomputed':45,'T1_T3_stages_and_candidates_unchanged':True},
            'limitations':['Single seed; small descriptive differences are not statistical confirmation.',
                           'T10 explicitly requested after observed T3 result, using the same development evaluation set.',
                           'Fit/check probes are disjoint training subsets, not independent representation-held-out validation.',
                           'Archive response guard is not an old-class classification guarantee.',
                           'LoRM comparison changes rank, optimizer, head and precision; C versus A isolates the added module.',
                           'Fuse timing includes original head aggregation; CPU solver time was not separately recorded.',
                           'Logical communication is not measured network traffic; shared hardware and waiting prevent controlled speed claims.'],
            'warning':'SciPy clipped an intermediate out-of-box iterate; all final candidates had separate feasibility checks; no training failure.',
            'sources':{'C':'QUALIFICATION.json','pair':'PAIRED_C_VS_A.json','references':'../boundary_repair_20261027/FINAL_COMPARISON.json',
                       'T3':'../fcca_three_20261028/QUALIFICATION.json','protocol':'../../research/fcca/TEN_TASK_EXTENSION_ZH.txt',
                       'implementation_commit':read(OUT/'SOURCE.json')['source_commit']}}
    assert identity_count==440 and all(gate.values())
    (OUT/'FINAL_REPORT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['FCCA-C 完整十任务报告（2026-10-28，seed0）','',
           '结论：十任务训练、数值与生命周期验收完成，满足预先冻结的描述性进展条件，但新增收益很小，不能视为稳定或显著改进。',
           'T10 C raw79.21%、centered85.92%；A79.13%/85.89%，B79.18%/86.16%，LoRM80.09%/84.15%。C相对A分别净增加8/3张正确；raw仍比LoRM低0.88个百分点，centered比LoRM高1.77个百分点。',
           'C相对B raw仅高0.03个百分点，centered低0.24个百分点，不能声称全面优于B。C相对A raw AIA增0.031274个百分点、centered AIA增0.030742个百分点。',
           'T10 raw旧类净增加5张、新类净增加3张；centered净增加3张全在旧类。新增机制不是没有效果，但总体收益有限。','',
           '协议与授权',
           'CIFAR100，10任务×10类、seed0、Dirichlet alpha=.05、10客户端全参与、每任务5轮、本地5epochs、batch16、rank32、cosine16、FP32。共享T1后C独立训练T2–T10；保留A联邦选空间、SSB、Head-RegMean，没有B本地边界修正。',
           'T3未超过A的原负结果保持不变。用户查看T3后明确要求扩十任务，所以从C自身完整T3状态/RNG继续，仅改tasks/output，全部候选/损失/历史阈值不变。原时钟不重置。这是同一开发序列上的续跑，不是独立确认。',
           'A/B/LoRM使用已交付同seed、相同数据划分和当时阶段模型的历史记录，不重新训练或调优比较方法，不以T10模型回测历史。LoRM为rank16、交替A/B AdamW、线性带bias头及混合精度系统；其差距是系统差距，不能全部归因于FCCA。','']
    for mode in ('raw','centered'):
        lines += [mode+'读出（准确率%，遗忘为百分点）','方法 | T10 overall | old | new | AIA | T2–T10初学均值 | forgetting']
        for name in ('LoRM','A','B','C'):
            lines.append(name+' | '+' | '.join(f'{methods[name][mode][k]:.6f}' for k in keys))
        lines.append('各阶段总体准确率：阶段 | LoRM | A | B | C | C-A净正确数')
        for t in range(10):
            lines.append(f'T{t+1} | '+' | '.join(f"{methods[name][mode]['stage_overall_percent'][t]:.6f}" for name in ('LoRM','A','B','C'))+f" | {pair[str(t+1)][mode]['all']['net_correct']:+d}")
        for name in ('LoRM','A','B','C'):
            m=methods[name][mode]
            lines += [name+'完整下三角矩阵（行是评测阶段，列是所学任务，%）：'+dump(m['matrix_percent']),
                      name+'任务内restricted矩阵（%）：'+dump(m['restricted_matrix_percent']),
                      name+'逐阶段old/new（%）：'+dump({k:m[k] for k in ('stage_old_percent','stage_new_percent')}),
                      name+'逐阶段W/S/总错误与旧→新、新→旧计数：'+dump(decomposition[name][mode]),
                      name+'任务组混淆矩阵（每阶段，行真值/列预测，每行1000图）：'+dump(m['task_group_confusion'])]
        lines += ['C相对A/B/LoRM差值（百分点）：'+dump({name:v[mode] for name,v in differences.items()}),
                  'C对A全部阶段同ID配对（all/old/new）：'+dump({t:v[mode] for t,v in pair.items()}),'']
    lines += ['AIA为10个阶段总体准确率等权均值。old为此前任务，new为当前任务。遗忘为各旧任务学习后到T10的最高准确率减T10准确率，再对9个旧任务平均。C相对A遗忘略增（raw约0.0333pp、centered约0.0222pp），与最终旧类绝对准确率略高并不矛盾；不能只挑遗忘或终点之一。',
              'W是任务内restricted错误，S是任务内正确但被其他任务类别抢走；总错误=W+S。方向混淆旧→新/新→旧不是S的同义词。已验证440个任务级分解恒等式及20组阶段/读出配对恒等式。',
              'T10 raw预测改变55张：纠正22、破坏14、双方都错但类别改变19，净增8；centered改变28张：纠正9、破坏6、双方都错但类别改变13，净增3。净收益不是改变预测的总数。','',
              '候选聚合、有限步损失与历史预算',
              '45轮均得到通过最终求解器约束检查的唯一候选，19接受、26回退原FedAvg。每轮拟合固定真实Head-RegMean头和FedAvg锚点，计算每个客户端梯度对所有客户端方向的作用。跨客户端加权导数及候选一阶预测均重新核对。',
              'fit/check各自真实前向检验raw CE和within CE，部署FP32位移另验历史预算；无二次候选、重搜或阈值放宽。仅fit下降不是接受条件，raw下降还必须达到预定相对幅度。',
              '有限步汇总：'+dump(finite),
              f'16轮实际FP32历史预算拒绝的超限绝对量范围为{min(margins):.6g}至{max(margins):.6g}，均按冻结阈值拒绝，没有因数值很小而放宽。拒绝原因有重叠，不能把各原因次数相加当拒绝总数；也不能据这些候选未部署后的未知结果推断放宽阈值会提高性能。',
              '每轮完整记录如下；CE差为候选减基准，负值表示下降。客户端权重按clients顺序。实际历史代价只约束归档输入局部线性Q/V响应，不能解释为旧类分类保证。']
    for r,s in zip(records,round_summary):
        lines += [f"T{r['task']} R{r['round']}："+dump(s),
                  'clients / sample / proposed / deployed：'+dump({k:r[k] for k in ('clients','sample_weights','proposed_weights','deployed_weights')}),
                  'fit/check基准与候选：'+dump({k:{f:r[k][f] for f in ('raw_ce','within_ce','probe_images','represented_training_images','client_probe_counts')}
                                                 for k in ('baseline_fit','candidate_fit','baseline_check','candidate_check')}),
                  '历史baseline / candidate_FP32 / limit：'+dump([r['history_baseline'],r['history_candidate_actual_FP32'],r['history_limit']])]
    lines += ['', '冻结描述性条件：'+dump(gate),
              '全部通过只表明本次seed0达到了预定描述性条件；不等于有统计显著性、可重复优势或优于LoRM主指标。原T3科学条件未通过仍成立。','',
              '计算、通信与资源',
              '完整协议500客户端轮/1250000训练访问；共享T1为50轮/125000，整个C新增450轮/1125000；本次T4–T10新增350轮/875000，继承T1–T3共150轮/375000。',
              '统计图像完整300000、C新增270000、本段210000；每阶段评测完整55000图、本段49000图。额外重载T3为3000、T10为10000。',
              'FCCA全C额外前向/反向：'+dump(cost['fcca_full_C'])+'；其中T4–T10：'+dump(cost['fcca_T4_T10']),
              '启动preflight另有前向2次/32图、反向2次/32图，没有优化器更新。原选空间成本另外记录：'+dump({k:v for k,v in cost.items() if k.startswith('selector_') and k!='selector_probe_records'}),
              f"完整逻辑通信{cost['logical_full_protocol_total_bytes']}字节，其中FCCA额外{cost['logical_FCCA_full_C_bytes']}；去掉FCCA项为{cost['base_logical_full_protocol_bytes']}字节。本段新增逻辑通信{cost['logical_T4_T10_total_bytes']}，其中FCCA {cost['logical_FCCA_T4_T10_bytes']}。完整数含复用T1账本，不是本次实测网络传输。",
              f"FCCA额外逻辑通信相当于A同口径基础通信增加{100*cost['logical_FCCA_full_C_bytes']/cost['base_logical_full_protocol_bytes']:.3f}%；当前小幅精度增益需结合这项成本判断，不宣称轻量通信优势。",
              '完整/本段逻辑通信分项：'+dump({'full':comm,'T4_T10':segment_comm}),
              f"FCCA fuse累计{cost['fcca_fuse_seconds_full_C']:.3f}秒，本段{cost['fcca_fuse_seconds_T4_T10']:.3f}秒；此区间含原头聚合、额外探针与CPU计算，不能称为纯新增开销。CPU求解未单独计时。",
              f"原时钟总历时{cost['total_original_clock_seconds']:.3f}秒，其中T3段{cost['T3_elapsed_seconds']:.3f}、段间等待{cost['between_segments_wait_seconds']:.3f}、T4–T10段{cost['T4_T10_elapsed_seconds']:.3f}秒。共享GPU负载、复用范围和等待使其不能用于受控速度比较。",
              f"全C GPU峰值分配{cost['peak_gpu_allocated_bytes_all_C_segments']}字节；根最终实际分配{job['allocated_bytes']}字节，续跑前{job['baseline_allocated_bytes']}；T10 checkpoint {q['checkpoint_bytes']}字节。最终占用不是峰值存储测量，运行监督始终采用6GB写停止/8GB上限。",
              '开始、续跑、结束（Asia/Shanghai）：'+dump({k:datetime.fromtimestamp(job[k],ZoneInfo('Asia/Shanghai')).isoformat() for k in ('started','continuation_started','ended')}),
              '原clock：'+dump(q['clock']),
              f"入口完成时检查500唯一客户端轮、10merge/history、50头证书和45唯一聚合；T1–T3原结果/候选记录完全保留。实际10000图checkpoint重载raw/centered预测一致，最大logit差{q['maximum_reload_logit_difference']:.12g}，不声称逐位logit一致。",'',
              '异常、限制与结论',
              '训练退出码0、qualified与COMPLETE均成立。日志保留SLSQP中间点越box自动clip警告；最终约束和有限步部署验收独立执行，未据警告放宽阈值。所有拒绝保留，没有成本性失败重跑。',
              'FCCA补足了统一部署头、跨客户端作用和有限步候选检查，工程/数值路径通过；然而最终收益仅8/10000张raw和3/10000张centered，raw仍低LoRM，centered低B。当前结果支持小幅、单seed的描述性改善，不支持强机制优势或替代旧类监督的保证。',
              'fit/check来自参与正式训练的数据，虽两者互斥但不是独立泛化验证。历史Gram响应约束不是旧标签分类风险。只测seed0，且本轮在已知T3结果后授权扩展；不把客户端、层、轮次或样本数冒充独立seed。',
              '重建命令：在仓库根执行 python3 scripts/summarize_fcca_ten.py，读取公开JSON，标准库即可。服务器同ID配对脚本scripts/pair_boundary_outputs.py使用TASK_COUNT=10、A既有逐阶段输出与C runs/ten，只导出聚合计数。',
              '源码版本/协议/配置见SOURCE.json与research/fcca/TEN_TASK_EXTENSION_ZH.txt。公开源码、配置、协议、聚合损失/方向内积/权重/计数与资格成本；checkpoint、逐图logits/ID、probe索引、原始梯度、Gram和特征仍私有保存。',
              '本轮授权十任务及交付已完成；不根据这些测试结果继续搜参、换候选或自动启动其他实验。']
    (OUT/'FINAL_REPORT_ZH.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'conditions':gate,'C_minus_A':differences['A'],'finite_step':finite,
                      'cost':{k:v for k,v in cost.items() if not isinstance(v,(list,dict))}},indent=2))

if __name__=='__main__':main()
