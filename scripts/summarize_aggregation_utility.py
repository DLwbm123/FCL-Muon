"""Aggregate completed arms and the existing same-seed temporal LoRM results."""
import json
import statistics
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/aggregation_utility_20261026'


def read(path):
    return json.loads(path.read_text())


def trajectory(stages):
    result={}
    for rule in ['raw','centered']:
        rows=[[v['all_seen_accuracy'] for v in stage[rule]['per_task']] for stage in stages]
        overall=[sum(row)/len(row) for row in rows]
        result[rule]={
            'matrix_percent':[[100*x for x in row] for row in rows],
            'stage_overall_percent':[100*x for x in overall],
            'stage_old_percent':[None if t==0 else 100*sum(row[:-1])/t for t,row in enumerate(rows)],
            'stage_new_percent':[100*row[t] for t,row in enumerate(rows)],
            'final_overall_percent':100*overall[-1],
            'final_old_percent':100*sum(rows[-1][:-1])/2,
            'final_new_percent':100*rows[-1][-1],
            'mean_new_T2_T3_percent':100*(rows[1][1]+rows[2][2])/2,
            'AIA_percent':100*sum(overall)/3,
            'forgetting_pp':100*sum(max(rows[t][k] for t in range(k,3))-rows[-1][k] for k in range(2))/2,
            'restricted_matrix_percent':[[100*v['restricted_accuracy'] for v in s[rule]['per_task']] for s in stages],
            'task_group_confusion':[s[rule]['task_group_confusion'] for s in stages],
            'restricted_correct_stolen_counts':[[v['restricted_correct_but_other_task_wins'] for v in s[rule]['per_task']] for s in stages],
            'old_to_current_rate':[s[rule]['old_to_current'] for s in stages],
            'current_to_old_rate':[s[rule]['current_to_old'] for s in stages],
        }
    return result


def main():
    now=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
    result={'generated_at':now,'scope':'CIFAR100 alpha .05 seed0 first three tasks',
            'reporting_priority':'User explicitly prioritizes exceeding LoRM; report original mechanism gate separately without rewriting it.',
            'arms':{},'mechanism':{},'cost':{},'audits':{},'sources':{}}
    lorm=[read(ROOT/f'reports/ten_task_20261026/lorm_seed0/task{t}_fixed_rules.json') for t in [1,2,3]]
    result['arms']['lorm']=trajectory(lorm)
    result['sources']['lorm']='reports/ten_task_20261026/lorm_seed0/task{1,2,3}_fixed_rules.json; original temporal evaluations, not final T10 weights'
    for mode in ['rrs','central','federated']:
        folder=OUT/mode
        if not (folder/'FINAL_CHECK.json').exists():continue
        audit=read(folder/'FINAL_CHECK.json');assert audit['passed']
        metrics=read(folder/'metrics.json');q=read(folder/'QUALIFICATION.json')
        stages=[]
        for stage in metrics['stages']:
            raw={'per_task':[{'all_seen_accuracy':acc,'restricted_accuracy':restrict,
                             'restricted_correct_but_other_task_wins':round(1000*(restrict-acc))}
                            for acc,restrict in zip(stage['per_task'],stage['restricted'])],
                 'old_to_current':stage['old_to_new_error'],'current_to_old':stage['new_to_old_error']}
            cm=stage['confusion'];n=stage['task']+1
            raw['task_group_confusion']=[[sum(cm[i][j] for i in range(10*a,10*a+10) for j in range(10*b,10*b+10)) for b in range(n)] for a in range(n)]
            stages.append({'raw':raw,'centered':stage['centered']})
        assert len(stages)==3
        result['arms'][mode]=trajectory(stages);result['audits'][mode]=audit
        result['sources'][mode]=f'reports/aggregation_utility_20261026/{mode}/metrics.json, QUALIFICATION.json, FINAL_CHECK.json'
        mechanism=[]
        for rec in q['selection_records']:
            accepted=[v for v in rec['layers'].values() if v['accepted']]
            mechanism.append({'task':rec['task']+1,'accepted_layers':len(accepted),'fallback_layers':12-len(accepted),
                'fit_samples':rec['fit_samples'],'check_samples':rec['check_samples'],'selection_seconds':rec['seconds'],
                'median_accepted_fit_utility_relative_gain':statistics.median(v['candidate']['utility']/v['baseline']['utility']-1 for v in accepted) if accepted else None,
                'accepted_norm_matched_actual_fed_check_improvements':sum(v['actual_federated_check_matched_update_norm']>v['actual_federated_check_baseline'] for v in accepted),
                'layers':rec['layers']})
        result['mechanism'][mode]=mechanism
        probes=q['probe_records']
        result['cost'][mode]={k:q[k] for k in ['training_visits_including_shared_prefix','new_branch_training_visits',
            'stats_visits_including_shared_prefix','new_branch_seconds','timing_and_peak_scope','peak_gpu_allocated_bytes','checkpoint_bytes','logical_communication_bytes']}
        result['cost'][mode].update(prototype_forward_visits=sum(v['samples'] for v in probes if v['kind']=='prototype_fit'),
            gradient_forward_and_backward_visits=sum(v['samples'] for v in probes if v['kind'] in ['fit_gradient','check_gradient']),
            logical_communication_total_bytes=sum(q['logical_communication_bytes'].values()))
    fed=result['arms']['federated'];rrs=result['arms']['rrs'];lo=result['arms']['lorm']
    keys=['final_overall_percent','final_old_percent','final_new_percent','mean_new_T2_T3_percent','AIA_percent','forgetting_pp']
    result['federated_minus_lorm_pp']={rule:{k:fed[rule][k]-lo[rule][k] for k in keys} for rule in ['raw','centered']}
    result['current_user_objective_observed']={rule:fed[rule]['final_overall_percent']>lo[rule]['final_overall_percent'] for rule in ['raw','centered']}
    if 'central' in result['arms']:
        cen=result['arms']['central']['centered'];f=fed['centered'];r=rrs['centered']
        gate={'T3_overall_above_RRS_and_central':f['final_overall_percent']>max(r['final_overall_percent'],cen['final_overall_percent']),
              'AIA_above_RRS_and_central':f['AIA_percent']>max(r['AIA_percent'],cen['AIA_percent']),
              'old_not_below_RRS':f['final_old_percent']>=r['final_old_percent'],
              'mean_new_not_below_RRS':f['mean_new_T2_T3_percent']>=r['mean_new_T2_T3_percent'],
              'both_tasks_accept_and_certified':all(v['accepted_layers']>0 and v['accepted_certificates_pass'] for v in result['audits']['federated']['selection_audits'])}
        result['original_mechanism_gate']={'checks':gate,'passed':all(gate.values())}
    else:result['original_mechanism_gate']={'status':'pending central completion'}
    result['limitations']=['One seed and three tasks for new mechanism; no claim of new-method ten-task or multiseed superiority.',
        'Shared development test set; no independent held-out confirmation or significance claim.',
        'LoRM uses standard rank16/alternating A-B/AdamW/linear-bias head/mixed precision; new method rank32/SSB/cosine head/FP32. Full-system comparison, not single-mechanism attribution.',
        'Original RRS already exceeds LoRM. Added aggregation-utility selection gives a small increment over RRS.',
        'New method current-task accuracy is lower than LoRM; report retention and acquisition jointly.',
        'Original total clock unchanged. Complete recorded training visits875000; uncommitted work from user-requested runtime handovers unknown, upper bound5420 additional visits.',
        'RRS resumed wall/peak are current segment only; shared-GPU time is not a controlled speed comparison.']
    if (OUT/'GROUP_FINAL_CHECK.json').exists():
        result['group_final_check']=read(OUT/'GROUP_FINAL_CHECK.json')
        result['study_qualification']=read(OUT/'STUDY_QUALIFICATION.json')
        result['status']='ALL_THREE_ARMS_COMPLETE_VERIFIED'
        assert result['group_final_check']['passed'] and set(result['audits'])=={'rrs','central','federated'}
    else:result['status']='INTERIM_COMPLETED_ARMS_ONLY'
    (OUT/'PRO_REPORT_DATA.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    names={'lorm':'标准 LoRM','rrs':'原 RRS + Head-RegMean','central':'集中式效用选子空间','federated':'Pro 联邦聚合效用新方法'}
    lines=['给 Pro 的实验报告：联邦聚合效用子空间方法 vs LoRM',f'更新：{now}（北京时间）','',
        '一、当前问题与结论',
        '用户当前目标是整体方法高于 LoRM，不要求新增机制相对原 RRS 有较大增益。训练协议没有因此调整，原机制判定标准保留为诊断记录。',
        f"已完成的新方法在 seed0 前三任务上达到：去均值总体准确率 {fed['centered']['final_overall_percent']:.3f}%，LoRM {lo['centered']['final_overall_percent']:.3f}%，高 {result['federated_minus_lorm_pp']['centered']['final_overall_percent']:.3f} 个百分点。raw 同样更高：{fed['raw']['final_overall_percent']:.3f}% vs {lo['raw']['final_overall_percent']:.3f}%，高 {result['federated_minus_lorm_pp']['raw']['final_overall_percent']:.3f} 个百分点。",
        '这支持“当前三任务整体方案高于标准 LoRM”的描述。优势主要来自旧类保持；新任务学习仍弱于 LoRM。新增机制自身相对原 RRS 的增量较小，应与整体方案效果分开解释。','',
        '二、实验设置与公平比较口径',
        'CIFAR100，前3个任务，每任务10类；10客户端，Dirichlet alpha=0.05，seed=0；每任务5轮，每轮客户端本地5 epochs，batch=16。T1/T2/T3分别评估1000/2000/3000张测试图像。',
        'LoRM来自已完成并验证的标准十任务实验的 T1/T2/T3 当时评估记录；只截取相同前三任务，不使用T10模型回测，也不把三任务与十任务终点混比。双方复用相同数据划分和固定预训练权重来源。',
        '主读出为固定的逐任务类别块logit减均值：对每个已见10类块减去该块均值，再在所有已见类中argmax；无需真实测试任务ID。双方均同时报告此规则与raw。',
        'LoRM保持官方标准配置：rank16，A/B交替更新，AdamW lr=5e-4，linear+bias head，16-mixed，regmean_all，head alpha=.5/backbone alpha=0。新方法rank32，B采用原SSB，cosine16 head及Head-RegMean alpha=.5，FP32；这是整体方案对比，不能将对LoRM的所有收益归因于新子空间目标。','',
        '三、新机制的具体实现',
        '在每层原RRS 32维基A0外扩展32个谱方向，形成64维候选池P；优化64×32的正交Z，正式训练仍使用A=(PZ)^T，rank保持32。',
        '当前训练样本按客户端/类别固定约80/20拆分fit/check。只用fit构建临时cosine原型探针头；零B探针得到客户端投影梯度C_k=G_k P。正式随机分类头不变，探针恢复RNG，不加入旧图像回放。',
        '联邦目标：U_fed(Z)=sum_Q,V <sum_k p_k C_k Z, sum_k p_k SSB(0.1 C_k Z)>。集中式对照把右侧改为SSB(0.1 sum_k p_k C_k Z)。先投影到rank32再做原SSB，Q/V独立归一化，不用FedMuon额外缩放。',
        '同时约束历史暴露Gamma与实际联邦更新的历史响应R不超过原RRS预算（仅允许预先设定数值容差）。每层最多60步，仅fit选候选；check只验一次fit更新的独立效用，不通过则精确退回A0，不据check反复搜索。',
        '正式训练保持current-only CE、B的SSB/FedAvg、cosine head、Head-RegMean；没有边界融合、teacher、蒸馏或额外推理模型。三组共用一次新鲜T1的模型/history/RNG；T2/T3各自推进。','',
        '四、T3终点与序列指标（单位%；遗忘为百分点，越低越好）']
    for rule,label in [('centered','固定去均值'),('raw','Raw')]:
        lines += ['',label,'方法 | 总体 | 旧类T1–T2 | 新类T3 | T2–T3平均新类 | AIA | 遗忘']
        for mode in ['lorm','rrs','central','federated']:
            if mode not in result['arms']:lines.append(names[mode]+' | 尚未完成，暂不填最终数值');continue
            m=result['arms'][mode][rule];lines.append(names[mode]+' | '+' | '.join(f'{m[k]:.3f}' for k in keys))
    lines += ['', 'AIA为T1/T2/T3三个阶段总体准确率的均值；平均新类为T2和T3各自刚学完时准确率均值；遗忘为最终时T1/T2相对各自历史最佳的平均下降。','',
        '五、完整前三任务准确率矩阵（每行依次是训练至T1、T2、T3后各任务的all-seen准确率）']
    for rule in ['centered','raw']:
        for mode in ['lorm','rrs','central','federated']:
            if mode in result['arms']:
                rows=result['arms'][mode][rule]['matrix_percent']
                lines.append(f"{names[mode]} / {rule}: "+' ; '.join('['+', '.join(f'{x:.2f}' for x in row)+']' for row in rows))
    lines += ['', '六、机制是否实际执行，以及它解释了多少增益',
        '新机制T2接受12/12层，T3接受10/12层；T3第4和第11层（0-based）因实际联邦历史响应约束未通过而精确回退。不是全部回退后原样训练。',
        'T2 fit/check=4009/991；T3=4007/993。接受层fit效用相对增益中位数分别约22.28%和25.72%；匹配更新范数后，独立check效用仍高于原RRS的接受层数分别为12/12、9/10。这只是虚拟一步代理量，不能等同于最终分类增益。',
        '新方法相对原RRS的T3去均值提升为0.033个百分点（多判对1/3000张），raw提升0.067个百分点（2/3000张）。原RRS整体方案本身已经明显高于LoRM，因而对LoRM的主要差距不能全部算作此次子空间选择的贡献。',
        f"相对LoRM，新方法去均值旧类准确率提高 {result['federated_minus_lorm_pp']['centered']['final_old_percent']:.3f} 个百分点，但T3新类下降 {-result['federated_minus_lorm_pp']['centered']['final_new_percent']:.3f} 个百分点，T2–T3平均新类下降 {-result['federated_minus_lorm_pp']['centered']['mean_new_T2_T3_percent']:.3f} 个百分点。因此当前证据更支持改善保留/跨任务竞争，而非各项学习能力均提高。",'',
        '七、执行检查与成本',
        '新方法已完成150个唯一client-round、375000次训练图像访问（包含共享T1的50/125000）、90000次基础统计访问、15个头聚合证书、3次merge/history提交。实际最终checkpoint全3000张raw/去均值重载预测一致；已完成一次CPU审计，包括split/prototype计数、接受证书、精确回退和实际保存的T3子空间矩阵。',
        '新机制T2/T3额外原型forward共8016张，梯度探针forward+backward共10000张；探针、统计与通信均计入成本，不是免费步骤。GPU7的T2–T3分支及最终验证约73.83分钟，torch记录峰值约2.045GB；该值不包含共享T1，也不是nvidia-smi整卡占用，不据共享GPU耗时宣称速度优势。',
        '三组记录内总训练访问875000；两次用户要求的调度迁移可能带来未提交客户端重算，额外实际访问未知、上界5420，原实验时钟未重置。数据、梯度、Gram、原型、权重、逐图logits保留服务器私有。','',
        '八、结论边界与给Pro的问题',
        '当前用户关心的比较已满足：在alpha=.05、seed0、前三任务上，完整新方案总体准确率高于标准LoRM，raw和去均值都成立。',
        '尚不能声称新机制已完成十任务或多seed验证，也不能把已有旧主线的十任务结果当作新机制结果。共享测试集属于开发证据，没有独立held-out或显著性结论。',
        '请基于上述完整数值判断：如何把“子空间中的联邦更新效用约束”与整体方案较好的旧类保持连成可检验、不过度归因的机制解释；如何解释新类学习仍弱于LoRM，以及代理效用较大提升但相对原RRS分类变化很小。当前不以新增机制必须大幅超过原RRS作为否定整体方案的标准。',
        '原冻结机制门槛（联邦组同时超过RRS和集中式组等）单独保留：'+json.dumps(result['original_mechanism_gate'],ensure_ascii=False),
        '', '证据：PRO_REPORT_DATA.json含完整矩阵、逐阶段old/new/overall、restricted/competition、逐层机制诊断及成本；各组metrics.json、QUALIFICATION.json、FINAL_CHECK.json；LoRM同seed的task1/2/3_fixed_rules.json。']
    if 'central' in result['arms']:
        lines += ['', '集中式对照补充：集中式组最终去均值准确率94.033%，与联邦组相同；raw为91.233%，联邦组91.100%。因此原先要求联邦组严格优于集中式组的机制门槛未通过。这不改变本次报告的主结论：完整新方案在同阶段总体准确率上高于LoRM；联邦目标相对集中式目标的独有收益目前尚未建立。']
    if result['status']=='ALL_THREE_ARMS_COMPLETE_VERIFIED':
        group=result['group_final_check']
        lines += ['',f"完成状态：三个实验组均完成并通过实际checkpoint验收，两个worker均以0退出。共享原始时钟总历时{group['group_wall_seconds']/60:.2f}分钟，守护退出时新目录实际占用{group['allocated_bytes_at_supervisor_exit']/1e9:.3f}GB。所有历史证据和私有模型/梯度保留；本报告无需新增实验或重复评估。"]
    (OUT/'PRO_REPORT_ZH.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'arms':list(result['arms']),'vs_lorm':result['federated_minus_lorm_pp'],'original_gate':result['original_mechanism_gate']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
