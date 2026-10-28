"""Aggregate-only FCCA closeout; recompute the three-stage comparison and gates."""
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from summarize_boundary_three import normalize, trajectory

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/fcca_three_20261028'
def read(path):
    return json.loads(path.read_text())

def prefix(method):
    stages = []
    for t in range(3):
        stage = {}
        for mode in ('raw', 'centered'):
            m = method[mode]
            stage[mode] = {
                'per_task': [{'n': 1000, 'all_seen_accuracy': a / 100,
                              'restricted_accuracy': b / 100,
                              'restricted_correct_but_other_task_wins': s}
                             for a, b, s in zip(m['matrix_percent'][t], m['restricted_matrix_percent'][t],
                                                m['restricted_correct_stolen_counts'][t])],
                'task_group_confusion': m['task_group_confusion'][t],
                'old_to_current': m['old_to_current_rate'][t],
                'current_to_old': m['current_to_old_rate'][t]}
        stages.append(stage)
    return trajectory(stages)

def main():
    q = read(OUT / 'QUALIFICATION.json'); job = read(OUT / 'job.json')
    complete = read(OUT / 'COMPLETE.json'); pair = read(OUT / 'PAIRED_C_VS_A.json')
    cfg = read(ROOT / 'configs/fcca_three_seed0.json')['fcca']
    assert q['qualified'] and complete['qualified'] and job['qualified'] and job['exit_code'] == 0
    assert q['client_rounds'] == 150 and q['training_visits'] == 375000 and q['new_training_visits'] == 250000
    assert q['merges'] == q['history_committed'] == [0, 1, 2] and q['certificates_passed'] == 15
    assert q['reload_images'] == 3000 and q['reload_raw_predictions_equal'] and q['reload_centered_predictions_equal']
    assert q['clock'] == {k: job[k] for k in ('started', 'deadline')}
    assert job['started'] == 1793160824.1203914 and job['ended'] < job['deadline']
    assert job['ended'] - job['started'] < 86400 and job['allocated_bytes'] < job['write_stop_bytes'] < 50_000_000_000
    methods = {k: prefix(v) for k, v in read(ROOT / 'reports/boundary_repair_20261027/FINAL_COMPARISON.json')['methods'].items()}
    methods['C'] = trajectory([normalize(s) for s in q['stages']])
    decomposition = {}
    identities = 0
    for name, method in methods.items():
        decomposition[name] = {}
        for mode, m in method.items():
            out = []
            for t, row in enumerate(m['matrix_percent']):
                counts = []
                for i, acc in enumerate(row):
                    w = 1000 - round(m['restricted_matrix_percent'][t][i] * 10)
                    s = m['restricted_correct_stolen_counts'][t][i]
                    error = 1000 - round(acc * 10)
                    assert w + s == error
                    counts.append({'W': w, 'S': s, 'errors': error}); identities += 1
                confusion = m['task_group_confusion'][t]
                assert all(sum(r) == 1000 for r in confusion)
                out.append({'per_task': counts, 'W': sum(r['W'] for r in counts),
                            'S': sum(r['S'] for r in counts), 'errors': sum(r['errors'] for r in counts),
                            'old_to_new_count': sum(confusion[i][t] for i in range(t)),
                            'new_to_old_count': sum(confusion[t][:t])})
            decomposition[name][mode] = out
    for t in range(1, 4):
        for mode in ('raw', 'centered'):
            for group, record in pair[str(t)][mode].items():
                assert record['net_correct'] == record['corrected'] - record['broken']
                assert record['changed_predictions'] == record['corrected'] + record['broken'] + record['both_wrong_but_changed']
            delta = methods['C'][mode]['stage_overall_percent'][t-1] - methods['A'][mode]['stage_overall_percent'][t-1]
            assert pair[str(t)][mode]['all']['net_correct'] == round(delta * t * 10)
    records = q['fcca_records']; assert len(records) == len({(r['task'], r['round']) for r in records}) == 10
    rounds = []; reasons = Counter()
    for r in records:
        reason = []
        for split in ('fit', 'check'):
            base = r['baseline_' + split]; cand = r['candidate_' + split]
            if cand['raw_ce'] > base['raw_ce'] - cfg['minimum_relative_raw_gain'] * max(base['raw_ce'], 1e-12):
                reason.append(split + '_raw_gain')
            if cand['within_ce'] > base['within_ce'] + cfg['within_ce_absolute_tolerance'] + cfg['within_ce_relative_tolerance'] * base['within_ce']:
                reason.append(split + '_within_ce')
        if r['history_candidate_actual_FP32'] > r['history_limit']:
            reason.append('actual_FP32_history_guard')
        assert reason == r['reasons'] and r['accepted'] == (not reason)
        assert r['solver']['success'] and r['fixed_actual_HeadRegMean'] and r['check_used_once'] and not r['old_class_guarantee']
        assert r['deployed_weights'] == (r['proposed_weights'] if r['accepted'] else r['sample_weights'])
        assert abs(sum(r['proposed_weights'])-1) <= cfg['weight_tolerance']
        assert all(.5*p-cfg['weight_tolerance'] <= a <= 1.5*p+cfg['weight_tolerance'] for p,a in zip(r['sample_weights'],r['proposed_weights']))
        reasons.update(reason)
        delta = {split: {key: r['candidate_'+split][key] - r['baseline_'+split][key] for key in ('raw_ce','within_ce')}
                 for split in ('fit','check')}
        rounds.append({'task':r['task'],'round':r['round'],'accepted':r['accepted'],'reasons':reason,
                       'predicted_fit_raw_ce_change':r['solver']['predicted_fit_raw_ce_change'],
                       'actual_change':delta,'history_actual_minus_limit':r['history_candidate_actual_FP32']-r['history_limit'],
                       'deployed_weight_L1_from_sample':sum(abs(a-p) for a,p in zip(r['deployed_weights'],r['sample_weights']))})
    c = methods['C']; a = methods['A']
    keys = ('final_overall_percent','final_old_percent','final_new_percent','AIA_percent','forgetting_pp')
    differences = {name:{mode:{k:c[mode][k]-m[mode][k] for k in keys} for mode in ('raw','centered')}
                   for name,m in methods.items() if name != 'C'}
    gate = {'T3_raw_strictly_above_A': c['raw']['final_overall_percent'] > a['raw']['final_overall_percent'] + 1e-10,
            'raw_AIA_not_below_A': c['raw']['AIA_percent'] >= a['raw']['AIA_percent'] - 1e-10,
            'T3_raw_old_not_below_A': c['raw']['final_old_percent'] >= a['raw']['final_old_percent'] - 1e-10,
            'T3_centered_not_below_A': c['centered']['final_overall_percent'] >= a['centered']['final_overall_percent'] - 1e-10,
            'some_candidate_accepted': any(r['accepted'] for r in records), 'numerical_lifecycle_qualified': q['qualified']}
    comm = q['logical_communication_bytes']; extra_comm = sum(v for k,v in comm.items() if k.startswith('fcca_'))
    cost = {k:q[k] for k in ('extra_forward_images','extra_backward_images','extra_forward_calls','extra_backward_calls',
                            'peak_gpu_allocated_bytes','checkpoint_bytes','elapsed_seconds','logical_communication_bytes')}
    cost.update(full_training_visits=375000,reused_training_visits=125000,new_training_visits=250000,
                full_client_rounds=150,reused_client_rounds=50,new_client_rounds=100,
                full_statistics_images=90000,new_statistics_images=60000,new_stage_evaluation_images=5000,reload_images=3000,
                logical_communication_total_bytes=sum(comm.values()),fcca_extra_logical_bytes=extra_comm,
                base_logical_communication_bytes=sum(comm.values())-extra_comm,
                fcca_fuse_wall_seconds=sum(r['seconds'] for r in records),
                supervisor_total_seconds=job['ended']-job['started'],allocated_bytes=job['allocated_bytes'],
                initial_allocated_bytes=job['baseline_allocated_bytes'],real_preflight=read(OUT/'REAL_PREFLIGHT.json'),
                selector_probe_records=q['probe_records'])
    result = {'status':'COMPLETE_QUALIFIED_SCIENTIFIC_PROGRESS_GATE_NOT_MET','methods':methods,'C_minus_reference_pp':differences,
              'error_decomposition':decomposition,'paired_C_vs_A':pair,'round_summary':rounds,'fcca_records':records,
              'rejection_reason_counts_nonexclusive':dict(reasons),'scientific_conditions':gate,'scientific_progress':all(gate.values()),
              'cost':cost,'clock':q['clock'],'qualification':{k:q[k] for k in ('qualified','client_rounds','training_visits','merges',
              'history_committed','certificates_passed','reload_images','reload_raw_predictions_equal','reload_centered_predictions_equal','maximum_reload_logit_difference')},
              'checks':{'W_plus_S_identities':identities,'paired_stage_readouts':6,'candidate_acceptance_recomputed':10},
              'warnings':['SciPy intermediate iterate outside box was clipped; all ten final candidates passed solver feasibility; no runtime failure.'],
              'limitations':['Single seed and reused development test set; not independent confirmation.',
              'Fit/check are disjoint probe subsets of training data, not representation-held-out validation.',
              'Historical response constraint is a proxy, not an old-class risk guarantee.',
              'LoRM is a full-system comparator with different rank/optimizer/head/precision; C minus A isolates this added module.',
              'No measured standalone CPU solver timing or controlled speed comparison; logical communication is not network traffic.',
              'No automatic ten-task extension, threshold change, or retrospective candidate search.'],
              'sources':{'C':'QUALIFICATION.json','pair':'PAIRED_C_VS_A.json','references':'../boundary_repair_20261027/FINAL_COMPARISON.json',
                         'protocol':'../../research/fcca/PROTOCOL_ZH.txt','source_commit':read(OUT/'SOURCE.json')['source_commit']}}
    assert not result['scientific_progress']
    (OUT/'FINAL_REPORT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines = ['FCCA-C 三任务完整报告（seed0，2026-10-28）','',
        '结论：训练和数值/生命周期验收完成，但未达到冻结科学进展条件。C与A的T3 raw及raw AIA持平，centered预测完全一致；没有证据表明新增模块改善了三任务总体性能。',
        'T3 raw只改变2张预测：纠正1张旧类、破坏1张新类，净收益0。T2 raw改变1张但双方均错误。C高于标准LoRM是完整系统结果，不能归因于FCCA。','',
        '设置：CIFAR100前三任务/每任务10类；seed0，Dirichlet .05，10客户端、每任务5轮、本地5epochs、batch16、rank32、cosine16、FP32。C保留A的联邦选空间、SSB和Head-RegMean，关闭B局部修正；复用同一合法T1，新增T2–T3。raw为本轮主指标，centered完整并报。',
        'A/B/LoRM全部截取既有当时模型的前三阶段记录；不使用T10模型回测，不重跑/调优比较方法。LoRM为标准rank16、交替A/B AdamW、线性带bias头和混合精度系统，对LoRM差距不能解释为单模块因果效果。','']
    for mode in ('raw','centered'):
        lines += [mode+'读出，单位%（遗忘为百分点）','方法 | T3 overall | old | new | AIA | forgetting']
        for name in ('LoRM','A','B','C'):
            m=methods[name][mode];lines.append(name+' | '+' | '.join(f'{m[k]:.6f}' for k in keys))
        for name in ('LoRM','A','B','C'):
            m=methods[name][mode]
            lines += [name+'准确率下三角矩阵：'+json.dumps(m['matrix_percent']),
                      name+'任务内restricted矩阵：'+json.dumps(m['restricted_matrix_percent']),
                      name+'逐阶段overall/old/new：'+json.dumps({k:m[k] for k in ('stage_overall_percent','stage_old_percent','stage_new_percent')}),
                      name+'逐阶段W/S/总错误、旧→新/新→旧计数：'+json.dumps(decomposition[name][mode]),
                      name+'逐阶段任务组混淆矩阵（行真值、列预测、每行1000）：'+json.dumps(m['task_group_confusion'])]
        lines += ['C相对各对照差值（百分点）：'+json.dumps({k:v[mode] for k,v in differences.items()}),
                  'C相对A同ID配对（all/old/new）：'+json.dumps({t:v[mode] for t,v in pair.items()}),'']
    lines += ['AIA为三个阶段总体准确率等权平均；old仅此前任务，new为当前任务。遗忘对T1/T2分别取学习后至T3最高准确率减T3准确率，再平均。',
              'W为任务内restricted错误；S为任务内正确却被其他任务类别抢走；总错误=W+S。旧→新包含所有旧图被预测为当前任务的情况，不能直接等同S。共核对48个任务级分解恒等式及6个阶段/读出的配对净正确数。','',
              '聚合机制与每轮完整证据',
              '10轮均求得可行候选，2轮接受（T2R3/R4）；8轮回退原FedAvg。T3五轮全部拒绝。fit raw CE十轮均下降且一阶预测同号；check未支持普遍收益。T3R5的fit损失虽下降但未达到相对下降阈值，不应说成fit恶化。',
              'T2R1/R5实际FP32历史代价分别超限约3.569e-12/2.658e-13，即使很小也按原阈值拒绝，没有放宽协议。T3R1–R4的check raw与within约束失败，R5的fit/check raw收益阈值失败。',
              'SciPy出现一次中间迭代越box后自动clip的RuntimeWarning；不是训练失败，最终候选约束已另验。技术运行无FAILURE，退出码0。',
              '每条记录包括样本权重、候选权重、实际部署权重、候选一阶预测、fit/check基准及候选raw/within CE、实际历史代价和上限。负的CE差表示下降。完整交叉客户端方向内积在QUALIFICATION/JSON中；它是梯度与更新方向的聚合标量，不是原始梯度。']
    for r,s in zip(records,rounds):
        lines += [f"T{r['task']} R{r['round']}："+json.dumps(s,ensure_ascii=False),
                  '客户端顺序：'+json.dumps(r['clients']),
                  '样本p：'+json.dumps(r['sample_weights']),'候选alpha：'+json.dumps(r['proposed_weights']),
                  '部署alpha：'+json.dumps(r['deployed_weights']),
                  '基准/候选损失：'+json.dumps({k:{n:r[k][n] for n in ('raw_ce','within_ce','probe_images','represented_training_images','client_probe_counts')}
                                                for k in ('baseline_fit','candidate_fit','baseline_check','candidate_check')}),
                  '历史baseline/candidate_FP32/limit：'+json.dumps([r['history_baseline'],r['history_candidate_actual_FP32'],r['history_limit']])]
    lines += ['', '科学进展条件：'+json.dumps(gate,ensure_ascii=False),
              '唯一未满足项是T3 raw严格超过A。其余通过不能替代这个条件，也不构成多seed显著收益；本轮不扩十任务。','',
              '成本与资源',
              f"完整150客户端轮/375000训练访问，其中复用50轮/125000，本次新增100轮/250000。新增统计60000图，新增阶段评测5000图，最终重载3000图。",
              f"FCCA额外前向{q['extra_forward_calls']}次/{q['extra_forward_images']}图，额外反向{q['extra_backward_calls']}次/{q['extra_backward_images']}图；另有启动preflight前向2次/32图、反向2次/32图。以上FCCA计数不包含原选空间成本，原探针逐条保留于JSON。",
              f"FCCA新增逻辑通信{extra_comm}字节，完整累计{sum(comm.values())}字节，去掉FCCA项的同口径基础通信{sum(comm.values())-extra_comm}字节。完整累计包括复用T1账本，不等于本次新增网络传输。",
              '逻辑通信分项：'+json.dumps(comm),
              f"FCCA fuse区间累计{cost['fcca_fuse_wall_seconds']:.3f}秒，含原头聚合与额外探针/CPU计算，不能称为纯新增开销；CPU求解器未单独计时。重载验收写出前总历时{q['elapsed_seconds']:.3f}秒，监督进程结束历时{cost['supervisor_total_seconds']:.3f}秒。共享硬件上不作速度优越性结论。",
              f"任务根最终实际分配{job['allocated_bytes']}字节，初始{job['baseline_allocated_bytes']}字节；checkpoint {q['checkpoint_bytes']}字节；GPU峰值分配{q['peak_gpu_allocated_bytes']}字节。低于6GB写停止/8GB上限。最终占用不是峰值存储计量。",
              '开始/结束（Asia/Shanghai）：'+str(datetime.fromtimestamp(job['started'],ZoneInfo('Asia/Shanghai')))+' / '+str(datetime.fromtimestamp(job['ended'],ZoneInfo('Asia/Shanghai'))),
              '原始时钟未重置：'+json.dumps(q['clock']),
              f"150唯一客户端轮由训练入口完成时断言；3次merge/history、15次实际FP32头证书、10唯一聚合均通过。最终3000图raw与centered重载预测一致，最大logit差{q['maximum_reload_logit_difference']:.12g}，不声称logits逐位一致。",'',
              '证据边界与交付',
              '本轮只支持：跨客户端方向统计和统一部署头的计算可运行，有限步验收会拒绝fit改善但check不支持的候选；没有显示三任务总体增益。不能由2次接受证明改善泛化，也不能由历史响应预算推出旧类分类保证。',
              'fit/check虽互斥，都来自参与本地训练的数据；这不是独立验证集。只测seed0，既有开发测试集和路线选择历史均限制泛化解释。未按测试结果改阈值、补候选或重跑。',
              '重建：在仓库根执行 python3 scripts/summarize_fcca_three.py（标准库，读取公开JSON）。服务器配对使用 scripts/pair_boundary_outputs.py，TASK_COUNT=3，A为既有A逐阶段私有输出，C为本轮逐阶段私有输出；逐样本ID/标签一致后仅导出聚合计数。',
              '科学实现版本、配置和协议见SOURCE.json；全部来源/资格/启动记录与最终报告同目录。公开源码/配置/协议和聚合损失、权重、计数；checkpoint、逐图数据/ID、探针索引、原始梯度、Gram和特征继续私有保存。']
    (OUT/'FINAL_REPORT_ZH.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'gate':gate,'C':{m:{k:c[m][k] for k in keys} for m in c},'cost':{k:v for k,v in cost.items() if not isinstance(v,(list,dict))}},indent=2))

if __name__ == '__main__':
    main()
