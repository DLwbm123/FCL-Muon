"""Aggregate all registered seeds, validate saved evidence, never train or select."""
import json
import math
import statistics as st
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/actual_update_repeat_20261029'


def describe(values):
    return {'mean':st.mean(values),'sample_std':st.stdev(values) if len(values)>1 else None,
            'min':min(values),'max':max(values),'values':values}


def main():
    seeds={};audit=[];operators={}
    for seed in (0,1,2):
        folder=ROOT/'reports/actual_update_global_20261029' if seed==0 else OUT/f'seed{seed}'
        report=json.loads((folder/'FINAL_REPORT.json').read_text())
        assert report['status']=='COMPLETED_QUALIFIED'
        seeds[str(seed)]=report
        for arm in ('control','global'):
            for t in (3,5):
                q=json.loads((folder/arm/f'QUALIFICATION_T{t}.json').read_text())
                assert q['config']['seed']==seed and q['qualified']
                assert q['client_rounds']==t*50 and q['training_visits']==t*125000 and q['reload_images']==t*1000
                assert q['reload_raw_predictions_equal'] and q['reload_centered_predictions_equal']
                assert q['extra_probe_forward_images']==q['extra_probe_backward_images']==0
                assert len(q['actual_update_rounds'])==t*5
                audit.append({'seed':seed,'arm':arm,'task':t,'qualified':True,'reload_images':q['reload_images'],
                              'max_reload_logit_difference':q['maximum_reload_logit_difference']})
            for view in ('raw','centered'):
                rows=report['arms'][arm][view];assert len(rows)==5
                for t,row in enumerate(rows,1):
                    assert row['task']==t and row['images']==t*1000
                    assert row['W']+row['S']==row['errors']==round((100-row['overall_percent'])*row['images']/100)
                    assert abs(row['AIA_percent']-st.mean(v['overall_percent'] for v in rows[:t]))<1e-10
                assert abs(rows[-1]['overall_percent']-100*(q['stages'][-1]['all_seen'] if view=='raw' else q['stages'][-1]['centered']['all_seen_accuracy']))<1e-10
            records=[v for rnd in q['actual_update_rounds'] for pair in rnd['layers'].values() for v in pair]
            assert len(records)==600
            for v in records:
                assert v['norm_relative_error']<=1e-10 and v['actual_FP32_norm_relative_error']<=5e-4
                assert abs(sum(v['energy_before'])-1)<1e-10 and abs(sum(v['energy_after'])-1)<1e-10
                if arm=='control':assert v['actual_FP32_relative_correction']==0
            grouped={}
            for task in range(1,6):
                rr=[v for rnd in q['actual_update_rounds'] if rnd['task']==task for pair in rnd['layers'].values() for v in pair]
                grouped[str(task)]={'records':len(rr),'relative_correction':describe([v['actual_FP32_relative_correction'] for v in rr]),
                    'effective_rank_delta':describe([v['effective_energy_rank_after']-v['effective_energy_rank_before'] for v in rr]),
                    'top_mode_energy_delta':describe([v['energy_after'][0]-v['energy_before'][0] for v in rr]),
                    'aggregate_to_client_rms_ratio':describe([v['input_norm']/math.sqrt(v['client_mean_squared_norm']) for v in rr])}
            operators[f'{seed}/{arm}']={'records':600,'active_records':sum(v['identity_reason'] is None for v in records),
                'mean_relative_correction':st.mean(v['actual_FP32_relative_correction'] for v in records),
                'max_relative_correction':max(v['actual_FP32_relative_correction'] for v in records),
                'maximum_deployed_norm_error':max(v['actual_FP32_norm_relative_error'] for v in records),
                'maximum_angle_degrees':max(math.degrees(math.acos(max(-1,min(1,v['cosine_with_fedavg'])))) for v in records),
                'rank_increase_count':sum(v['effective_energy_rank_after']>v['effective_energy_rank_before']+1e-10 for v in records),
                'top_mode_energy_decrease_count':sum(v['energy_after'][0]<v['energy_before'][0]-1e-10 for v in records),
                'by_task':grouped}
        for t in range(1,6):
            for view in ('raw','centered'):
                pair=report['paired_vs_control'][str(t)][view]
                delta=report['arms']['global'][view][t-1]['overall_percent']-report['arms']['control'][view][t-1]['overall_percent']
                assert pair['corrected']-pair['destroyed']==pair['net_corrected']
                assert abs(delta-100*pair['net_corrected']/pair['images'])<1e-10
    fields=['overall_percent','old_percent','new_percent','AIA_percent','forgetting_pp','signed_forgetting_pp','W','S','old_to_new_percent','new_to_old_percent']
    differences={}
    for view in ('raw','centered'):
        differences[view]={}
        for t in range(1,6):
            differences[view][str(t)]={k:describe([s['arms']['global'][view][t-1][k]-s['arms']['control'][view][t-1][k] for s in seeds.values()]) for k in fields if seeds['0']['arms']['control'][view][t-1][k] is not None}
    last=differences['raw']['5']
    conditions={'mean_raw_overall_at_least_0.2pp':last['overall_percent']['mean']>=.2-1e-10,
        'mean_raw_AIA_nonnegative':last['AIA_percent']['mean']>=-1e-10,
        'mean_raw_old_nonnegative':last['old_percent']['mean']>=-1e-10,
        'every_seed_new_at_least_minus1pp':last['new_percent']['min']>=-1-1e-10,
        'mean_S_not_increased':last['S']['mean']<=0,
        'at_least_two_positive_raw_seeds':sum(v>1e-10 for v in last['overall_percent']['values'])>=2,
        'all_reload_qualified':len(audit)==12 and all(v['qualified'] for v in audit)}
    result={'status':'COMPLETED_QUALIFIED','exploratory_continue_conditions':conditions,
        'exploratory_gate_passed':all(conditions.values()),'registered_seeds':[0,1,2],
        'seed0_original_gate_unchanged':seeds['0']['hypothesis_supported_in_this_pilot'],
        'per_seed':seeds,'paired_difference_statistics':differences,'operator_audit':operators,'qualification_audit':audit,
        'repetition_completion':json.loads((OUT/'COMPLETION.json').read_text()),
        'limitations':['Seeds1/2 registered after observing seed0; exploratory, not independent confirmation.',
            'Three seeds on one dataset/alpha/T5; no statistical significance or T10 claim.',
            'Repeated per-image records across seeds are not independent people or datasets.',
            'Layer/round operator records are not independent experimental replications.',
            'No matched pre-aggregation endpoint or gradient archive: spectral summaries cannot establish loss alignment or causality.']}
    (OUT/'THREE_SEED_REPORT.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['实际更新全局谱修正：三seed五任务探索性复验最终报告','',
        '结论：'+('通过' if result['exploratory_gate_passed'] else '未通过')+'冻结的三seed探索性继续门槛。工程验收全部通过；这不等于科学收益成立。',
        'seed0原失败判定保持。seed1/2为看到seed0后预注册的探索性重复，不是独立确认，不作显著性或T10结论。',
        '配置：CIFAR100 alpha=.05，10客户端；同seed fresh同预训练权重。control eta0、global eta.1，其余训练配方一致；每路线5任务/250客户端轮/625000访问。',
        '', 'T5逐seed（精度为%，差值pp；S为计数）','seed | control raw | global raw | Δraw | Δold | Δnew | ΔAIA | ΔS | 原单seed门槛']
    for seed,s in seeds.items():
        d=s['T5_raw_differences_pp_except_S_count']
        lines.append(f"{seed} | {s['arms']['control']['raw'][-1]['overall_percent']:.4f} | {s['arms']['global']['raw'][-1]['overall_percent']:.4f} | "+' | '.join(f'{d[k]:+.6f}' for k in ['overall_percent','old_percent','new_percent','AIA_percent','S'])+f" | {s['hypothesis_supported_in_this_pilot']}")
    lines+=['','全部任务配对差（global-control；均值±样本标准差；括号为seed0/1/2，绝不选最好seed）']
    for view in ('raw','centered'):
        for t in range(1,6):
            d=differences[view][str(t)]['overall_percent'];lines.append(f"{view} T{t}: {d['mean']:+.6f} ± {d['sample_std']:.6f} pp; {d['values']}")
    lines+=['','T5 raw各指标三seed配对统计：']
    for k,d in last.items():lines.append(f"{k}: mean={d['mean']:+.6f}, sample_std={d['sample_std']:.6f}, min={d['min']:+.6f}, max={d['max']:+.6f}")
    lines+=['','冻结条件：'+json.dumps(conditions,ensure_ascii=False),'','T5同图配对纠正/破坏（逐seed计数，不将跨seed重复图当独立样本）：']
    for seed,s in seeds.items():lines.append(seed+': '+json.dumps(s['paired_vs_control']['5'],ensure_ascii=False))
    lines+=['','机制审计（每seed600个层/QV/通信轮记录，不视作600个独立实验）：']
    for seed in seeds:
        o=operators[f'{seed}/global'];lines.append(f"seed{seed}: active={o['active_records']}/600, correction mean={100*o['mean_relative_correction']:.4f}%, max={100*o['max_relative_correction']:.4f}%, angle max={o['maximum_angle_degrees']:.4f}deg, norm error max={o['maximum_deployed_norm_error']:.3g}, rank increase={o['rank_increase_count']}/600, top-mode decrease={o['top_mode_energy_decrease_count']}/600")
    lines+=['','解释：正负seed均实际部署温和谱修正，不能归因于某seed算子没生效或范数失控。谱能量/有效秩变化不含分类损失方向，不能据此认定放大eta能改善性能。新类与旧类、任务间误判的损益随seed不同，纯谱形目标没有提供稳定分类收益证据。',
        '下一步不增加同配置seed、不扩展T10、不搜索eta。先做训练输入上的共同状态单轮诊断，比较同一批客户端终点产生的原聚合与eta=.1修正的CE/KL变化及梯度方向；不使用测试图或测试标签，目的仅定位谱代理是否与既有训练目标一致。须另立冻结诊断协议和硬预算，不能把该诊断称为泛化收益或据此改写本报告。',
        '', '资源与验收：新增seed1/2四路线全部T3/T5预测重载一致；三seed共12个里程碑通过。每路线625000训练访问、500000 teacher前向、额外probe0；所有成本完整保存在JSON各seed/arm。新增复验共同历时'+f"{result['repetition_completion']['elapsed_seconds']/3600:.4f}小时，退出新增占用{result['repetition_completion']['allocated_bytes_at_exit']/1e9:.4f}GB，记录进程均退出。不同GPU墙钟差异不能解释为方法加速。",
        '完整各seed T1-T5 raw/centered、old/new/AIA/两种遗忘、W+S/方向混淆、逻辑通信/显存/时长见THREE_SEED_REPORT.json与各seed FINAL_REPORT.json；资格逐层谱细节保留。权重、teacher、逐图logits/ID与完整日志留服务器，不公开。']
    (OUT/'THREE_SEED_REPORT_ZH.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'gate':conditions,'T5':last,'operator':{k:{a:b for a,b in v.items() if a!='by_task'} for k,v in operators.items()}},indent=2))


if __name__=='__main__':main()
