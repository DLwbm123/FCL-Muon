"""Aggregate exactly the two prespecified completed seeds; all metrics retained."""
import json
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/'reports/ten_task_20261026'
SEEDS=[0,1]
METHODS=['regmean','lorm']
KEYS=['final_all_seen','final_old','final_new','mean_new_T2_T10','average_incremental_accuracy','forgetting']


def stat(values):
    assert len(values)==2 and all(v is not None for v in values)
    return {'seed_values':dict(zip(map(str,SEEDS),values)), 'mean':statistics.mean(values),
            'sample_sd':statistics.stdev(values),'n':2,'sd_ddof':1}


def matched(a,b):
    return {'regmean':stat(a),'lorm':stat(b),'paired_head_minus_lorm':stat([x-y for x,y in zip(a,b)])}


def diagnostic(method,seed,rule,t):
    directory=ROOT/f'{method}_seed{seed}'
    tr=json.loads((directory/'TRAJECTORY.json').read_text())
    if method=='lorm':
        d=tr['competition'][t][rule]
        restricted=[v['restricted_accuracy'] for v in d['per_task']]
        stolen=[v['restricted_correct_but_other_task_wins'] for v in d['per_task']]
        group=d['task_group_confusion'];old=d['old_to_current'];new=d['current_to_old']
    else:
        stage=json.loads((directory/'metrics.json').read_text())['stages'][t]
        if rule=='centered':
            d=stage['centered'];restricted=[v['restricted_accuracy'] for v in d['per_task']]
            stolen=[v['restricted_correct_but_other_task_wins'] for v in d['per_task']]
            group=d['task_group_confusion'];old=d['old_to_current'];new=d['current_to_old']
        else:
            restricted=stage['restricted'];stolen=[round(1000*(a-b)) for a,b in zip(restricted,stage['per_task'])]
            cm=stage['confusion'];n=t+1
            group=[[sum(cm[i][j] for i in range(10*a,10*a+10) for j in range(10*b,10*b+10)) for b in range(n)] for a in range(n)]
            old=stage['old_to_new_error'];new=stage['new_to_old_error']
    return {'restricted_accuracy_percent':100*statistics.mean(restricted),
            'restricted_per_task_percent':[100*v for v in restricted],
            'restricted_correct_but_other_task_wins_count':sum(stolen),'stolen_per_task_count':stolen,
            'old_to_current_percent':100*old if old is not None else None,
            'current_to_old_percent':100*new if new is not None else None,'task_group_confusion':group}


def main():
    pairs={s:json.loads((ROOT/f'SEED{s}_COMPARISON.json').read_text()) for s in SEEDS}
    data={m:{s:json.loads((ROOT/f'{m}_seed{s}/TRAJECTORY.json').read_text()) for s in SEEDS} for m in METHODS}
    for m in METHODS:
        for s in SEEDS:
            directory=ROOT/f'{m}_seed{s}'
            assert json.loads((directory/'QUALIFICATION.json').read_text())['qualified']
            assert json.loads((directory/'FINAL_CHECK.json').read_text())['passed']
            for rule in ['raw','centered']:
                matrix=data[m][s][rule]['matrix'];assert len(matrix)==10 and all(len(v)==i+1 for i,v in enumerate(matrix))
                for k in KEYS:
                    assert abs(100*data[m][s][rule][k]-pairs[s]['summary'][rule][m][k])<1e-9
    result={'status':'ALL_FOUR_RUNS_COMPLETE_VERIFIED_TWO_SEED_FINAL','seeds':SEEDS,'methods':METHODS,
            'units':'accuracy percent; forgetting and paired differences percentage points; sample SD ddof1 across two seeds',
            'primary_rule':'Prespecified centered for BOTH methods; retain raw for BOTH',
            'summary':{},'stage_comparison':{},'full_matrices_percent':{},'matrix_cell_statistics':{},
            'competition':{},'cost':{},'qualification':'All four complete500client-round/1.25M visits and actual10000-image final checkpoint reload; no repeated GPU evaluation for reporting.'}
    for rule in ['raw','centered']:
        result['summary'][rule]={k:matched(*[[pairs[s]['summary'][rule][m][k] for s in SEEDS] for m in METHODS]) for k in KEYS}
        result['full_matrices_percent'][rule]={m:{str(s):[[100*v for v in row] for row in data[m][s][rule]['matrix']] for s in SEEDS} for m in METHODS}
        result['matrix_cell_statistics'][rule]=[[matched(*[[100*data[m][s][rule]['matrix'][t][j] for s in SEEDS] for m in METHODS]) for j in range(t+1)] for t in range(10)]
        result['stage_comparison'][rule]=[];result['competition'][rule]=[]
        for t in range(10):
            stage={'task':t+1}
            for key in ['new','old','overall','forgetting']:
                stage[key]=None if t==0 and key in ['old','forgetting'] else matched(*[[pairs[s]['stage_comparison'][rule][t][m][key] for s in SEEDS] for m in METHODS])
            result['stage_comparison'][rule].append(stage)
            ds={m:{str(s):diagnostic(m,s,rule,t) for s in SEEDS} for m in METHODS}
            aggregates={}
            for key in ['restricted_accuracy_percent','restricted_correct_but_other_task_wins_count','old_to_current_percent','current_to_old_percent']:
                values=[[ds[m][str(s)][key] for s in SEEDS] for m in METHODS]
                aggregates[key]=None if any(v is None for group in values for v in group) else matched(*values)
            result['competition'][rule].append({'task':t+1,'per_seed':ds,'aggregate':aggregates})
    result['cost']['per_seed']={str(s):pairs[s]['cost'] for s in SEEDS}
    result['cost']['aggregate']={}
    for m in METHODS:
        result['cost']['aggregate'][m]={}
        for key in pairs[0]['cost'][m]:
            values=[pairs[s]['cost'][m][key] for s in SEEDS]
            result['cost']['aggregate'][m][key]=None if any(v is None for v in values) else {**stat(values),**({'maximum_across_runs':max(values)} if key=='recorded_round_peak_gpu_bytes' else {'total':sum(values)})}
    totals={key:sum(pairs[s]['cost'][m][key] for s in SEEDS for m in METHODS) for key in
            ['training_visits','client_rounds','statistics_forward_images','task_evaluation_images','round_evaluation_images','final_reload_images']}
    assert totals=={'training_visits':5000000,'client_rounds':2000,'statistics_forward_images':1200000,'task_evaluation_images':220000,'round_evaluation_images':550000,'final_reload_images':40000}
    result['cost']['study_totals']=totals
    result['limits']=['Two prespecified seeds only; sample SD is descriptive, not a significance or robustness guarantee.',
        'Shared CIFAR development test set, not independent held-out validation.',
        'Architecture/rank/optimizer/numerics differ; this is a whole-scheme comparison, not a single-mechanism causal ablation.',
        'Both methods use the same frozen centered rule. Raw results must not be omitted.',
        'Task-restricted accuracy uses true task only as an offline diagnostic, not inference and not attainable improvement guaranteed.',
        'Lower forgetting can partly reflect lower initial acquisition; compare absolute old accuracy and all a_tt.',
        'Shared GPUs and unequal round evaluation workloads prevent controlled wall-time efficiency claims.',
        'Head GPU peak and LoRM communication were not instrumented; null means unavailable, not zero.',
        'LoRM seed1 COMPLETE.elapsed uses original shared group clock; cost uses worker receipt elapsed instead.',
        'Historical PROGRESS files and RESULTS_SNAPSHOT are dated snapshots; this file supersedes their pending status.',
        'The newly authorized independent three-task aggregation-utility pilot is separate; this final report does not extend the ten-task scope.']
    (ROOT/'TWO_SEED_FINAL.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['两seed完整十任务最终实验报告','CIFAR100 alpha=0.05；seeds0/1；Head-RegMean与标准LoRM各两条，全部完成并通过最终检查点验收。','每个单元为 mean ± sample SD（n=2，ddof=1）；准确率为%，遗忘和方法差为百分点。差值的SD由同seed配对差计算。','']
    labels={'final_all_seen':'最终总体准确率','final_old':'最终旧类准确率','final_new':'T10新类准确率','mean_new_T2_T10':'T2–T10平均新类准确率','average_incremental_accuracy':'平均增量准确率','forgetting':'遗忘（越低越好）'}
    def fmt(v):return f"{v['mean']:.4f} ± {v['sample_sd']:.4f}"
    for rule in ['centered','raw']:
        lines += [rule+'：指标 | Head-RegMean | LoRM | 配对差Head−LoRM']
        for key in KEYS:
            x=result['summary'][rule][key]
            lines.append(labels[key]+' | '+' | '.join(fmt(x[m]) for m in ['regmean','lorm','paired_head_minus_lorm']))
        lines+=['','每seed：']
        for seed in SEEDS:
            lines.append('seed'+str(seed)+' '+json.dumps(pairs[seed]['summary'][rule],ensure_ascii=False))
        lines+=['','逐阶段：task | 主线overall | LoRMoverall | 主线a_tt | LoRMa_tt']
        for stage in result['stage_comparison'][rule]:
            lines.append(str(stage['task'])+' | '+' | '.join([fmt(stage['overall']['regmean']),fmt(stage['overall']['lorm']),fmt(stage['new']['regmean']),fmt(stage['new']['lorm'])]))
        lines+=['']
    lines+=['结果解释：','两seed去均值总体、平均增量准确率和绝对旧类准确率均支持主线优势；平均新任务学习仍较低，不能只挑T10优势。双方原始推理下主线最终总体和平均新类均落后，收益依赖预固定去均值。','较低遗忘须结合a_tt和旧类绝对准确率，不能单独当作更强保留机制的因果证据。两个seed的重复结果仍不等于独立heldout或统计显著性。','', '最终任务竞争诊断（centered）：']
    for key,value in result['competition']['centered'][-1]['aggregate'].items():
        lines.append(key+': '+json.dumps(value,ensure_ascii=False))
    lines+=['','成本（逐seed详见JSON，训练/统计/评估量均完整披露）：',json.dumps(result['cost'],indent=2,ensure_ascii=False),'','完整双方原始/去均值矩阵、每个矩阵单元的两seed统计、全阶段a_tt/old/new/overall/forgetting及task competition见TWO_SEED_FINAL.json。','']+result['limits']
    (ROOT/'TWO_SEED_FINAL_ZH.txt').write_text('\n'.join(lines)+'\n')
    directory=ROOT/'lorm_seed1';trajectory=data['lorm'][1]
    text=['LoRM seed1完整十任务完成并通过终验。','500 unique client-round、1250000训练访问、79200优化步；实际最终checkpoint在全部10000图raw/centered预测重现，最大logit差0。','单位：准确率%；遗忘为百分点。']
    for rule in ['centered','raw']:
        text.append(rule+': '+json.dumps({k:100*trajectory[rule][k] for k in KEYS},ensure_ascii=False))
    text+=['两seed完整配对见../TWO_SEED_FINAL_ZH.txt。',json.dumps(pairs[1]['cost']['lorm'],indent=2),'不重复GPU评估；最终只读CPU检查点核对通过。']
    (directory/'FINAL_RESULTS_ZH.txt').write_text('\n'.join(text)+'\n')
    print(json.dumps(result['summary'],indent=2))


if __name__=='__main__':main()
