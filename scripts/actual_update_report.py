"""Report the frozen paired pilot using saved evaluations; never train or tune."""
import json
import os
from pathlib import Path
import numpy as np
from research.basic_repair.fcl_basic_repair_tools import center_task_logits
from research.fed_spectra.lifecycle import write


def stage_metrics(stages,view):
    matrix=[];rows=[]
    for i,s in enumerate(stages):
        assert s['task']==i
        centered=s['centered'];counts=[p['n'] for p in centered['per_task']]
        n=sum(counts);c=len(s['confusion'])//(i+1);cm=np.asarray(s['confusion'])
        assert cm.shape==(c*(i+1),c*(i+1)) and int(cm.sum())==n
        if view=='raw':
            row=s['per_task'];restricted=s['restricted'];overall=s['all_seen']
            old=s['old_accuracy'];new=s['new_accuracy']
            groups=cm.reshape(i+1,c,i+1,c).sum(axis=(1,3))
            assert abs(cm.trace()/n-overall)<1e-12
        else:
            row=[p['all_seen_accuracy'] for p in centered['per_task']]
            restricted=[p['restricted_accuracy'] for p in centered['per_task']]
            overall=centered['all_seen_accuracy'];old=centered['old_all_seen_accuracy'];new=centered['current_all_seen_accuracy']
            groups=np.asarray(centered['task_group_confusion'])
        assert len(row)==i+1 and abs(sum(a*b for a,b in zip(row,counts))/n-overall)<1e-12
        matrix.append(row)
        signed=sum(max(matrix[j][k] for j in range(k,i))-row[k] for k in range(i))/i if i else None
        forgetting=sum(max(matrix[j][k] for j in range(k,i+1))-row[k] for k in range(i))/i if i else None
        within=sum(round((1-r)*count) for r,count in zip(restricted,counts))
        stolen=sum(round((r-a)*count) for r,a,count in zip(restricted,row,counts))
        errors=round((1-overall)*n);assert within+stolen==errors
        rows.append({'task':i+1,'images':n,'overall_percent':100*overall,
            'old_percent':100*old if old is not None else None,'new_percent':100*new,
            'AIA_percent':sum(r['overall_percent'] for r in rows+[{'overall_percent':100*overall}])/(i+1),
            'forgetting_pp':100*forgetting if forgetting is not None else None,
            'signed_forgetting_pp':100*signed if signed is not None else None,
            'W':within,'S':stolen,'errors':errors,'task_group_confusion':groups.tolist(),
            'old_to_new_percent':100*float(groups[:-1,-1].sum())/sum(counts[:-1]) if i else None,
            'new_to_old_percent':100*float(groups[-1,:-1].sum())/counts[-1] if i else None,
            'per_task_percent':[100*v for v in row]})
    return rows


def paired(reference,candidate,cpt):
    with np.load(reference) as f:a={k:f[k] for k in ('indices','labels','logits')}
    with np.load(candidate) as f:b={k:f[k] for k in ('indices','labels','logits')}
    assert np.array_equal(a['indices'],b['indices']) and np.array_equal(a['labels'],b['labels'])
    assert a['logits'].shape==b['logits'].shape
    result={}
    for view in ['raw','centered']:
        z1,z2=a['logits'],b['logits']
        if view=='centered':z1,z2=center_task_logits(z1,cpt),center_task_logits(z2,cpt)
        aa=z1.argmax(1)==a['labels'];bb=z2.argmax(1)==b['labels']
        corrected=int((~aa&bb).sum());destroyed=int((aa&~bb).sum())
        assert corrected-destroyed==int(bb.sum())-int(aa.sum())
        result[view]={'images':len(aa),'corrected':corrected,'destroyed':destroyed,
            'net_corrected':corrected-destroyed,'accuracy_delta_pp':100*(corrected-destroyed)/len(aa)}
    return result


def validate_comparison_configs(reference,candidate):
    a=json.loads(json.dumps(reference));b=json.loads(json.dumps(candidate))
    for c in [a,b]:c.pop('output');c['actual_update']['operator'].pop('eta')
    assert sum(bool(b.get(k)) for k in ['direction_selection','spectral_tangent','matrix_tangent'])<=1
    gated=bool(b.get('training_gate'))
    if b.get('rank8_transport'):
        assert b.get('all_seen_gate') and b.pop('rank8_transport')=={'rank':8,'wire_dtype':'float32'}
    if b.get('all_seen_gate'):
        assert b.get('matrix_tangent') and gated
        assert b.pop('all_seen_gate')=={'current_loss_tolerance':1e-6}
    if b.get('direction_selection'):
        assert b.pop('direction_selection')=={'type':'gradient_reflection','tie_tolerance':1e-12}
    if b.get('spectral_tangent'):
        assert b.pop('spectral_tangent')=={'reference':'original_eta_angle','mode_gradient':'client_local_fp64'}
    if b.get('matrix_tangent'):
        assert b.pop('matrix_tangent')=={'reference':'original_eta_angle','gradient':'client_fp32_server_fp64'}
    if gated:
        assert b.pop('training_gate')=={'minimum_improvement':1e-6,'probe_per_client':32,'selection_seed_base':61029}
        for k in ['protocol_version','maximum_seconds','storage_limit_bytes','write_stop_bytes']:
            a.pop(k);b.pop(k)
    assert a==b,'Unmatched scientific training configuration'
    return gated


def main():
    root=Path(os.environ['PROJECT_ROOT']);report=root/'reports';arms={};records={}
    for arm in ['control','global']:
        q=json.loads((root/'runs'/arm/'QUALIFICATION.json').read_text())
        assert q['qualified'] and q['tasks']==5 and q['client_rounds']==250 and q['training_visits']==625000
        cfg=q['config'];cfg_copy=json.loads(json.dumps(cfg));cfg_copy.pop('output');cfg_copy['actual_update']['operator'].pop('eta')
        if arm=='control':frozen=cfg_copy
        else:validate_comparison_configs(records['control']['config'],cfg)
        records[arm]=q
        rr=[v for r in q['actual_update_rounds'] for layer in r['layers'].values() for v in layer]
        measured=[r for r in rr if 'singular_values_before' in r]
        arms[arm]={view:stage_metrics(q['stages'],view) for view in ['raw','centered']}
        arms[arm]['mechanism']={'eta':cfg['actual_update']['operator']['eta'],'rounds':len(q['actual_update_rounds']),
            'layer_QV_records':len(rr),'identity_records':sum(r['identity_reason'] is not None for r in rr),
            'mean_relative_correction':float(np.mean([r['actual_FP32_relative_correction'] for r in rr])),
            'maximum_deployed_norm_relative_error':max(r['actual_FP32_norm_relative_error'] for r in rr),
            'mean_effective_energy_rank_before':float(np.mean([r['effective_energy_rank_before'] for r in measured])) if measured else None,
            'mean_effective_energy_rank_after':float(np.mean([r['effective_energy_rank_after'] for r in measured])) if measured else None,
            'singular_scale_min':min(r['singular_scale_min'] for r in measured) if measured else None,
            'singular_scale_max':max(r['singular_scale_max'] for r in measured) if measured else None,
            'minimum_cosine_with_fedavg':min(r['cosine_with_fedavg'] for r in measured) if measured else None,
            'operator_seconds':sum(r['seconds'] for r in q['actual_update_rounds']),
            'clipping':'not used','Bures_fallback':'not used'}
        arms[arm]['cost']={k:q[k] for k in ['client_rounds','training_visits','stats_visits','anchor_forward_images',
            'extra_probe_forward_images','extra_probe_backward_images','logical_communication_bytes','peak_gpu_allocated_bytes','elapsed_seconds']}
    gated=bool(records['global']['config'].get('training_gate'))
    if gated:
        q=records['global'];gates=q['training_gate_records']
        arms['global']['training_gate']={'accepted_rounds':q['gate_accepted_rounds'],'rounds':len(gates),
            'decisions':[{'task':v['task'],'round':v['round'],'accepted':v['accepted'],'loss_difference':v['loss_difference']} for v in gates],
            'student_forward_images':q['gate_student_forward_images'],'teacher_forward_images':q['gate_teacher_forward_images']}
        if q['config'].get('direction_selection'):
            arms['global']['training_gate'].update(backward_images=q['gate_backward_images'],negative_direction_rounds=q['negative_direction_rounds'],
              direction_choices=[v['direction']['selected'] for v in gates])
        if q['config'].get('spectral_tangent'):
            arms['global']['training_gate'].update(backward_images=q['gate_backward_images'],tangent_capped_layers=q['tangent_capped_layers'])
        if q['config'].get('matrix_tangent'):
            arms['global']['training_gate'].update(backward_images=q['gate_backward_images'],matrix_tangent=True)
        if q['config'].get('all_seen_gate'):
            arms['global']['training_gate'].update(objective='all_seen_ce_plus_old_kl',current_guard_rejected_rounds=q['gate_current_guard_rejected_rounds'],
                original_loss_differences=[v['all_seen_objective']['current_loss_difference'] for v in gates])
    a,b=arms['control']['raw'][-1],arms['global']['raw'][-1]
    differences={k:b[k]-a[k] for k in ['overall_percent','old_percent','new_percent','AIA_percent','forgetting_pp','S']}
    gate=frozen['scientific_gate']
    conditions={'T5_raw_gain':differences['overall_percent']+1e-10>=gate['raw_gain_min_pp'],
        'raw_AIA':differences['AIA_percent']>=-1e-10,'raw_old':differences['old_percent']>=-1e-10,
        'raw_new':differences['new_percent']+1e-10>=gate['raw_new_delta_min_pp'],
        'cross_task_stolen':differences['S']<=gate['cross_task_stolen_delta_max'],
        'both_reload_qualified':all(q['reload_raw_predictions_equal'] and q['reload_centered_predictions_equal'] for q in records.values())}
    pairs={str(t):paired(root/'runs/control'/f'task{t}_private_logits.npz',root/'runs/global'/f'task{t}_private_logits.npz',cfg['cpt']) for t in range(1,6)}
    for t in range(1,6):
        for view in ['raw','centered']:
            expected=arms['global'][view][t-1]['overall_percent']-arms['control'][view][t-1]['overall_percent']
            assert abs(expected-pairs[str(t)][view]['accuracy_delta_pp'])<1e-10
    result={'status':'COMPLETED_QUALIFIED','seed':cfg['seed'],'hypothesis_supported_in_this_pilot':all(conditions.values()),
        'scientific_conditions':conditions,'frozen_gate':gate,'T5_raw_differences_pp_except_S_count':differences,
        'arms':arms,'paired_vs_control':pairs,'comparison_control_reused':gated,
        'validation':{('same_scientific_config_except_gate_eta' if gated else 'same_configs_except_eta_output'):True,
        'matrix_and_W_plus_S':True,'paired_counts_match_accuracy':True},
        'limitations':[f"single seed{cfg['seed']} development pilot; no statistical significance",
            'T5 endpoint; no T10 claim','current-input KL is not old-task risk guarantee',
            'norm is preserved, but vectorized direction changes under nonuniform spectral scaling',
            'no causal attribution from spectral proxy improvements alone'],
        'source_revision':json.loads((report/'SOURCE.json').read_text())['commit']}
    if gated:result['limitations'].append('Control is the qualified existing same-seed run; gate uses extra current-training probes and logical communication, not held-out data.')
    if records['global']['config'].get('direction_selection'):
        result['limitations'].append('Direction selected by training gradient, then loss-gated; extra backward cost included; reflection is not original spectral interpolation.')
    if records['global']['config'].get('spectral_tangent'):
        result['limitations'].append('Loss-directed singular-value tangent with fixed reference angle ceiling; client-local mode gradients, basis broadcasts, and backward costs included.')
    if records['global']['config'].get('matrix_tangent') and not records['global']['config'].get('rank8_transport'):
        result['limitations'].append('Full matrix norm-sphere tangent; reference angle unchanged; full client FP32 gradients accumulated in FP64, charged and cleared. No spectral support guarantee.')
    if records['global']['config'].get('all_seen_gate'):
        result['limitations'].append('Candidate and gate use all-seen CE plus old KL with a current-loss nonworsening guard. Client optimization remains unchanged. Three scalar losses per point are charged.')
    if records['global']['config'].get('rank8_transport'):
        result['limitations'].append('Client gradients use fixed rank8 FP32 factors and FP64 reconstruction; logical factor payload and local SVD cost are recorded, not network measurements.')
        arms['global']['training_gate']['rank8_transport']={'rank':8,'client_SVDs':sum(len(g['baseline']['rank8_transport']['records']) for g in gates),'SVD_seconds':sum(g['baseline']['rank8_transport']['seconds'] for g in gates)}
    write(report/'FINAL_REPORT.json',result)
    lines=['实际更新全局谱修正：五任务配对实验结果',
        '结论：'+('满足预先固定的单seed初步收益条件。' if result['hypothesis_supported_in_this_pilot'] else '未满足预先固定的初步收益条件；保留全部负结果与权衡，不修改阈值。'),
        ('Control eta=0；新方案为eta=.1谱候选加训练目标接受门。原训练配方不变，新增probe前向和逻辑通信，复用历史对照。' if gated else 'Control eta=0；新方案eta=.1。RRS、当前任务CE+旧类KL、AdamW、Head-RegMean和数据访问相同，仅服务器增量谱形修正不同。'),
        '单位：精度/AIA为%，遗忘为pp。raw主指标，centered仅诊断。',
        '任务 | 路线 | 视图 | overall | old | new | AIA | forgetting | W | S']
    if records['global']['config'].get('all_seen_gate'):
        lines[0]='全已见类目标与原目标保护门：五任务配对结果'
        lines[2]='Control eta=0；候选为原eta.1角度的全矩阵切向。全已见类CE+旧KL改善且原CE_cur+KL不恶化才接受，正式client训练不变。'
    if records['global']['config'].get('rank8_transport'):
        lines[0]='固定rank8梯度传输与全目标保护门：五任务配对结果'
        lines[2]+=' 客户端梯度固定rank8 FP32因子传输，server FP64重构；其余正式训练不变。'
    for t in range(5):
        for arm in arms:
            for view in ['raw','centered']:
                row=arms[arm][view][t]
                vals=[f'{row[k]:.4f}' if row[k] is not None else 'NA' for k in ['overall_percent','old_percent','new_percent','AIA_percent','forgetting_pp']]
                lines.append(f"T{t+1} | {arm} | {view} | "+' | '.join(vals)+f" | {row['W']} | {row['S']}")
    lines+=['','T5 raw新方案减Control：'+json.dumps(differences,ensure_ascii=False),
        '预先固定条件：'+json.dumps(conditions,ensure_ascii=False),
        'T5同图纠正/破坏：'+json.dumps(pairs['5'],ensure_ascii=False),
        '遗忘主表最大值包含当前阶段，与评测代码一致；排除当前阶段的有符号版本另存JSON。',
        '两路线均250客户端轮/625000训练访问，T3/T5实际checkpoint重载raw和centered预测一致；没有按T3分数选择eta或省略T5。']
    for arm in arms:
        lines+=['',arm+'机制：'+json.dumps(arms[arm]['mechanism'],ensure_ascii=False),arm+'成本：'+json.dumps(arms[arm]['cost'],ensure_ascii=False)]
    lines+=['','本报告不证明普遍优于LoRM、不作统计显著性或T10结论。不因Bures/有效秩等代理变化宣称性能收益。',
        '完整分阶段新旧类、方向混淆、矩阵与同图净变化见FINAL_REPORT.json；逐层谱细节见QUALIFICATION_T5.json。',
        '权重、teacher、逐图logits/ID及完整运行日志保留私有；公开聚合报告、配置、源码和验收摘要。']
    if gated:lines+=['训练目标门：仅同probe CE+KL改善>1e-6时接受eta=.1候选，否则精确原FedAvg。对照复用已合格同seed五任务运行；新方案有额外probe/逻辑通信开销，不是零成本比较。',json.dumps(arms['global']['training_gate'],ensure_ascii=False)]
    if records['global']['config'].get('direction_selection'):
        lines+=['方向版本：训练梯度在原候选与等范数反射方向间固定选择，再应用原训练接受门；新增反向成本，负向不是原谱插值，不能将原候选统计当实际部署。']
    if records['global']['config'].get('spectral_tangent'):
        lines+=['谱切向版本：客户端训练梯度先投影到公共SVD模式，再FP64聚合；按梯度切向调整奇异值，参考角度不超过eta.1候选并解析限制非负，随后沿用训练接受门。成本包含SVD基下发/模式向量上传和额外反向。']
    if records['global']['config'].get('matrix_tangent'):
        lines+=['全矩阵切向版本：每client本地FP32完整B梯度作为模拟消息，server按原权重FP64累加；范数球切向在原参考角内旋转，允许改变奇异向量/谱，无原支持保持声明。通信包含完整梯度上传，梯度不落盘，训练门不保证旧类安全。']
    (report/'FINAL_REPORT_ZH.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'report_ready':True,'scientific_conditions':conditions}))


if __name__=='__main__':main()
