"""Render all prespecified pilot arms without selecting results or tuning."""
import json
import os
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    output = Path(os.environ.get('REPORT_DIR', root / 'reports/boundary_memory_20261026'))
    result = json.loads((output / 'RESULTS.json').read_text())
    assert result['qualification']['qualified'] and len(result['stages']) == 3
    labels = {
        'raw': '原主线 raw', 'centered': '原主线 fixed centered',
        'equal_task_support': '任务内 log-softmax／等任务支持',
        'reference_lda': '冻结参考 LDA（固定收缩0.1）',
        'evidence_alone': '双向边界证据单独分类',
        'boundary_fused': '双向边界证据＋原主线条件概率',
        'no_old_to_new_fused': '移除旧类分布→新类约束',
    }
    lines = [
        '完整参考统计＋双向边界证据：3任务验证',
        'CIFAR100 alpha=0.05，seed0，固定前3任务；单seed开发证据。',
        '来源：https://chatgpt.com/share/6adefa60-1a7c-83e8-aa39-f2ebdc1f76cd',
        '复用历史T1–T3已保存主线输出；新增冻结参考分支，不重训MuLoRA，不使用未来T10 checkpoint。',
        '',
        '单位：准确率%；遗忘为百分点。',
        '方案 | 最终总体 | 最终旧类 | 最终新类 | T2–T3平均新类 | AIA | 遗忘',
    ]
    for arm, label in labels.items():
        s = result['summary'][arm]
        values = [s[k] * 100 for k in ('final_overall', 'final_old', 'final_new', 'mean_new_T2_T3',
                                      'average_incremental_accuracy', 'forgetting')]
        lines.append(label + ' | ' + ' | '.join(f'{v:.4f}' for v in values))
    lines.extend(['', '逐阶段完整矩阵：'])
    for arm, label in labels.items():
        lines.append(label)
        for stage, row in enumerate(result['summary'][arm]['matrix'], 1):
            lines.append(f'  T{stage}: ' + ', '.join(f'{v * 100:.2f}' for v in row))
    d = result['decision']
    lines.extend(['', f'预注册开发门槛：{"PASS" if d["gate_passed"] else "FAIL"}'])
    for name, passed in d['conditions'].items():
        lines.append(f'  {name}: {passed}')
    lines.extend([
        f'融合终点相对LDA：{d["fused_minus_reference_lda_final_pp"]:+.4f}pp',
        f'融合终点相对证据单独分类：{d["fused_minus_evidence_alone_final_pp"]:+.4f}pp',
        f'双向约束相对移除旧→新约束：{d["fused_minus_no_old_to_new_final_pp"]:+.4f}pp',
        '', '跨任务竞争与优化诊断：',
    ])
    for stage in result['stages']:
        lines.append(f'T{stage["task"] + 1}')
        for arm in ('centered', 'boundary_fused', 'no_old_to_new_fused'):
            a = stage['arms'][arm]
            stolen = sum(v['restricted_correct_but_other_task_wins'] for v in a['per_task'])
            lines.append(f'  {labels[arm]}：跨任务夺走正确样本{stolen}；旧→新={a["old_to_current"]}；新→旧={a["current_to_old"]}')
        for name, fit in stage['solver'].items():
            lines.append(f'  {name}: iterations={fit["iterations"]}, converged={fit["converged"]}, objective={fit["objective"]:.8g}, gradient_mapping={fit["gradient_mapping_norm"]:.8g}, max_norm={fit["max_row_norm"]:.6g}')
    lines.extend(['', '新增成本与验收：', json.dumps(result['cost'], indent=2, ensure_ascii=False),
                  json.dumps(result['qualification'], indent=2, ensure_ascii=False), '',
                  '限制：不代表独立留出验证，不代表多seed确认；3任务不能验证T7/T8或十任务结论。',
                  '完整协方差版本尚未验证主动查询、压缩通信或未来方向的误差归档。',
                  '证据单独分类在T1没有跨任务约束，零初始化保持为零，因此T1为10%；其AIA不能作为强参考分类器对照。标准全类别LDA才承担参考分类器诊断。',
                  '删除旧→新约束可能导致任务偏置失去约束；该消融的崩溃只说明当前目标需要双向约束，不单独证明原创性或优于其他校准机制。',
                  '参考统计目标及超参数运行前固定；没有测试集拟合、超参搜索、checkpoint或seed选择。',
                  '实际部署需要冻结参考与适配模型两个前向；离线复用logits不等于部署零开销。',
                  '求解器若未达到预设驻点阈值，必须保留该限制，不能把有限迭代称为精确最优。',
                  '本轮仅按已授权3任务交付，不自动扩展主动查询或十任务。'])
    (output / 'FINAL_RESULTS_ZH.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines[:16]))


if __name__ == '__main__':
    main()
