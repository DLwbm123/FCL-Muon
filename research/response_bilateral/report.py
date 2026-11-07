"""Report complete and partial paired groups without choosing favorable seeds."""
import csv
import json
from pathlib import Path
from .runtime import write


def report(root):
    root = Path(root); records = {}; long_records = []
    for path in sorted((root/'reports').glob('train_s*_T*.json')):
        r = json.loads(path.read_text()); key = (r['seed'], r['arm'])
        if r['task'] > 5:
            long_records.append(r)
            continue
        if key not in records or records[key]['task'] < r['task']: records[key] = r
    rows = []
    for (seed, arm), r in sorted(records.items()):
        m = r['stages'][-1]
        row = {'seed': seed, 'arm': arm, 'tasks_completed': r['task'], 'all_seen': 100*m['all_seen'],
               'AIA': 100*r['AIA'], 'old_accuracy': 100*m['old_accuracy'] if m['old_accuracy'] is not None else None,
               'new_accuracy': 100*m['new_accuracy'], 'forgetting': 100*m['forgetting'] if m['forgetting'] is not None else None,
               'old_to_new_error': m['old_to_new_error'], 'new_to_old_error': m['new_to_old_error'],
               'paired_E0_pp': None, 'E3_minus_E2_pp': None, 'E3_minus_E1_pp': None}
        baseline = records.get((seed, 'E0'))
        if baseline and baseline['task'] == r['task']:
            row['paired_E0_pp'] = 100*(m['all_seen']-baseline['stages'][-1]['all_seen'])
        if arm == 'E3':
            for other in ('E1', 'E2'):
                control = records.get((seed, other))
                if control and control['task'] == r['task']:
                    row[f'E3_minus_{other}_pp'] = 100*(m['all_seen']-control['stages'][-1]['all_seen'])
        rows.append(row)
    fields = list(rows[0]) if rows else ['seed', 'arm', 'tasks_completed', 'all_seen']
    with (root/'reports/results.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    complete = [s for s in (3, 4, 5) if all((s, a) in records and records[s, a]['task'] == 5 for a in ('E0', 'E1', 'E2', 'E3'))]
    lines = ['# 双侧近端研究：实际运行结果', '', '当前只报告已完成任务与实际 n；缺失臂不以历史实验代替。', '',
             '| seed | arm | 已完成任务 | 最后准确率 % | AIA % | 对 E0 差 pp |', '|---|---|---|---|---|---|']
    for r in rows:
        lines.append(f"|{r['seed']}|{r['arm']}|{r['tasks_completed']}|{r['all_seen']:.4f}|{r['AIA']:.4f}|{r['paired_E0_pp']}|")
    lines += ['', f'完整 T5 四臂配对组：{complete}，n={len(complete)}。',
        'seed0 为开发窗口；seeds3/4/5 仍属同一 benchmark，不称新盲测试。',
        'AIA 为截至当前已完成任务的 all-seen accuracy 算术平均。遗忘为最后阶段各旧任务准确率相对历次最高值的下降均值。',
        '所有分类使用已见类别整体 argmax，无 task-id；完整任务矩阵和混淆见逐任务私有 JSON。',
        '通信是逻辑字节，不是实测网络流量。复用 T1 的历史成本和本轮新增成本分列；前缀训练只计一次。',
        '近端目标下降与双侧罚项下降不等于旧类准确率改善；现有头部导数为原隐式代理，未宣称全链严格梯度认证。']
    if long_records:
        lines += ['', '## 冻结胜出版本的长序列配对', '', '|seed|arm|任务|总体准确率 %|', '|---|---|---|---|']
        for r in long_records:
            lines.append(f"|{r['seed']}|{r['arm']}|{r['task']}|{100*r['stages'][-1]['all_seen']:.4f}|")
    (root/'reports/RESULTS_ZH.md').write_text('\n'.join(lines)+'\n')
    return records, rows, complete


def decisions(records, complete):
    if complete != [3, 4, 5]: return {'status': 'insufficient_complete_three_seed_groups', 'complete': complete}
    result = {}
    for arm in ('E2', 'E3'):
        ds = [100*(records[s, arm]['stages'][-1]['all_seen']-records[s, 'E0']['stages'][-1]['all_seen']) for s in complete]
        def diff(key): return sum(100*(records[s, arm]['stages'][-1][key]-records[s, 'E0']['stages'][-1][key]) for s in complete)/3
        aia = sum(100*(records[s, arm]['AIA']-records[s, 'E0']['AIA']) for s in complete)/3
        result[arm] = {'mean_difference_pp': sum(ds)/3, 'per_seed_pp': ds,
            'eligible': sum(ds)/3 >= .30 and aia >= 0 and diff('old_accuracy') >= 0
                        and diff('new_accuracy') >= -.20 and sum(x > 0 for x in ds) >= 2 and min(ds) >= -.50}
    d32 = [100*(records[s, 'E3']['stages'][-1]['all_seen']-records[s, 'E2']['stages'][-1]['all_seen']) for s in complete]
    a32 = sum(records[s, 'E3']['AIA']-records[s, 'E2']['AIA'] for s in complete)/3
    result['prefer_response'] = sum(d32)/3 >= .10 and sum(x >= 0 for x in d32) >= 2 and a32 >= 0
    result['winner'] = ('E3' if result['E3']['eligible'] and result['prefer_response'] else
                        'E2' if result['E2']['eligible'] else None)
    return result
