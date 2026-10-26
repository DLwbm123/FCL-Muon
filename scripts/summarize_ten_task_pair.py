"""Report a fully qualified same-seed pair; never select checkpoints or inference rules."""
import json,os
from pathlib import Path


def summarize(root,seed):
    arms={}
    for method in ['regmean','lorm']:
        out=root/f'{method}_seed{seed}'
        check=json.loads((out/'FINAL_CHECK.json').read_text());q=json.loads((out/'QUALIFICATION.json').read_text())
        assert check['passed'] and q['qualified']
        arms[method]={'trajectory':json.loads((out/'TRAJECTORY.json').read_text()),'qualification':q,'final_check':check}
    keys=['final_all_seen','final_old','final_new','mean_new_T2_T10','average_incremental_accuracy','forgetting']
    result={'seed':seed,'status':'FULL_SINGLE_SEED_PAIR_COMPLETE','primary':'Both methods use fixed centered inference; raw retained for both','summary':{},'stage_comparison':{},'final_per_task':{},'cost':{}}
    for mode in ['raw','centered']:
        a,b=[arms[m]['trajectory'][mode] for m in ['regmean','lorm']]
        result['summary'][mode]={m:{k:100*arms[m]['trajectory'][mode][k] for k in keys} for m in arms}
        result['summary'][mode]['head_minus_lorm_pp']={k:100*(a[k]-b[k]) for k in keys}
        result['stage_comparison'][mode]=[]
        for t in range(10):
            row={'task':t+1}
            for method in arms:
                mat=arms[method]['trajectory'][mode]['matrix'];v=mat[t];assert len(v)==t+1
                row[method]={'new':100*v[t],'old':100*sum(v[:-1])/t if t else None,'overall':100*sum(v)/(t+1),'forgetting':100*sum(max(mat[j][i] for j in range(i,t+1))-v[i] for i in range(t))/t if t else None}
            result['stage_comparison'][mode].append(row)
        result['final_per_task'][mode]={'regmean':[100*v for v in a['matrix'][-1]],'lorm':[100*v for v in b['matrix'][-1]],'head_minus_lorm_pp':[100*(v-w) for v,w in zip(a['matrix'][-1],b['matrix'][-1])]}
    for method,data in arms.items():
        q=data['qualification'];c=data['final_check']
        result['cost'][method]={'training_visits':q['training_visits'],'optimizer_steps':q['optimizer_steps'],'client_rounds':q['client_rounds'],'statistics_forward_images':q.get('stats_visits',q.get('statistics_image_forwards')),'task_evaluation_images':q['task_evaluation_images'],'round_evaluation_images':q.get('round_evaluation_images',0),'final_reload_images':q['audit_images'],'worker_wall_seconds':c['worker_elapsed_seconds'],'checkpoint_bytes':c['checkpoint_bytes'],'run_output_allocated_bytes':c['run_output_allocated_bytes'],'logical_communication_bytes':c.get('logical_communication_bytes_total'),'recorded_round_peak_gpu_bytes':c.get('max_recorded_round_peak_gpu_bytes')}
    result['limits']=['This report describes one seed pair; consult TWO_SEED_FINAL.json for aggregation across both prespecified seeds.','Shared development test set, no independent held-out confirmation.','Different architectures/ranks/optimizers/numerics; not a single-mechanism causal ablation.','Wall time includes different GPU workloads and contention; not a controlled throughput benchmark.','Head GPU peak and LoRM logical communication were not directly recorded; null means unavailable, not zero.','Lower forgetting alone can reflect lower initial task acquisition; jointly report absolute old accuracy and a_tt.']
    (root/f'SEED{seed}_COMPARISON.json').write_text(json.dumps(result,indent=2)+'\n')
    rows=['完整十任务单seed配对：Head-RegMean vs 标准LoRM（seed'+str(seed)+'）','双方完成500clientround/1.25M训练、实际最终checkpoint终验；主线50部署证书和完整raw/centered重载通过。','本文件为该seed的完整配对；两seed统计见TWO_SEED_FINAL.json。','']
    for mode in ['centered','raw']:
        rows.append(mode+'：指标 | 主线 | LoRM | 主线减LoRM（百分点）')
        for k in keys:rows.append(f"{k}: {result['summary'][mode]['regmean'][k]:.4f} | {result['summary'][mode]['lorm'][k]:.4f} | {result['summary'][mode]['head_minus_lorm_pp'][k]:+.4f}")
        rows.append('')
    rows+=['判读：固定去均值是预先规定的主线推理规则，必须与双方raw一起报告。终点当前新任务优势不代表整个序列的新类学习都更好；用T2–T10 a_tt均值和逐任务值共同判断。遗忘更低也不单独证明旧类绝对准确率更高。','所有seed按预先冻结协议报告，不改参数、筛seed或挑checkpoint。','',json.dumps(result['cost'],indent=2,ensure_ascii=False),'']+result['limits']
    (root/f'SEED{seed}_COMPARISON_ZH.txt').write_text('\n'.join(rows)+'\n')
    return result


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]/'reports/ten_task_20261026'
    result=summarize(root,int(os.environ.get('REPORT_SEED','0')))
    print(json.dumps(result['summary'],indent=2))
