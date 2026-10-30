"""Reuse the matched all-seen audit with explicit uncompressed and cost gates."""
import summarize_all_seen_gate as audit
import json
if __name__=='__main__':
    audit.OUT=audit.ROOT/'reports/rank8_gate_20261030'
    audit.REFERENCE_FOLDERS=dict(audit.REFERENCE_FOLDERS,uncompressed='all_seen_gate_20261030')
    audit.main()
    path=audit.OUT/'THREE_SEED_REPORT.json'
    result=json.loads(path.read_text())
    result['limitations']=[v for v in result['limitations'] if not v.startswith(('Mean forgetting','Original seed1'))]
    result['limitations'].append('This protocol froze the previously amended FP64 identity and FP32 layer-L2 oracle before launch; the recorded seed1 pointwise result is false, but there was no new preflight failure or retry.')
    result['transport_summary']={}
    for seed in range(3):
        q=json.loads((audit.OUT/f'seed{seed}/global/QUALIFICATION_T5.json').read_text())
        records=[v for g in q['training_gate_records'] for v in g['baseline']['rank8_transport']['records']]
        result['transport_summary'][str(seed)]={'retained_energy_fraction':audit.describe([v['retained_energy_fraction'] for v in records]),'relative_reconstruction_error':audit.describe([v['relative_reconstruction_error'] for v in records]),'svd_seconds':q['rank8_SVD_seconds'],'client_layer_svds':len(records),'gradient_upload_bytes':q['logical_communication_bytes']['rank8_gradient_factor_upload']}
    path.write_text(json.dumps(result,indent=2)+'\n')
    report=audit.OUT/'THREE_SEED_REPORT_ZH.txt'
    lines=report.read_text().splitlines()
    lines=[line for line in lines if not line.startswith(('权衡：','seed1原FP32','若本轮失败'))]
    d=result['differences']['control']['raw']['5']
    lines.append(f"相对Control平均遗忘差{d['forgetting_pp']['mean']:+.6f}pp、old-to-new混淆差{d['old_to_new_percent']['mean']:+.6f}pp；seed1 raw仍负，不能称所有指标改善。")
    lines.append('本轮从启动前冻结采用已公开的修订预检oracle；seed1旧式逐元素结果为false，本轮没有新预检失败或重试，实际预检费用均已计入。')
    lines.append('原full-gradient正向结论保留；本轮未通过相对full均值不下降的条件，不补seed、换秩、改角度或目标。')
    report.write_text('\n'.join(lines)+'\n')
