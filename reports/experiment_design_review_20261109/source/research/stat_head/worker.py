"""Full fresh T1-T10, per-task readback and paired report; no score gate."""
import copy,json,time,torch
from .runtime import setup,build,validate
from research.adamw_t10.storage import write


def main():
    setup();root,f=validate();p=build({k:f[k] for k in ('started','deadline')});arm=p.cfg['head_protocol']
    reports=root/'src/reports'/arm;reports.mkdir(exist_ok=True)
    if (p.out/'latest.pt').exists():raise RuntimeError('fresh launch expected; audit recovery')
    begin=time.monotonic()
    while p.s['phase']!='complete':
        old=p.s['phase'];p.step()
        if old=='ready_evaluation':
            expected=copy.deepcopy(p.s['stages'][-1]);p.restore()
            if p.evaluate(False)!=expected:raise ArithmeticError('readback mismatch')
            record=dict(status='evaluated_and_reload_verified',task=p.s['task']+1,
                raw_AIA=sum(s['all_seen'] for s in p.s['stages'])/len(p.s['stages']),
                centered_AIA=sum(s['centered']['all_seen'] for s in p.s['stages'])/len(p.s['stages']),
                stages=p.s['stages'],matrix=p.s['matrix'],communication=p.s['communication'],
                route_records=p.s.get('route_records',[]),head_codec_records=p.s.get('head_codec_records',[]),
                client_rounds=len(p.s['training_records']),training_visits=sum(s['visited'] for s in p.s['training_records']),
                training_steps=sum(s['steps'] for s in p.s['training_records']),
                wall_seconds=time.monotonic()-begin,accesses=p.accesses)
            write(root,reports/f'T{record["task"]}.json',record)
            print(json.dumps({k:record[k] for k in ('task','raw_AIA','centered_AIA')}),flush=True)
    if len(p.s['training_records'])!=500 or sum(s['visited'] for s in p.s['training_records'])!=1250000:
        raise ValueError('incomplete formal schedule')
    p.restore();p.s['task']=9;merged=p.evaluate(False);p.s['task']=10
    if merged['confusion']!=p.s['stages'][-1]['confusion'] or merged['centered']['confusion']!=p.s['stages'][-1]['centered']['confusion']:
        raise ArithmeticError('merged terminal confusion mismatch')
    if p.s['merges']!=list(range(10)) or p.s['history'].committed!=set(range(10)):raise ValueError('commit mismatch')
    complete=dict(status='completed',arm=arm,tasks=10,raw_AIA=sum(s['all_seen'] for s in p.s['stages'])/10,
        centered_AIA=sum(s['centered']['all_seen'] for s in p.s['stages'])/10,
        stages=p.s['stages'],matrix=p.s['matrix'],communication=p.s['communication'],
        route_records=p.s.get('route_records',[]),head_codec_records=p.s.get('head_codec_records',[]),
        training_visits=1250000,client_rounds=500,training_steps=sum(s['steps'] for s in p.s['training_records']),
        loader_accesses=p.accesses,wall_seconds=time.monotonic()-begin,all_10_reload_verified=True,
        final_merged_raw_centered_confusion_equal=True,peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    write(root,reports/'COMPLETE.json',complete)


if __name__=='__main__':main()
