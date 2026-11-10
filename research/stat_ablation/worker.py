import copy,json,time,torch
from .runtime import setup,build,validate
from research.adamw_t10.storage import write


def main():
    setup();root,f=validate();p=build({k:f[k] for k in ('started','deadline')});arm=p.cfg['arm']
    reports=root/'src/reports'/arm;reports.mkdir(exist_ok=True)
    if (p.out/'latest.pt').exists():raise RuntimeError('fresh launch expected; audit recovery')
    begin=time.monotonic()
    while p.s['phase']!='complete':
        old=p.s['phase'];p.step()
        if old=='ready_evaluation':
            expected=copy.deepcopy(p.s['stages'][-1]);p.restore()
            if p.evaluate(False)!=expected:raise ArithmeticError('all branch readbacks changed')
            record=dict(status='evaluated_and_reload_verified',task=p.s['task']+1,stages=p.s['stages'],
                branch_raw_AIA={mode:sum(s['branch_readouts'][mode]['all_seen'] for s in p.s['stages'])/len(p.s['stages']) for mode in p.cfg['branch_modes']},
                communication=p.s['communication'],branch_statistics_upload=p.s['branch_statistics_upload'],route_records=p.s.get('route_records',[]),
                client_rounds=len(p.s['training_records']),training_visits=sum(s['visited'] for s in p.s['training_records']),
                training_steps=sum(s['steps'] for s in p.s['training_records']),loader_accesses=p.accesses,wall_seconds=time.monotonic()-begin)
            write(root,reports/f'T{record["task"]}.json',record)
            print(json.dumps(dict(task=record['task'],raw_AIA=record['branch_raw_AIA'])),flush=True)
    if (len(p.s['training_records'])!=500 or sum(s['visited'] for s in p.s['training_records'])!=1250000
        or sum(s['steps'] for s in p.s['training_records'])!=79350 or p.s['merges']!=list(range(10))
        or p.s['history'].committed!=set(range(10))):raise ValueError('formal protocol incomplete')
    p.restore();p.s['task']=9;merged=p.evaluate(False);p.s['task']=10
    expected=copy.deepcopy(p.s['stages'][-1]);validated=copy.deepcopy(merged)
    # BA merge is FP32: continuous evaluator diagnostics can change by tiny
    # roundoff. Official branch confusion/accuracy must still match exactly.
    for value in (expected,validated):
        value.pop('evaluator_only_transfer',None);value.pop('conditional_readouts',None)
    if validated!=expected:raise ArithmeticError('merged branch confusion/readouts changed')
    branches={}
    for mode in p.cfg['branch_modes']:
        cost=dict(p.s['communication']);cost['unified_statistics_upload']=p.s['branch_statistics_upload'][mode]
        branches[mode]=dict(raw_AIA=sum(s['branch_readouts'][mode]['all_seen'] for s in p.s['stages'])/10,
            centered_AIA=sum(s['branch_readouts'][mode]['centered']['all_seen'] for s in p.s['stages'])/10,
            final=p.s['stages'][-1]['branch_readouts'][mode],standalone_formal_communication=cost)
    write(root,reports/'COMPLETE.json',dict(status='completed',arm=arm,tasks=10,branches=branches,
        stages=p.s['stages'],route_records=p.s.get('route_records',[]),client_rounds=500,training_visits=1250000,
        training_steps=79350,shared_research_communication=p.s['communication'],loader_accesses=p.accesses,
        wall_seconds=time.monotonic()-begin,all_10_branch_reload_verified=True,merged_branch_readouts_equal=True,terminal_validation_scope='all official branch raw/centered confusion and derived metrics; conditional/evaluator diagnostics retained at premerge endpoint',
        limits='shared training/evaluation cost charged once as research cost; each algorithm standalone formal payloads listed separately; selector and evaluator diagnostics distinct',
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved()))


if __name__=='__main__':main()
