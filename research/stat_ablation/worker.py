import copy,json,math,os,time,torch
from .runtime import setup,build,validate
from research.adamw_t10.storage import write


def main():
    setup();root,f=validate();p=build({k:f[k] for k in ('started','deadline')});arm=p.cfg['arm']
    reports=root/'src/reports'/arm;reports.mkdir(exist_ok=True)
    if (p.out/'latest.pt').exists():
        if os.environ.get('RESUME_WORKER')!='1':raise RuntimeError('fresh launch expected; audit recovery')
        p.restore()
    begin=time.monotonic()
    while p.s['phase']!='complete':
        old=p.s['phase'];p.step()
        stage=p.s['task']
        if old=='committed' and stage in f.get('retain_stage_checkpoints',[]):
            path=p.out/f'T{stage}.pt'
            if not path.exists():os.link(p.out/'latest.pt',path)
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
    tasks=p.cfg['tasks'];rounds=p.cfg['rounds'];epochs=p.cfg['epochs'];batch=p.cfg['batch']
    groups=[group for task in p.manifest['partitions']['train'] for group in task if group]
    client_rounds=rounds*len(groups)
    training_visits=rounds*epochs*sum(map(len,groups))
    training_steps=rounds*epochs*sum(math.ceil(len(group)/batch) for group in groups)
    if (len(p.s['training_records'])!=client_rounds
        or sum(s['visited'] for s in p.s['training_records'])!=training_visits
        or sum(s['steps'] for s in p.s['training_records'])!=training_steps
        or p.s['merges']!=list(range(tasks)) or p.s['history'].committed!=set(range(tasks))):
        raise ValueError('formal protocol incomplete')
    p.restore();p.s['task']=tasks-1;merged=p.evaluate(False);p.s['task']=tasks
    expected=copy.deepcopy(p.s['stages'][-1]);validated=copy.deepcopy(merged)
    # BA merge is FP32: continuous evaluator diagnostics can change by tiny
    # roundoff. Official branch confusion/accuracy must still match exactly.
    for value in (expected,validated):
        value.pop('evaluator_only_transfer',None);value.pop('conditional_readouts',None)
    if validated!=expected:raise ArithmeticError('merged branch confusion/readouts changed')
    branches={}
    for mode in p.cfg['branch_modes']:
        cost=dict(p.s['communication']);cost['unified_statistics_upload']=p.s['branch_statistics_upload'][mode]
        branches[mode]=dict(raw_AIA=sum(s['branch_readouts'][mode]['all_seen'] for s in p.s['stages'])/tasks,
            centered_AIA=sum(s['branch_readouts'][mode]['centered']['all_seen'] for s in p.s['stages'])/tasks,
            final=p.s['stages'][-1]['branch_readouts'][mode],standalone_formal_communication=cost)
    write(root,reports/'COMPLETE.json',dict(status='completed',arm=arm,tasks=tasks,branches=branches,
        stages=p.s['stages'],route_records=p.s.get('route_records',[]),client_rounds=client_rounds,training_visits=training_visits,
        training_steps=training_steps,shared_research_communication=p.s['communication'],loader_accesses=p.accesses,
        wall_seconds=time.monotonic()-begin,all_10_branch_reload_verified=True,merged_branch_readouts_equal=True,terminal_validation_scope='all official branch raw/centered confusion and derived metrics; conditional/evaluator diagnostics retained at premerge endpoint',
        limits='shared training/evaluation cost charged once as research cost; each algorithm standalone formal payloads listed separately; selector and evaluator diagnostics distinct',
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved()))


if __name__=='__main__':main()
