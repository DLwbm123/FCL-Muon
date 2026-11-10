"""Native one-pass frozen encoder statistics; dense and encoded readouts."""
import copy,json,os,time
import torch
from fcl.checkpoint import restore_rng
from research.head_regmean.pilot import observe
from research.adamw_t10.storage import write
from .statistics import local_packet,dense_packet,merge,class_wire
from .runtime import setup,build,validate


def restore_frozen(p):
    x=torch.load(p.out/'latest.pt',map_location='cpu',weights_only=False)
    if x['config']!=p.cfg or x['clock']!=p.clock or x['identity']!=p.identity:raise ValueError('frozen recovery identity changed')
    if not all(x['merged_flags'].values()) or x['head_count']!=0:raise ValueError('frozen backbone modified')
    p.model.load_state_dict(x['model'],strict=True);p.s=x['state'];restore_rng(x['rng'])


def step_frozen(p,t):
    p.s.update(task=t,phase='frozen_statistics',round=0,conditional_readouts={},evaluator_old_anchor=None)
    packets={'dense':[],'compressed':[]};cost={'dense':0,'compressed':0}
    with observe(p.model):
        for k in range(p.cfg['clients']):
            z,y=p._features(k)
            compressed,wire,_=local_packet(z,z,y,p.cfg['cpt'],p.R,p.cfg['head_rank'])
            dense,dense_wire=dense_packet(z,y,p.cfg['cpt'],p.R)
            packets['dense'].append(dense);packets['compressed'].append(compressed)
            cost['dense']+=len(dense_wire);cost['compressed']+=len(class_wire(wire,z.shape[1],p.R.shape[1],p.cfg['cpt'],'identity'))
        p.s['branch_heads']={}
        for mode in ('dense','compressed'):
            old=p.s['frozen_stats'][mode]
            stats,W,beta,_=merge(packets[mode],old,p.R,mode='identity')
            p.s['frozen_stats'][mode]=stats;p.s['branch_heads'][mode]=(W,beta)
            counters=p.s['frozen_communication'][mode]
            counters['class_statistics_upload']=counters.get('class_statistics_upload',0)+cost[mode]
            counters['all_seen_head_download']=counters.get('all_seen_head_download',0)+p.cfg['clients']*(W.numel()+beta.numel())*4
            if int(stats['counts'].sum())!=(t+1)*len(p.ids('train',t)):raise AssertionError('frozen count error')
        p.model.deploy(*p.s['branch_heads']['dense'])
    result=p.evaluate(False);p.s['stages'].append(result);p.s['matrix'].append(result['per_task'])
    p.s['phase']='frozen_evaluated';p.save();restore_frozen(p)
    if p.evaluate(False)!=result:raise ArithmeticError('frozen checkpoint readout mismatch')
    return result


def main():
    setup();root,f=validate();p=build({k:f[k] for k in ('started','deadline')})
    p.cfg['primary_mode']='dense';p.cfg['branch_modes']=['dense','compressed']
    p.s['frozen_stats']={k:None for k in ('dense','compressed')};p.s['frozen_communication']={k:{} for k in ('dense','compressed')}
    if (p.out/'latest.pt').exists():raise RuntimeError('fresh frozen run expected')
    start=time.monotonic()
    for t in range(p.cfg['tasks']):
        result=step_frozen(p,t)
        write(root,root/'src/reports'/f'frozen_T{t+1}.json',dict(task=t+1,branch_readouts=result['branch_readouts'],
            raw_AIA={mode:sum(s['branch_readouts'][mode]['all_seen'] for s in p.s['stages'])/len(p.s['stages']) for mode in ('dense','compressed')},
            communication=p.s['frozen_communication'],reload_verified=True))
        print(json.dumps(dict(task=t+1,dense=result['all_seen'])),flush=True)
    if p.accesses['train_loaded']!=50000 or any(torch.count_nonzero(b) for b in p.model.b_parameters()):raise ValueError('frozen protocol changed')
    p.s['phase']='frozen_complete';p.save();restore_frozen(p)
    complete=dict(status='completed',tasks=10,trainable_updates=0,training_visits=0,
        raw_AIA={mode:sum(s['branch_readouts'][mode]['all_seen'] for s in p.s['stages'])/10 for mode in ('dense','compressed')},
        stages=p.s['stages'],communication=p.s['frozen_communication'],loader_accesses=p.accesses,
        wall_seconds=time.monotonic()-start,all_10_reload_verified=True,final_no_adapter_update=True,
        limits='dense scatter FP64 vs lowrank64 FP32; both class means FP32; cached initial pretrained backbone distribution excluded')
    write(root,root/'src/reports/frozen_COMPLETE.json',complete)


if __name__=='__main__':main()
