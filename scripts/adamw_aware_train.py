"""Independent AdamW-aware basis hypothesis from an audited shared T1."""
import copy, json, os, time
from pathlib import Path
import numpy as np
import torch, timm
from safetensors.torch import load_file
from torchvision.datasets import CIFAR100
from fcl.data import loader, transforms
from fcl.model import MuLoRAFL
from research.adamw_combined.pilot import CombinedPilot
from research.aggregation_utility.lifecycle import import_prefix


def write(path, value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def main():
    root=Path(os.environ['PROJECT_ROOT']);data=Path(os.environ['SOURCE_ROOT']);old=Path(os.environ['PRIOR_ROOT'])
    job=json.loads((root/'reports/job.json').read_text());clock={k:job[k] for k in ('started','deadline')}
    oldjob=json.loads((old/'reports/job.json').read_text());oldclock={k:oldjob[k] for k in ('started','deadline')}
    prefix=torch.load(old/'runs/prefix/latest.pt',map_location='cpu',weights_only=False)
    baseline=torch.load(old/'runs/rrs_server/latest.pt',map_location='cpu',weights_only=False)
    results=json.loads((old/'reports/ARM_SUMMARY_PRIVATE.json').read_text())
    s=prefix['state'];b=baseline['state']
    if (prefix['clock']!=oldclock or baseline['clock']!=oldclock or prefix['identity']!=baseline['identity'] or
        (s['task'],s['round'],s['phase'])!=(1,0,'task_start') or s['history'].committed!={0} or
        s['merges']!=[0] or len(s['training_records'])!=50 or sum(v['visited'] for v in s['training_records'])!=125000 or
        prefix['head_count']!=1 or not all(prefix['merged_flags'].values()) or s['pending'] or
        (b['task'],b['phase'])!=(3,'complete') or len(b['training_records'])!=150 or
        sum(v['visited'] for v in b['training_records'])!=375000 or len(b['combined_server_records'])!=10 or
        baseline['config']['selection_mode']!='rrs' or not baseline['config']['server_correction'] or
        b.get('prefix_provenance',{}).get('same_clock') is not True or
        any(not v for k,v in prefix['rng'].items() if k=='cuda')):
        raise ValueError('invalid previously completed shared source or control')
    differences={k for k in prefix['config'] if prefix['config'].get(k)!=baseline['config'].get(k)}
    if differences!={'output','selection_mode','server_correction'}:raise ValueError('source config mismatch')
    cfg=copy.deepcopy(prefix['config']);cfg.update(output='runs/main',selection_mode='adamw',server_correction=True)
    for k,v in {'seed':0,'dataset':'seq-cifar100','alpha':.05,'clients':10,'tasks':3,'rounds':5,
                'epochs':5,'batch':16,'rank':32,'lr':.0002,'aggregation':'regmean'}.items():
        if cfg.get(k)!=v:raise ValueError('source scientific config mismatch')
    if any(torch.count_nonzero(x) for k,x in prefix['model'].items() if k.endswith(('.Bq','.Bv'))):
        raise ValueError('source merged adapter not zero')
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True);torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    manifest=json.loads((data/cfg['manifest']).read_text());weights=json.loads((data/'weights.json').read_text())
    if manifest['identity']!={k:cfg[k] for k in ('dataset','seed','alpha','clients')} or prefix['identity']['weight_identity']!=weights['sha256']:
        raise ValueError('source manifest or weights mismatch')
    raw=[CIFAR100(data/'data',train=x,download=False) for x in (True,False)]
    if [list(x.targets) for x in raw]!=[manifest['labels']['train'],manifest['labels']['test']]:raise ValueError('dataset mismatch')
    backbone=timm.create_model(cfg['backbone'],pretrained=False,num_classes=0)
    backbone.load_state_dict({k:v for k,v in load_file(weights['path']).items() if not k.startswith('head.')},strict=True)
    model=MuLoRAFL(backbone,100,cfg['rank'],cfg['cosine_scale']).cuda();train_tf,test_tf=transforms()
    p=CombinedPilot(model,raw,manifest,loader,train_tf.cuda(),test_tf.cuda(),cfg,root/'runs/main',prefix['identity'],clock)
    fork=copy.deepcopy(prefix);fork['clock']=clock;fork['config']['server_correction']=True
    import_prefix(p,fork)
    p.s['prefix_provenance'].update(source_clock=oldclock,same_clock=False,source_verified=True,
                                    source_optimizer='AdamW',source_teacher=False)
    p.save()
    write(root/'reports/SOURCE_CHECK.json',{'passed':True,'source_task':1,'source_round':0,'source_phase':'task_start',
          'source_train_visits':125000,'source_clock_verified':True,'source_and_control_identity_match':True,
          'source_control_config_differences':sorted(differences),'baseline_train_visits_including_t1':375000,
          'baseline_completed_server_rounds':10,'source_gpu_rng_count':len(prefix['rng']['cuda'])})
    while p.step():
        write(root/'reports/PROGRESS.json',{'time':time.time(),'task':p.s['task'],'round':p.s['round'],
              'phase':p.s['phase'],'train_records':len(p.s['training_records'])})
    records=p.s['training_records'];select=p.s['selection_records'];server=p.s.get('combined_server_records',[])
    if (len(records)!=150 or len({(r['task'],r['round'],r['client']) for r in records})!=150 or
        sum(r['visited'] for r in records)!=375000 or p.s['history'].committed!={0,1,2} or
        len(select)!=2 or len(server)!=10 or len(p.s['certificates'])!=15):
        raise ValueError('new branch lifecycle/cost failure')
    stages=copy.deepcopy(p.s['stages']);comm=copy.deepcopy(p.s['communication'])
    control=results['rrs_server']['stages'][2];new=stages[2]
    if any(abs(b['stages'][2][k]-control[k])>1e-12 for k in ('all_seen','old_accuracy','new_accuracy','forgetting')):
        raise ValueError('control metrics source mismatch')
    with np.load(p.out/'task3_private_logits.npz') as data:
        labels=data['labels'].copy();pred=data['logits'].argmax(1).copy()
    p.restore();p.s['task']=2;again=p.evaluate()
    with np.load(p.out/'task3_private_logits.npz') as data:
        if len(data['labels'])!=3000 or not np.array_equal(labels,data['labels']) or not np.array_equal(pred,data['logits'].argmax(1)):
            raise ValueError('branch endpoint reload mismatch')
    if again['confusion']!=stages[2]['confusion']:raise ValueError('reloaded confusion mismatch')
    metrics={name:{'raw':v['all_seen'],'old':v['old_accuracy'],'new':v['new_accuracy'],'forgetting':v['forgetting']}
             for name,v in [('control',control),('adamw_basis',new)]}
    checks={'raw_net_at_least_6':round((new['all_seen']-control['all_seen'])*3000)>=6,
            'old_correct_not_lower':new['old_accuracy']>=control['old_accuracy'],
            'new_loss_at_most_10':round((control['new_accuracy']-new['new_accuracy'])*1000)<=10,
            'forgetting_not_higher':new['forgetting']<=control['forgetting']+1e-12,
            'at_least_one_basis_accept':sum(v['accepted_layers'] for v in select)>0,
            'at_least_one_server_accept':any(v['accepted'] for v in server)}
    basis_cost={}
    for record in p.s.get('probe_records',[]):
        basis_cost[record['kind']]=basis_cost.get(record['kind'],0)+int(record['samples'])
    write(root/'reports/RESULT.json',{'status':'completed','seed':0,'metrics':metrics,'checks':checks,
          'exploratory_gate_passed':all(checks.values()),'source_t1_train_visits':125000,
          'new_train_visits':250000,'client_rounds_including_source_t1':150,'teacher_forward_images':0,
          'old_training_images':0,'current_stats_visits':sum(x['samples'] for x in p.s['stats_records'])-30000,
          'selection_accepted_layers':[v['accepted_layers'] for v in select],
          'server_accepted_rounds':sum(v['accepted'] for v in server),'server_records':server,
          'basis_probe_samples_by_kind':basis_cost,
          'server_probe_forward_images':sum(v['forward_images'] for v in server),
          'server_probe_backward_images':sum(v['backward_images'] for v in server),
          'server_assumed_gradient_upload_bytes':sum(v['assumed_gradient_upload_bytes'] for v in server),
          'server_assumed_probe_broadcast_bytes':sum(v['assumed_probe_broadcast_bytes'] for v in server),
          'server_seconds_including_probes':sum(v['seconds'] for v in server),
          'logical_communication_bytes':comm,'raw_reload_equal':True,'test_images':8000,
          'test_count_basis':'task2 2000 + task3 3000 + task3 reload 3000; no per-image counter',
          'source_test_previously_seen':True,'network_measured':False,
          'elapsed_seconds':time.time()-clock['started'],'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated()})

if __name__=='__main__':main()
