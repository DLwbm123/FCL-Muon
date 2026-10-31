"""Four paired teacher-free three-task branches, from one fresh AdamW T1."""
import copy, gc, json, os, random, shutil, time
from pathlib import Path
import numpy as np
import torch
import timm
from safetensors.torch import load_file
from torchvision.datasets import CIFAR100
from fcl.data import loader, transforms
from fcl.model import MuLoRAFL
from research.adamw_combined.pilot import CombinedPilot
from research.aggregation_utility.lifecycle import import_prefix


def write(path, data):
    temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(data, indent=2, allow_nan=False)); temp.replace(path)


def main():
    root=Path(os.environ['PROJECT_ROOT']); source=Path(os.environ['SOURCE_ROOT'])
    cfg=json.loads((root/'configs/aggregation_utility_three_task_seed0.json').read_text())
    cfg.update(retain_round_checkpoints=False,retain_first_round_endpoints=False,server_correction=False)
    job=json.loads((root/'reports/job.json').read_text()); clock={k:job[k] for k in ('started','deadline')}
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed'])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True);torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    manifest=json.loads((source/cfg['manifest']).read_text());weights=json.loads((source/'weights.json').read_text())
    if manifest['identity']!={k:cfg[k] for k in ('dataset','seed','alpha','clients')}:raise ValueError('manifest mismatch')
    raw=[CIFAR100(source/'data',train=train,download=False) for train in (True,False)]
    if [list(x.targets) for x in raw] != [manifest['labels']['train'],manifest['labels']['test']]:raise ValueError('dataset mismatch')
    identity={'manifest':cfg['manifest'],'weight_identity':weights['sha256'],'source_revision':cfg['source_revision'],
              'purpose':'teacher-free AdamW subspace and all-seen combination'}
    run=root/'runs';run.mkdir()
    def build(name, mode, correction):
        config=copy.deepcopy(cfg);config.update(selection_mode=mode,output='runs/'+name,server_correction=correction)
        backbone=timm.create_model(config['backbone'],pretrained=False,num_classes=0)
        backbone.load_state_dict({k:v for k,v in load_file(weights['path']).items() if not k.startswith('head.')},strict=True)
        model=MuLoRAFL(backbone,100,config['rank'],config['cosine_scale']).cuda()
        train_tf,test_tf=transforms()
        return CombinedPilot(model,raw,manifest,loader,train_tf.cuda(),test_tf.cuda(),config,run/name,identity,clock)
    def progress(p,name):
        write(root/'reports/PROGRESS.json',{'time':time.time(),'arm':name,'task':p.s['task'],
              'round':p.s['round'],'phase':p.s['phase'],'train_records':len(p.s['training_records'])})
    prefix=build('prefix','prefix',False)
    while not (prefix.s['task']==1 and prefix.s['phase']=='task_start'):
        if not prefix.step():raise ValueError('prefix incomplete')
        progress(prefix,'prefix')
    prefix.restore()
    if prefix.s['merges']!=[0] or prefix.s['history'].committed!={0}:raise ValueError('prefix lifecycle')
    source_ck=torch.load(prefix.out/'latest.pt',map_location='cpu',weights_only=False)
    del prefix;gc.collect();torch.cuda.empty_cache()
    summary={}
    for name,mode,server in [('rrs_avg','rrs',False),('fed_avg','federated',False),
                              ('rrs_server','rrs',True),('fed_server','federated',True)]:
        p=build(name,mode,server)
        # import_prefix validates all identical fields and the true source clock/RNG;
        # this controlled config override is limited to the separately frozen arm flag.
        fork=copy.deepcopy(source_ck);fork['config']['server_correction']=server
        import_prefix(p,fork);del fork
        shutil.copyfile(run/'prefix/task1_private_logits.npz',p.out/'task1_private_logits.npz')
        while p.step():progress(p,name)
        records=p.s['training_records'];selections=p.s['selection_records'];server_records=p.s.get('combined_server_records',[])
        if (len(records)!=150 or len({(v['task'],v['round'],v['client']) for v in records})!=150 or
            sum(v['visited'] for v in records)!=375000 or p.s['history'].committed!={0,1,2} or
            len(selections)!=2 or len(server_records)!=(10 if server else 0)):
            raise ValueError('branch lifecycle/cost failure')
        stages=copy.deepcopy(p.s['stages']);comm=copy.deepcopy(p.s['communication'])
        with np.load(p.out/'task3_private_logits.npz') as data:
            labels=data['labels'].copy();pred=data['logits'].argmax(1).copy()
        p.restore();p.s['task']=2;again=p.evaluate()
        with np.load(p.out/'task3_private_logits.npz') as data:
            if len(data['labels'])!=3000 or not np.array_equal(labels,data['labels']) or not np.array_equal(pred,data['logits'].argmax(1)):
                raise ValueError('endpoint reload mismatch')
        if again['confusion']!=stages[-1]['confusion']:raise ValueError('evaluation mismatch')
        summary[name]={'stages':stages,'selection_accepted_layers':[s['accepted_layers'] for s in selections],
                       'server_records':server_records,'communication':comm,'train_visits':375000,
                       'client_rounds':150,'reload_raw_equal':True}
        write(root/'reports/ARM_SUMMARY_PRIVATE.json',summary)
        del p;gc.collect();torch.cuda.empty_cache()
    base=summary['rrs_avg']['stages'][2]; full=summary['fed_server']['stages'][2]
    score={k:{'raw':v['stages'][2]['all_seen'],'old':v['stages'][2]['old_accuracy'],
              'new':v['stages'][2]['new_accuracy'],'forgetting':v['stages'][2]['forgetting'],
              'selection_accepted_layers':v['selection_accepted_layers'],
              'server_accepted_rounds':sum(r['accepted'] for r in v['server_records'])}
           for k,v in summary.items()}
    checks={'full_vs_rrs_avg_net_at_least_6':round((full['all_seen']-base['all_seen'])*3000)>=6,
            'old_not_lower':full['old_accuracy']>=base['old_accuracy'],
            'new_loss_at_most_10':round((base['new_accuracy']-full['new_accuracy'])*1000)<=10,
            'forgetting_not_higher':full['forgetting']<=base['forgetting']+1e-12,
            'at_least_one_basis_accept':sum(summary['fed_server']['selection_accepted_layers'])>0,
            'at_least_one_server_accept':score['fed_server']['server_accepted_rounds']>0}
    clean={name:{'selection_accepted_layers':v['selection_accepted_layers'],'server_records':v['server_records'],
                 'communication':v['communication'],'train_visits':v['train_visits'],
                 'client_rounds':v['client_rounds'],'reload_raw_equal':v['reload_raw_equal']}
           for name,v in summary.items()}
    write(root/'reports/RESULT.json',{'status':'completed','seed':0,'metrics':score,'checks':checks,
          'exploratory_gate_passed':all(checks.values()),'arms':clean,'fresh_t1_visits':125000,
          'new_branch_visits':1000000,'total_new_training_visits':1125000,
          'teacher_images':0,'old_training_images':0,'test_images_including_reload':33000,
          'test_count_basis':'T1 shared 1000 + four arms (T2 2000 + T3 3000 + T3 reload 3000)',
          'elapsed_seconds':time.time()-clock['started'],'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated()})

if __name__=='__main__':main()
