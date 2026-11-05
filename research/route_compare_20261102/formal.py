"""Frozen four-arm worker. Never run before the protocol is frozen."""
import copy, gc, hashlib, json, os, random, time
from pathlib import Path
import numpy as np
import timm
import torch
from safetensors.torch import load_file
from torchvision.datasets import CIFAR100
from fcl.data import loader, transforms
from fcl.model import MuLoRAFL
from research.aggregation_utility.lifecycle import import_prefix
from research.route_compare_20261102.pilot import RoutePilot
from research.route_compare_20261102.smoke import write


def main():
    root, source = Path(os.environ['PROJECT_ROOT']), Path(os.environ['SOURCE_ROOT'])
    # The supervisor creates this only after both smoke receipts and full
    # pre-test protocol/config/source checks have been completed.
    frozen=json.loads((root/'reports/FROZEN.json').read_text())
    if frozen.get('status')!='frozen_before_new_test' or frozen.get('tasks')!=3:
        raise ValueError('missing frozen three-task protocol')
    if not frozen.get('source_sha256'):
        raise ValueError('missing source freeze')
    for relative, expected in frozen['source_sha256'].items():
        candidate=(root/relative).resolve()
        if not candidate.is_relative_to(root.resolve()) or hashlib.sha256(candidate.read_bytes()).hexdigest()!=expected:
            raise ValueError('frozen source mismatch')
    seed=int(os.environ['STUDY_SEED'])
    if seed not in (0,1,2):raise ValueError('unregistered seed')
    job=json.loads((root/'reports/job.json').read_text());clock={k:job[k] for k in ('started','deadline')}
    if time.time()>=clock['deadline']:raise TimeoutError('original clock exhausted')
    cfg=json.loads((root/'configs/aggregation_utility_three_task_seed0.json').read_text())
    cfg.update(seed=seed,retain_round_checkpoints=False,retain_first_round_endpoints=False,
               retain_task_checkpoints=False,selection_mode='prefix')
    cfg['manifest']='manifests/seq-cifar100_n10_a0.05_s0.json' if seed==0 else f'manifests/seed{seed}.json'
    manifest_path=(source/cfg['manifest']) if seed==0 else (root/cfg['manifest'])
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest()!=frozen['manifest_sha256'][str(seed)]:
        raise ValueError('frozen manifest mismatch')
    manifest=json.loads(manifest_path.read_text())
    if manifest['identity']!={k:cfg[k] for k in ('dataset','seed','alpha','clients')}:
        raise ValueError('manifest identity mismatch')
    weights=json.loads((source/'weights.json').read_text())
    if weights['sha256']!=frozen['weight_sha256']:raise ValueError('weight metadata mismatch')
    raw=[CIFAR100(source/'data',train=t,download=False) for t in (True,False)]
    if [list(v.targets) for v in raw]!=[manifest['labels']['train'],manifest['labels']['test']]:
        raise ValueError('dataset labels mismatch')
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True);torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    torch.cuda.set_per_process_memory_fraction(.7,0)
    train_tf,test_tf=transforms();train_tf=train_tf.cuda();test_tf=test_tf.cuda()
    def build(optimizer,mode,out):
        current=copy.deepcopy(cfg);current['output']=out
        current['selection_mode']='prefix' if mode=='prefix' else mode
        current['route']=copy.deepcopy(frozen['route'])
        current['route'].update(optimizer=optimizer,mode=mode)
        backbone=timm.create_model(current['backbone'],pretrained=False,num_classes=0)
        backbone.load_state_dict({k:v for k,v in load_file(weights['path']).items() if not k.startswith('head.')},strict=True)
        model=MuLoRAFL(backbone,100,current['rank'],current['cosine_scale']).cuda()
        identity={'manifest':current['manifest'],'weights_sha256':weights['sha256'],
                  'source_revision':current['source_revision'],'optimizer':optimizer,'seed':seed}
        return RoutePilot(model,raw,manifest,loader,train_tf,test_tf,current,root/out,identity,clock)
    def progress(p,arm):
        write(root/'reports'/f'progress_{seed}.json',{'time':time.time(),'seed':seed,'arm':arm,
             'task':p.s['task'],'round':p.s['round'],'phase':p.s['phase'],
             'client_rounds':len(p.s['training_records'])})
    result={}
    for optimizer,arms in (('adamw',('A0','A1')),('muon',('M0','M1'))):
        torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
        name=f'runs/smoke_{optimizer}' if seed==0 else f'runs/seed{seed}_{optimizer}_prefix'
        p=build(optimizer,'prefix',name)
        if seed==0:
            p.restore()
            if (p.s['task'],p.s['phase'])!=(0,'ready_evaluation') or len(p.s['training_records'])!=50:
                raise ValueError('smoke checkpoint is not the frozen T1 endpoint')
        while not (p.s['task']==1 and p.s['phase']=='task_start'):
            if not p.step():raise ValueError('incomplete shared T1')
            progress(p,optimizer+'_prefix')
        p.restore(); ck=torch.load(p.out/'latest.pt',map_location='cpu',weights_only=False)
        if ck['state']['history'].committed!={0} or len(ck['state']['training_records'])!=50:
            raise ValueError('invalid committed T1')
        del p;gc.collect();torch.cuda.empty_cache()
        for arm,mode in zip(arms,('rrs','selected')):
            p=build(optimizer,mode,f'runs/seed{seed}_{arm}')
            fork=copy.deepcopy(ck)
            # Explicitly permit only the pre-registered route flag to differ.
            fork['config']['route']['mode']=mode
            import_prefix(p,fork);del fork
            while p.step():progress(p,arm)
            if len(p.s['training_records'])!=150 or p.s['history'].committed!={0,1,2}:
                raise ValueError('formal lifecycle incomplete')
            stages=copy.deepcopy(p.s['stages']);records=copy.deepcopy(p.s.get('route_records',[]))
            comm=copy.deepcopy(p.s['communication']);p.restore();p.s['task']=2
            check=p.evaluate()
            if check['confusion']!=stages[-1]['confusion']:raise ValueError('reloaded metrics differ')
            result[arm]={'stages':stages,'route_records':records,'communication':comm,
                         'client_rounds':150,'reload_confusion_equal':True}
            write(root/'reports'/f'result_seed{seed}_private.json',result)
            del p;gc.collect();torch.cuda.empty_cache()
        del ck
    write(root/'reports'/f'seed{seed}_done.json',{'status':'completed','arms':list(result)})

if __name__=='__main__':main()
