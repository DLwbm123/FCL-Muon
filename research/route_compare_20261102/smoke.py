"""Seed-0 training-only T1 smoke: stop before any test evaluation."""
import hashlib, json, os, random, time
from pathlib import Path
import numpy as np
import timm
import torch
from safetensors.torch import load_file
from torchvision.datasets import CIFAR100
from fcl.data import loader, transforms
from fcl.model import MuLoRAFL
from research.route_compare_20261102.pilot import RoutePilot


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, allow_nan=False))
    os.replace(tmp, path)


def main():
    root, source = Path(os.environ['PROJECT_ROOT']), Path(os.environ['SOURCE_ROOT'])
    optimizer = os.environ['LOCAL_OPTIMIZER']
    if optimizer not in ('adamw', 'muon'): raise ValueError('unknown optimizer')
    state = json.loads((root/'reports/job.json').read_text())
    clock = {k:state[k] for k in ('started','deadline')}
    if time.time() >= clock['deadline']: raise TimeoutError('original clock exhausted')
    cfg = json.loads((root/'configs/aggregation_utility_three_task_seed0.json').read_text())
    cfg.update(retain_round_checkpoints=False, retain_first_round_endpoints=False,
               retain_task_checkpoints=False, selection_mode='prefix')
    cfg['route'] = {'optimizer':optimizer,'mode':'prefix','pool_width':64,
                    'split_seed':76103,'trial_seed':192017,'sample_cap':32,
                    'short_rounds':2,'steps':[.05,.025,.0125],
                    'lambdas':[0.,.001,.01,.1,1.,10.,100.], 'trust_radius':1.,
                    'orthogonal_tolerance':1e-5,'loss_tolerance':1e-7,
                    'space_tolerance':1e-5,'outer_iterations':2}
    cfg['output'] = 'runs/smoke_'+optimizer
    manifest = json.loads((source/cfg['manifest']).read_text())
    if manifest['identity'] != {k:cfg[k] for k in ('dataset','seed','alpha','clients')}:
        raise ValueError('manifest identity mismatch')
    weights = json.loads((source/'weights.json').read_text())
    digest = hashlib.sha256()
    with open(weights['path'],'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): digest.update(block)
    if digest.hexdigest()!=weights['sha256']: raise ValueError('pretrained weight checksum mismatch')
    # Test data are neither opened nor evaluated by the smoke runner.
    raw = CIFAR100(source/'data',train=True,download=False)
    if list(raw.targets) != manifest['labels']['train']: raise ValueError('training data labels mismatch')
    torch.set_num_threads(4); torch.set_float32_matmul_precision('highest')
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed'])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True);torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    torch.cuda.set_per_process_memory_fraction(.7,0)
    backbone=timm.create_model(cfg['backbone'],pretrained=False,num_classes=0)
    backbone.load_state_dict({k:v for k,v in load_file(weights['path']).items() if not k.startswith('head.')},strict=True)
    model=MuLoRAFL(backbone,100,cfg['rank'],cfg['cosine_scale']).cuda()
    train_tf,test_tf=transforms()
    identity={'manifest':cfg['manifest'],'weights_sha256':weights['sha256'],
              'source_revision':cfg['source_revision'],'optimizer':optimizer,'seed':0}
    pilot=RoutePilot(model,[raw,None],manifest,loader,train_tf.cuda(),test_tf.cuda(),cfg,
                     root/cfg['output'],identity,clock)
    start=time.monotonic()
    while pilot.s['phase'] != 'ready_evaluation':
        pilot.step()
        write(root/'reports'/('smoke_'+optimizer+'.json'),
              {'status':'running','phase':pilot.s['phase'],'task':pilot.s['task'],
               'round':pilot.s['round'],'client_rounds':len(pilot.s['training_records']),
               'elapsed':time.monotonic()-start})
    if pilot.s['task']!=0 or len(pilot.s['training_records'])!=50:
        raise ValueError('unexpected smoke endpoint')
    record={'status':'completed_training_only','phase':pilot.s['phase'],
            'client_rounds':len(pilot.s['training_records']),
            'train_visits':sum(v['visited'] for v in pilot.s['training_records']),
            'elapsed':time.monotonic()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'weight_sha256_verified':True,'test_evaluations':0}
    write(root/'reports'/('smoke_'+optimizer+'.json'),record)
    print(json.dumps(record),flush=True)

if __name__ == '__main__': main()
