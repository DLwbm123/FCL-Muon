"""Two explicitly authorized runs. A imports unchanged UtilityPilot, B isolated repair."""
import copy,gc,json,os,random,time
from pathlib import Path
import numpy as np
import torch,timm
from safetensors.torch import load_file
from torchvision.datasets import CIFAR100
from fcl.data import loader,transforms
from fcl.model import MuLoRAFL
from research.aggregation_utility.pilot import UtilityPilot
from research.boundary_repair.lifecycle import import_boundary,config_hash
from research.basic_repair.fcl_basic_repair_tools import center_task_logits


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def main():
    root=Path(os.environ['PROJECT_ROOT']);arm=os.environ['EXPERIMENT_ARM'];assert arm in ('A','B')
    cfg=json.loads((root/'configs'/('utility_ten_seed0.json' if arm=='A' else 'boundary_three_seed0.json')).read_text())
    clock=json.loads((root/'reports/job.json').read_text());clock={k:clock[k] for k in ('started','deadline')}
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.manual_seed(0);np.random.seed(0);random.seed(0)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    manifest=json.loads((root/cfg['manifest']).read_text());weights=json.loads((root/'weights.json').read_text())
    assert manifest['identity']=={k:cfg[k] for k in ('dataset','seed','alpha','clients')}
    raw=[CIFAR100(root/'data',train=t,download=False) for t in (True,False)]
    assert [list(d.targets) for d in raw]==[manifest['labels']['train'],manifest['labels']['test']]
    identity={'manifest':cfg['manifest'],'weight_identity':weights['sha256'],'source_revision':cfg['source_revision'],
              'purpose':'three-task aggregation-utility selection'}
    backbone=timm.create_model(cfg['backbone'],pretrained=False,num_classes=0)
    backbone.load_state_dict({k:v for k,v in load_file(weights['path']).items() if not k.startswith('head.')},strict=True)
    model=MuLoRAFL(backbone,100,cfg['rank'],cfg['cosine_scale']).cuda()
    train_tf,test_tf=transforms();cls=UtilityPilot
    if arm=='B':
        from research.boundary_repair.pilot import BoundaryPilot
        cls=BoundaryPilot
    p=cls(model,raw,manifest,loader,train_tf.cuda(),test_tf.cuda(),cfg,root/cfg['output'],identity,clock)
    torch.cuda.reset_peak_memory_stats()
    if (p.out/'latest.pt').exists():
        assert os.environ.get('RESUME_RUN')=='1','No implicit retry'
        p.restore()
    else:
        expected=json.loads((root/'reports/IMPORT_SOURCE.json').read_text())
        source_path=Path(os.environ['SOURCE_CHECKPOINT'])
        source=torch.load(source_path,map_location='cpu',weights_only=False)
        proof=import_boundary(p,source,expected);del source
        write(root/'reports/IMPORT_PROOF.json',proof)
        # Read-only source reference preserves the T3/T1 boundary and original logits.
        (p.out/'source_boundary.pt').symlink_to(source_path)
        for t in range(1,expected['completed_tasks']+1):
            (p.out/f'task{t}_private_logits.npz').symlink_to(source_path.parent/f'task{t}_private_logits.npz')
    if arm=='B' and not (root/'reports/REAL_PREFLIGHT.json').exists():
        from research.boundary_repair.preflight import real_batch
        write(root/'reports/REAL_PREFLIGHT.json',real_batch(p));gc.collect();torch.cuda.empty_cache()
    def progress():
        write(root/'reports/PROGRESS.json',{'time':time.time(),'arm':arm,'task':p.s['task'],'round':p.s['round'],
             'phase':p.s['phase'],'next_client':p.s['next_client'],'client_rounds':len(p.s['training_records']),
             'training_visits':sum(v['visited'] for v in p.s['training_records']),'completed_stages':len(p.s['stages'])})
    progress()
    while p.step():progress()
    n=cfg['tasks'];records=p.s['training_records']
    assert len(records)==n*50 and len({(v['task'],v['round'],v['client']) for v in records})==n*50
    assert sum(v['visited'] for v in records)==n*125000 and sum(v['samples'] for v in p.s['stats_records'])==n*30000
    assert p.s['merges']==list(range(n)) and p.s['history'].committed==set(range(n)) and len(model.rows)==n
    assert len(p.s['certificates'])==n*5 and all(v['deployed_fp32']['passed'] for v in p.s['certificates'])
    assert len(p.s['selection_records'])==n-1 and all(not torch.count_nonzero(b) for b in model.b_parameters())
    with np.load(p.out/f'task{n}_private_logits.npz') as f:before={k:f[k].copy() for k in f.files}
    stages=copy.deepcopy(p.s['stages']);p.restore();p.s['task']=n-1
    final=p.evaluate()
    with np.load(p.out/f'task{n}_private_logits.npz') as f:after={k:f[k].copy() for k in f.files}
    assert len(after['labels'])==n*1000 and np.array_equal(before['indices'],after['indices']) and np.array_equal(before['labels'],after['labels'])
    raw_equal=np.array_equal(before['logits'].argmax(1),after['logits'].argmax(1))
    centered_equal=np.array_equal(center_task_logits(before['logits'],10).argmax(1),center_task_logits(after['logits'],10).argmax(1))
    assert raw_equal and centered_equal and final['confusion']==stages[-1]['confusion'] and final['centered']==stages[-1]['centered']
    p.s['task']=n  # in-memory task restored; never save the temporary evaluation phase
    for t in range(1,n+1):
        with np.load(p.out/f'task{t}_private_logits.npz') as f:
            np.savez(p.out/f'task{t}_private_predictions.npz',indices=f['indices'],labels=f['labels'],
                     raw=f['logits'].argmax(1),centered=center_task_logits(f['logits'],10).argmax(1))
    reused=p.s['boundary_import']['reused_training_visits']
    q={'qualified':True,'arm':arm,'clock':clock,'config_sha256':config_hash(cfg),'stages':stages,
       'client_rounds':len(records),'training_visits':n*125000,'new_training_visits':n*125000-reused,
       'stats_visits':n*30000,'merges':p.s['merges'],'history_committed':sorted(p.s['history'].committed),
       'certificates_passed':n*5,'selection_records':p.s['selection_records'],'probe_records':p.s['probe_records'],
       'repair_records':p.s.get('repair_records',[]),'finite_step_windows':p.s.get('finite_step_windows',[]),
       'logical_communication_bytes':p.s['communication'],'reload_images':n*1000,
       'reload_raw_predictions_equal':raw_equal,'reload_centered_predictions_equal':centered_equal,
       'maximum_reload_logit_difference':float(np.abs(before['logits']-after['logits']).max()),
       'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'elapsed_seconds':time.time()-clock['started'],
       'checkpoint_bytes':(p.out/'latest.pt').stat().st_size,'import_proof':p.s['boundary_import']}
    write(p.out/'QUALIFICATION.json',q);write(p.out/'COMPLETE.json',{'qualified':True,'completed':time.time()})


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        write(Path(os.environ['PROJECT_ROOT'])/'reports/FAILURE.json',{'time':time.time(),'error':repr(exc)})
        raise
