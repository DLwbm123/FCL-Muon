"""One fresh T1, then RRS/central/federated T2-T3 branches under one frozen clock."""
import copy
import gc
import json
import os
import random
import shutil
import time
from pathlib import Path
import numpy as np
import torch
import timm
from safetensors.torch import load_file
from torchvision.datasets import CIFAR100
from fcl.data import loader,transforms
from fcl.model import MuLoRAFL
from research.aggregation_utility.pilot import UtilityPilot
from research.aggregation_utility.lifecycle import import_prefix
from research.basic_repair.fcl_basic_repair_tools import center_task_logits


def write_json(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def main():
    root=Path(os.environ['PROJECT_ROOT']);cfg=json.loads((root/'configs/aggregation_utility_three_task_seed0.json').read_text())
    clock=json.loads((root/'reports/job.json').read_text());clock={k:clock[k] for k in ['started','deadline']}
    resume=os.environ.get('RESUME_RUN')=='1'
    migration=json.loads((root/'reports/GPU0_MIGRATION_READY.json').read_text()) if resume else None
    if resume:
        assert migration['clock']==clock and migration['authorized_gpu']=='0' and os.environ['CUDA_VISIBLE_DEVICES']=='0'
    out=root/'runs';out.mkdir(exist_ok=resume)
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed'])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    manifest=json.loads((root/cfg['manifest']).read_text())
    assert manifest['identity']=={k:cfg[k] for k in ['dataset','seed','alpha','clients']}
    raw=[CIFAR100(root/'data',train=t,download=False) for t in [True,False]]
    assert [list(v.targets) for v in raw]==[manifest['labels']['train'],manifest['labels']['test']]
    weights=json.loads((root/'weights.json').read_text())
    identity={'manifest':cfg['manifest'],'weight_identity':weights['sha256'],
              'source_revision':cfg['source_revision'],'purpose':'three-task aggregation-utility selection'}

    def build(mode):
        config=copy.deepcopy(cfg);config.update(selection_mode=mode,output='runs/'+mode)
        backbone=timm.create_model(config['backbone'],pretrained=False,num_classes=0)
        backbone.load_state_dict({k:v for k,v in load_file(weights['path']).items() if not k.startswith('head.')},strict=True)
        model=MuLoRAFL(backbone,100,config['rank'],config['cosine_scale']).cuda()
        train_tf,test_tf=transforms()
        return UtilityPilot(model,raw,manifest,loader,train_tf.cuda(),test_tf.cuda(),config,out/mode,identity,clock)

    def progress(p):
        write_json(out/'STUDY_PROGRESS.json',{'time':time.time(),'mode':p.cfg['selection_mode'],
            'task':p.s['task'],'round':p.s['round'],'phase':p.s['phase'],
            'client_rounds_including_shared_prefix':len(p.s['training_records']),
            'completed_stages':len(p.s['stages']),'clock':clock})

    if resume and (out/'PREFIX_READY.json').exists():
        prefix_meta=json.loads((out/'PREFIX_READY.json').read_text())
        assert prefix_meta['clock']==clock and prefix_meta['task1_complete']
    else:
        prefix=build('prefix');prefix_started=time.time();torch.cuda.reset_peak_memory_stats()
        if resume and (prefix.out/'latest.pt').exists():
            prefix.restore()
            prefix.event('user_requested_gpu_resume',gpu=0,original_clock=clock,
                         checkpoint_client_records=len(prefix.s['training_records']))
        while not (prefix.s['task']==1 and prefix.s['phase']=='task_start'):
            assert prefix.step();progress(prefix)
        prefix.restore()
        assert prefix.s['merges']==[0] and prefix.s['history'].committed=={0}
        assert len(prefix.s['certificates'])==5 and all(v['deployed_fp32']['passed'] for v in prefix.s['certificates'])
        prefix_meta={'task1_complete':True,'training_visits':sum(v['visited'] for v in prefix.s['training_records']),
                     'client_rounds':len(prefix.s['training_records']),'stats_visits':sum(v['samples'] for v in prefix.s['stats_records']),
                     'current_worker_segment_seconds':time.time()-prefix_started,
                     'study_elapsed_at_prefix_completion_seconds':time.time()-clock['started'],
                     'peak_gpu_allocated_bytes_current_segment':torch.cuda.max_memory_allocated(),
                     'migration':migration,'first_stage':prefix.s['stages'][0],'clock':clock}
        write_json(out/'PREFIX_READY.json',prefix_meta)
        del prefix;gc.collect();torch.cuda.empty_cache()
    qualifications={}
    for mode in ['rrs','central','federated']:
        p=build(mode);branch_start=time.time();torch.cuda.reset_peak_memory_stats()
        if resume and (p.out/'COMPLETE.json').exists():
            q=json.loads((p.out/'QUALIFICATION.json').read_text());assert q['qualified'] and q['clock']==clock
            qualifications[mode]={'qualified':True,'accepted_layers':[v['accepted_layers'] for v in q['selection_records']],
                                 'new_branch_seconds':q['new_branch_seconds']}
            del p;gc.collect();torch.cuda.empty_cache();continue
        if resume and (p.out/'latest.pt').exists():
            p.restore()
        else:
            checkpoint=torch.load(out/'prefix/latest.pt',map_location='cpu',weights_only=False)
            import_prefix(p,checkpoint);del checkpoint
            shutil.copyfile(out/'prefix/task1_private_logits.npz',p.out/'task1_private_logits.npz')
        while p.step():progress(p)
        records=p.s['training_records'];stages=p.s['stages'];selections=p.s['selection_records']
        assert len(records)==150 and len({(v['task'],v['round'],v['client']) for v in records})==150
        assert sum(v['visited'] for v in records)==375000
        assert sum(v['samples'] for v in p.s['stats_records'])==90000
        assert p.s['merges']==[0,1,2] and p.s['history'].committed=={0,1,2}
        assert len(p.s['certificates'])==15 and all(v['deployed_fp32']['passed'] for v in p.s['certificates'])
        assert len(selections)==2 and all(v['fit_samples']+v['check_samples']==5000 for v in selections)
        assert all(torch.count_nonzero(b)==0 for b in p.model.b_parameters())
        with np.load(p.out/'task3_private_logits.npz') as data:
            before={k:data[k].copy() for k in data.files}
        p.restore();p.s['task']=2
        final=p.evaluate()
        with np.load(p.out/'task3_private_logits.npz') as data:
            after={k:data[k].copy() for k in data.files}
        assert np.array_equal(before['indices'],after['indices']) and np.array_equal(before['labels'],after['labels'])
        raw_equal=np.array_equal(before['logits'].argmax(1),after['logits'].argmax(1))
        centered_equal=np.array_equal(center_task_logits(before['logits'],10).argmax(1),center_task_logits(after['logits'],10).argmax(1))
        assert raw_equal and centered_equal and len(after['labels'])==3000
        assert final['confusion']==stages[-1]['confusion'] and final['centered']==stages[-1]['centered']
        qualification={'qualified':True,'mode':mode,'clock':clock,'training_visits_including_shared_prefix':375000,
            'new_branch_training_visits':250000,'client_rounds_including_shared_prefix':150,'stats_visits_including_shared_prefix':90000,
            'selection_records':selections,'probe_records':p.s['probe_records'],'logical_communication_bytes':p.s['communication'],
            'full_3000_checkpoint_reload_raw_predictions_equal':raw_equal,'full_3000_checkpoint_reload_centered_predictions_equal':centered_equal,
            'maximum_reload_logit_difference':float(np.abs(before['logits']-after['logits']).max()),
            'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'new_branch_seconds':time.time()-branch_start,
            'checkpoint_bytes':(p.out/'latest.pt').stat().st_size,'certificates_passed':15,'merges':[0,1,2]}
        write_json(p.out/'QUALIFICATION.json',qualification)
        write_json(p.out/'COMPLETE.json',{'completed':time.time(),'qualified':True})
        qualifications[mode]={'qualified':True,'accepted_layers':[v['accepted_layers'] for v in selections],
                              'new_branch_seconds':qualification['new_branch_seconds']}
        del p;gc.collect();torch.cuda.empty_cache()
    write_json(out/'STUDY_QUALIFICATION.json',{'qualified':True,'prefix':prefix_meta,'arms':qualifications,
               'completed_record_training_visits':875000,'attributed_training_visits_including_prefix_per_arm':375000,
               'user_requested_gpu_migration':migration,
               'actual_client_rounds':350,'actual_base_statistics_visits':210000,
               'actual_task_eval_images':16000,'actual_reload_images':9000,'clock':clock})
    write_json(out/'COMPLETE.json',{'completed':time.time(),'qualified':True})


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        root=Path(os.environ['PROJECT_ROOT']);root.joinpath('reports/FAILURE.json').write_text(json.dumps({'time':time.time(),'error':repr(exc)},indent=2))
        raise
