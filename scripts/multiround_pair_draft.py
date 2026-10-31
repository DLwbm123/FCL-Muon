"""Unregistered two-arm multiround pilot; do not run before protocol freeze."""
import copy
import json
import os
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from torchvision.datasets import CIFAR100
from fcl.checkpoint import rng_state,restore_rng
from fcl.data import loader,transforms
from fcl.math import mulora_ssb
from research.head_regmean.pilot import Pilot
from research.fed_spectra.heldout_round import GuardedLoader,split_clients
from research.continual_muon.current_cifar import CurrentCIFAR
from research.continual_muon.current_task_guard import CurrentTaskGuard
from research.continual_muon.import_t2_round import import_boundary
from research.continual_muon.multiround_server_draft import MainlinePilot
from research.continual_muon.paired_metrics import compare
from scripts.paired_boundary_worker import model_for,predictions
from scripts.real_boundary_worker import same_rng,write


def main():
    root,source,endpoints=[Path(os.environ[k]) for k in ('PROJECT_ROOT','SOURCE_ROOT','ENDPOINT_ROOT')]
    report=root/'reports';job=json.loads((report/'job.json').read_text())
    clock={k:job[k] for k in ('started','deadline')}
    if not clock['started']<=time.time()<clock['deadline']:raise TimeoutError('study original deadline expired')
    with (report/'WORKER_CLAIM.json').open('x') as f:json.dump({'started':time.time(),'clock':clock},f)
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    checkpoint=torch.load(endpoints/'runs/t2_endpoint_private.pt',map_location='cpu',weights_only=False)
    snapshot=torch.load(endpoints/'runs/t2_private_state.pt',map_location='cpu',weights_only=False)
    cfg=copy.deepcopy(checkpoint['config'])
    expected={'seed':0,'dataset':'seq-cifar100','alpha':.05,'clients':10,'rounds':5,'epochs':5,
              'batch':16,'rank':32,'lr':.0002,'aggregation':'regmean','selection_mode':'prefix'}
    if any(cfg.get(k)!=v for k,v in expected.items()):raise ValueError('source configuration mismatch')
    manifest=json.loads((source/cfg['manifest']).read_text());weights=json.loads((source/'weights.json').read_text())
    identity={'manifest':cfg['manifest'],'weight_identity':weights['sha256'],
              'source_revision':cfg['source_revision'],'purpose':'three-task aggregation-utility selection'}
    if manifest['identity']!={k:cfg[k] for k in ('dataset','seed','alpha','clients')}:
        raise ValueError('source identity mismatch')
    original=manifest['partitions']['train'];fit,heldout=split_clients(original[2],0)
    if (sum(map(len,fit)),sum(map(len,heldout)))!=(4005,995) or sum(min(len(x),32) for x in fit)!=281:
        raise ValueError('frozen fit split mismatch')
    indices={t:[i for g in original[t] for i in g] for t in (1,2)}
    groups=copy.deepcopy(original);groups[2]=fit
    train_manifest={'identity':manifest['identity'],'partitions':{'train':groups},'labels':{'train':manifest['labels']['train']}}
    raw=CurrentCIFAR(source/'data/cifar-100-python/train',train_manifest['labels']['train'],indices)
    taskguard=CurrentTaskGuard(loader,raw,indices)
    guard=GuardedLoader(taskguard,raw,[i for g in fit for i in g],[i for g in heldout for i in g])
    train_tf,test_tf=transforms();train_tf=train_tf.cuda();test_tf=test_tf.cuda()
    try:
        shared=Pilot(model_for(cfg),[raw,None],train_manifest,guard,train_tf,test_tf,cfg,
                     root/'runs/shared',identity,clock)
        receipt=import_boundary(shared,checkpoint,snapshot,identity)
        if len(checkpoint['rng']['cuda'])!=1 or not torch.equal(torch.cuda.get_rng_state(),checkpoint['rng']['cuda'][0]):
            raise ValueError('source GPU RNG mismatch')
        t1_accuracy=checkpoint['state']['matrix'][0][0]
        before=rng_state();i=torch.arange(768,dtype=torch.float32)[:,None]+1;j=torch.arange(32,dtype=torch.float32)[None,:]+1
        q=torch.linalg.qr(torch.sin(i*j*.013),mode='reduced').Q
        err=float(torch.linalg.vector_norm(mulora_ssb(q)-mulora_ssb(q.cuda()).cpu()))
        if not np.isfinite(err) or err>1e-4 or not same_rng(before,rng_state()):raise ValueError('GPU SSB preflight failed')
        write(report/'REAL_PREFLIGHT.json',{'passed':True,'cpu_gpu_ssb_l2':err,'rng_unchanged':True,'image_accesses':0})
        shared.s['phase']='evaluated'
        for phase in ('merged','committed','task_start'):
            shared.step()
            if shared.s['phase']!=phase:raise ValueError('source task boundary transition mismatch')
        raw.enter_t3();taskguard.enter_t3();shared.step()
        if (shared.s['task'],shared.s['round'],shared.s['phase'])!=(2,0,'round_start'):
            raise ValueError('shared task initialization mismatch')
        initial_rng=rng_state();shared.save();shared_path=shared.out/'latest.pt'
        if guard.training_access_images!=4005 or guard.validation_access_images:raise ValueError('shared RRS count mismatch')
        branch_paths={};branch_comm={};main_records=[];fit_counts={}
        for mode,cls in (('control',Pilot),('mainline',MainlinePilot)):
            out=root/'runs'/mode;out.mkdir();shutil.copyfile(shared_path,out/'latest.pt')
            pilot=cls(model_for(cfg),[raw,None],train_manifest,guard,train_tf,test_tf,cfg,out,identity,clock)
            pilot.restore()
            if not same_rng(initial_rng,rng_state()):raise ValueError('branch RNG mismatch')
            before_reads=guard.training_access_images;before_comm=copy.deepcopy(pilot.s['communication'])
            while pilot.s['phase']!='ready_evaluation':
                if pilot.s['task']!=2:raise ValueError('unexpected branch task')
                pilot.step()
                write(report/'PROGRESS.json',{'time':time.time(),'stage':mode,'round':pilot.s['round'],
                      'phase':pilot.s['phase'],'current_train_reads':guard.training_access_images,
                      'heldout_reads':guard.validation_access_images})
            records=[r for r in pilot.s['training_records'] if r['task']==2]
            if len(records)!=50 or sum(r['visited'] for r in records)!=100125:
                raise ValueError('branch training budget mismatch')
            branch_reads=guard.training_access_images-before_reads
            expected_reads=120150+(1405 if mode=='mainline' else 0)
            if branch_reads!=expected_reads or guard.validation_access_images:
                raise ValueError('branch train read or heldout mismatch')
            fit_counts[mode]=branch_reads
            branch_comm[mode]={k:int(v-before_comm.get(k,0)) for k,v in pilot.s['communication'].items()}
            if mode=='mainline':
                main_records=copy.deepcopy(pilot.s.get('mainline_server_records',[]))
                if (len(main_records)!=5 or sum(r['fit_probe_cost']['student_forward_images'] for r in main_records)!=1405 or
                        sum(r['fit_probe_cost']['student_backward_images'] for r in main_records)!=2810):
                    raise ValueError('mainline probe accounting mismatch')
            branch_paths[mode]=out
            del pilot;torch.cuda.empty_cache()
        if raw.reads!={1:0,2:245710} or raw.reads!=taskguard.visits or guard.validation_access_images:
            raise ValueError('training-side image isolation mismatch')
        # No test file or test images are opened until both training endpoints have been saved.
        test=CIFAR100(source/'data',train=False,download=False)
        if list(test.targets)!=manifest['labels']['test']:raise ValueError('test labels mismatch')
        t2_model=model_for(cfg)
        for _ in range(checkpoint['head_count']):t2_model.add_head(cfg['cpt'])
        t2_model.load_state_dict(checkpoint['model'],strict=True)
        for name,a in t2_model.adapters.items():a.merged=checkpoint['merged_flags'][name]
        evaluator=Pilot(t2_model,[None,test],manifest,loader,train_tf,test_tf,cfg,root/'runs/t2_eval',identity,clock)
        evaluator.s=copy.deepcopy(checkpoint['state']);restore_rng(checkpoint['rng'])
        _,_,t2=predictions(evaluator,1)
        del evaluator,t2_model;torch.cuda.empty_cache()
        predicted={};metrics={};decision=[]
        for mode,cls in (('control',Pilot),('mainline',MainlinePilot)):
            evaluator=cls(model_for(cfg),[None,test],manifest,loader,train_tf,test_tf,cfg,branch_paths[mode],identity,clock)
            evaluator.restore();y,pred,per_task=predictions(evaluator,2);predicted[mode]=(y,pred)
            if mode=='mainline':decision=copy.deepcopy(evaluator.s['mainline_server_records'])
            del evaluator;torch.cuda.empty_cache()
        y0,p0=predicted['control'];y1,p1=predicted['mainline']
        if not np.array_equal(y0,y1):raise ValueError('paired evaluation order mismatch')
        metrics['control']=compare(y0,p0,p0,cfg['cpt'],t1_accuracy,t2)
        metrics['mainline']=compare(y1,p1,p0,cfg['cpt'],t1_accuracy,t2)
        metrics['mainline']['wrong_to_different_wrong']=int(((p0!=y0)&(p1!=y1)&(p0!=p1)).sum())
        metrics['control']['wrong_to_different_wrong']=0
        base,main=metrics['control'],metrics['mainline']
        checks={'raw_net_gain_at_least_6':bool(main['corrected_vs_zero']-main['broken_vs_zero']>=6),
                'old_correct_not_lower':bool(main['old_correct']>=base['old_correct']),
                'new_correct_at_most_10_lower':bool(main['new_correct']>=base['new_correct']-10),
                'forgetting_not_higher':bool(main['forgetting']<=base['forgetting']+1e-12),
                'at_least_one_server_update_accepted':bool(any(r['decision']['accepted'] for r in decision))}
        write(report/'RESULT.json',{'status':'completed','seed':0,'exploratory_gate_passed':all(checks.values()),
              'checks':checks,'metrics':metrics,'t1_reference_accuracy':t1_accuracy,'t2_reference_accuracies':t2,
              'mainline_server_records':decision,'new_training_visits':200250,
              'shared_rrs_forward_images':4005,'branch_gram_forward_images':40050,
              'mainline_probe_forward_images':1405,'mainline_probe_backward_images':2810,
              'fit_image_reads':245710,'heldout_reads':0,'old_train_images':0,'test_images':8000,
              'test_opened_after_all_training':True,'branch_fit_reads':fit_counts,'branch_logical_communication_bytes':branch_comm,
              'source_t1_t2_training_visits':receipt['source_training_visits'],
              'elapsed_seconds':time.time()-clock['started'],'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),
              'limitations':['single seed and only T3; not final FCL method or statistical significance',
                  'fit data seen in other research; unmeasured pretraining overlap',
                  'final client momenta are not common-anchor gradients or full local trajectory bias',
                  'historical proxy and current input probes do not guarantee old class safety',
                  'theoretical added communication is not a network measurement']})
    finally:raw.close()


if __name__=='__main__':main()
