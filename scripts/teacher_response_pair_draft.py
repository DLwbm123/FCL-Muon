"""Unregistered teacher-response Muon server study; do not run before freeze."""
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
from research.continual_muon.teacher_server_pilot_draft import TeacherServerPilot
from research.continual_muon.paired_metrics import compare
from scripts.paired_boundary_worker import model_for,predictions
from scripts.real_boundary_worker import same_rng,write


def attach_model(cfg,checkpoint):
    model=model_for(cfg)
    for _ in range(checkpoint['head_count']):model.add_head(cfg['cpt'])
    model.load_state_dict(checkpoint['model'],strict=True)
    for name,a in model.adapters.items():a.merged=checkpoint['merged_flags'][name]
    return model


def main():
    root,source,endpoints,t2root=[Path(os.environ[k]) for k in ('PROJECT_ROOT','SOURCE_ROOT','ENDPOINT_ROOT','T2_ENDPOINT_ROOT')]
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
    start=torch.load(endpoints/'runs/shared/latest.pt',map_location='cpu',weights_only=False)
    control=torch.load(endpoints/'runs/control/latest.pt',map_location='cpu',weights_only=False)
    t2=torch.load(t2root/'runs/t2_endpoint_private.pt',map_location='cpu',weights_only=False)
    old_result=json.loads((endpoints/'reports/RESULT.json').read_text())
    if (start['schema']!='cosine-head-regmean-pilot-1' or start['clock']!=control['clock'] or
            start['identity']!=control['identity'] or start['config']!=control['config'] or
            start['identity']!=t2['identity'] or clock==start['clock'] or
            (start['state']['task'],start['state']['round'],start['state']['phase'])!=(2,0,'round_start') or
            (control['state']['task'],control['state']['round'],control['state']['phase'])!=(2,5,'ready_evaluation') or
            (t2['state']['task'],t2['state']['round'],t2['state']['phase'])!=(1,5,'ready_evaluation') or
            start['head_count']!=3 or control['head_count']!=3 or t2['head_count']!=2 or
            start['state']['history'].committed!={0,1} or len(start['state']['pending'])!=12 or
            set(start['state']['client_states'])!=set(range(10))):
        raise ValueError('source identity or lifecycle mismatch')
    history=start['state']['history'].raw;past=t2['state']['history'].raw;pending=t2['state']['pending']
    if set(history)!=set(past) or set(history)!=set(pending) or any(not torch.equal(history[k],past[k]+pending[k]) for k in history):
        raise ValueError('source T2-to-T3 historical state mismatch')
    bkeys=[k for k in start['model'] if k.endswith(('.Bq','.Bv'))]
    if len(bkeys)!=24 or any(torch.count_nonzero(start['model'][k]) for k in bkeys):
        raise ValueError('source T3 adapter start mismatch')
    cfg=copy.deepcopy(start['config'])
    expected={'seed':0,'dataset':'seq-cifar100','alpha':.05,'clients':10,'rounds':5,'epochs':5,
              'batch':16,'rank':32,'lr':.0002,'aggregation':'regmean','selection_mode':'prefix'}
    if any(cfg.get(k)!=v for k,v in expected.items()):raise ValueError('source configuration mismatch')
    manifest=json.loads((source/cfg['manifest']).read_text());weights=json.loads((source/'weights.json').read_text())
    identity={'manifest':cfg['manifest'],'weight_identity':weights['sha256'],
              'source_revision':cfg['source_revision'],'purpose':'three-task aggregation-utility selection'}
    if start['identity']!=identity or manifest['identity']!={k:cfg[k] for k in ('dataset','seed','alpha','clients')}:
        raise ValueError('source manifest identity mismatch')
    original=manifest['partitions']['train'];fit,heldout=split_clients(original[2],0)
    records=[v for v in control['state']['training_records'] if v['task']==2]
    if ((sum(map(len,fit)),sum(map(len,heldout)))!=(4005,995) or len(records)!=50 or
            sum(v['visited'] for v in records)!=100125 or any(v['N']!=len(fit[v['client']]) for v in records) or
            {(v['round'],v['client']) for v in records}!={(r,k) for r in range(5) for k in range(10)}):
        raise ValueError('control data split or train receipt mismatch')
    bases=[k for k in start['model'] if k.endswith('.A')]
    if len(bases)!=12 or any(not torch.equal(start['model'][k],control['model'][k]) for k in bases):
        raise ValueError('shared task basis mismatch')
    groups=copy.deepcopy(original);groups[2]=fit
    train_manifest={'identity':manifest['identity'],'partitions':{'train':groups},'labels':{'train':manifest['labels']['train']}}
    indices={t:[i for g in original[t] for i in g] for t in (1,2)}
    raw=CurrentCIFAR(source/'data/cifar-100-python/train',train_manifest['labels']['train'],indices)
    taskguard=CurrentTaskGuard(loader,raw,indices);raw.enter_t3();taskguard.enter_t3()
    guard=GuardedLoader(taskguard,raw,[i for g in fit for i in g],[i for g in heldout for i in g])
    train_tf,test_tf=transforms();train_tf=train_tf.cuda();test_tf=test_tf.cuda()
    try:
        shared=Pilot(attach_model(cfg,start),[raw,None],train_manifest,guard,train_tf,test_tf,cfg,root/'runs/shared',identity,clock)
        teacher=attach_model(cfg,t2).eval()
        for parameter in teacher.parameters():parameter.requires_grad_(False)
        shared.s=copy.deepcopy(start['state']);restore_rng(start['rng'])
        if len(start['rng']['cuda'])!=1 or not torch.equal(torch.cuda.get_rng_state(),start['rng']['cuda'][0]):
            raise ValueError('source GPU RNG mismatch')
        initial_rng=rng_state();i=torch.arange(768,dtype=torch.float32)[:,None]+1;j=torch.arange(32,dtype=torch.float32)[None,:]+1
        q=torch.linalg.qr(torch.sin(i*j*.013),mode='reduced').Q
        err=float(torch.linalg.vector_norm(mulora_ssb(q)-mulora_ssb(q.cuda()).cpu()))
        buffers=[{k:v.detach().clone() for k,v in m.named_buffers()} for m in (shared.model,teacher)]
        with torch.no_grad():
            synthetic=torch.zeros(1,3,224,224,device=shared.device)
            old=teacher(synthetic);current=shared.model(synthetic)
            if old.shape!=(1,2*cfg['cpt']) or current.shape!=(1,3*cfg['cpt']):raise ValueError('preflight head mismatch')
            old_diff=float((old-current[:,:2*cfg['cpt']]).abs().max())
        if (not np.isfinite(err) or err>1e-4 or not np.isfinite(old_diff) or old_diff>1e-4 or
                not same_rng(initial_rng,rng_state()) or any(not torch.equal(buffers[i][k],v)
                for i,m in enumerate((shared.model,teacher)) for k,v in m.named_buffers())):
            raise ValueError('GPU SSB or teacher source preflight failed')
        write(report/'REAL_PREFLIGHT.json',{'passed':True,'cpu_gpu_ssb_l2':err,'source_old_logits_max_abs_diff':old_diff,
              'synthetic_teacher_forward_images':1,'synthetic_student_forward_images':1,'rng_unchanged':True,'dataset_images':0})
        shared.save();shared_path=shared.out/'latest.pt'
        branch_paths={};communications={};receipts={};branch_reads={}
        for mode in ('full','no_consensus','kl_only'):
            out=root/'runs'/mode;out.mkdir();shutil.copyfile(shared_path,out/'latest.pt')
            pilot=TeacherServerPilot(model_for(cfg),[raw,None],train_manifest,guard,train_tf,test_tf,cfg,out,identity,clock,
                                     teacher=teacher,heldout_groups=heldout,teacher_variant=mode)
            pilot.restore()
            if not same_rng(initial_rng,rng_state()):raise ValueError('branch initial RNG mismatch')
            pilot.s['teacher_variant']=mode;pilot.save()
            if not same_rng(initial_rng,rng_state()):raise ValueError('branch initialization changed RNG')
            before_reads=guard.training_access_images;before_heldout=guard.validation_access_images
            before_comm=copy.deepcopy(pilot.s['communication'])
            while pilot.s['phase']!='ready_evaluation':
                if pilot.s['task']!=2:raise ValueError('unexpected branch task')
                pilot.step()
                write(report/'PROGRESS.json',{'time':time.time(),'stage':mode,'round':pilot.s['round'],
                      'phase':pilot.s['phase'],'current_train_reads':guard.training_access_images,
                      'heldout_reads':guard.validation_access_images})
            train=[v for v in pilot.s['training_records'] if v['task']==2]
            if len(train)!=50 or sum(v['visited'] for v in train)!=100125:raise ValueError('branch train count mismatch')
            reads=guard.training_access_images-before_reads
            if reads!=122960 or guard.validation_access_images-before_heldout!=2360 or guard.training_validation_accesses:
                raise ValueError('branch data isolation mismatch')
            branch_reads[mode]=reads;branch_paths[mode]=out
            communications[mode]={k:int(v-before_comm.get(k,0)) for k,v in pilot.s['communication'].items()}
            receipts[mode]=copy.deepcopy(pilot.s.get('teacher_server_records',[]))
            if len(receipts[mode])!=5 or any(r['teacher_variant']!=mode or r['gradient_cost']['samples']!=281 or
                    r['fit_candidate_cost']['samples']!=281 or r['confirmation_baseline_cost']['samples']!=236 or
                    r['confirmation_selected_cost']['samples']!=236 or
                    r['confirmation_reads_after']-r['confirmation_reads_before_selection']!=472 for r in receipts[mode]):
                raise ValueError('teacher gate accounting mismatch')
            del pilot;torch.cuda.empty_cache()
        if (raw.reads!={1:0,2:375960} or raw.reads!=taskguard.visits or guard.validation_access_images!=7080 or
                guard.training_access_images!=368880 or guard.allow_validation or guard.training_validation_accesses):
            raise ValueError('all-arm data accounting mismatch')
        # The baseline is read-only and was previously evaluated; no test image opened in this job yet.
        test=CIFAR100(source/'data',train=False,download=False)
        if list(test.targets)!=manifest['labels']['test']:raise ValueError('test labels mismatch')
        t2model=attach_model(cfg,t2)
        evaluator=Pilot(t2model,[None,test],manifest,loader,train_tf,test_tf,cfg,root/'runs/t2_eval',identity,clock)
        evaluator.s=copy.deepcopy(t2['state']);restore_rng(t2['rng']);_,_,t2_scores=predictions(evaluator,1)
        t1_accuracy=t2['state']['matrix'][0][0];del evaluator,t2model;torch.cuda.empty_cache()
        cmodel=attach_model(cfg,control)
        evaluator=Pilot(cmodel,[None,test],manifest,loader,train_tf,test_tf,cfg,root/'runs/control_eval',identity,clock)
        evaluator.s=copy.deepcopy(control['state']);restore_rng(control['rng']);y0,p0,_=predictions(evaluator,2)
        del evaluator,cmodel;torch.cuda.empty_cache()
        metrics={'control':compare(y0,p0,p0,cfg['cpt'],t1_accuracy,t2_scores)}
        for name in ('raw_accuracy','old_accuracy','new_accuracy','forgetting'):
            if abs(metrics['control'][name]-old_result['metrics']['control'][name])>1e-12:
                raise ValueError('reused control evaluation not numerically consistent')
        for mode in ('full','no_consensus','kl_only'):
            evaluator=Pilot(model_for(cfg),[None,test],manifest,loader,train_tf,test_tf,cfg,branch_paths[mode],identity,clock)
            evaluator.restore();y,p,_=predictions(evaluator,2)
            if not np.array_equal(y,y0):raise ValueError('paired test order mismatch')
            metrics[mode]=compare(y,p,p0,cfg['cpt'],t1_accuracy,t2_scores)
            metrics[mode]['wrong_to_different_wrong']=int(((p0!=y0)&(p!=y)&(p0!=p)).sum())
            del evaluator;torch.cuda.empty_cache()
        base,full=metrics['control'],metrics['full']
        checks={'raw_net_gain_at_least_6':bool(full['corrected_vs_zero']-full['broken_vs_zero']>=6),
                'old_correct_not_lower':bool(full['old_correct']>=base['old_correct']),
                'new_correct_at_most_10_lower':bool(full['new_correct']>=base['new_correct']-10),
                'forgetting_not_higher':bool(full['forgetting']<=base['forgetting']+1e-12),
                'at_least_one_round_accepted':bool(any(r['decision']['accepted'] for r in receipts['full'])),
                'raw_not_below_no_consensus':bool(full['raw_accuracy']+1e-12>=metrics['no_consensus']['raw_accuracy']),
                'raw_not_below_kl_only':bool(full['raw_accuracy']+1e-12>=metrics['kl_only']['raw_accuracy'])}
        write(report/'RESULT.json',{'status':'completed','seed':0,'exploratory_gate_passed':all(checks.values()),
              'checks':checks,'metrics':metrics,'t1_reference_accuracy':t1_accuracy,'t2_reference_accuracies':t2_scores,
              'server_records':receipts,'new_training_visits':300375,'new_gram_forward_images':60075,
              'new_rrs_forward_images':0,'shared_source_rrs_forward_images':4005,
              'fit_image_reads':368880,'heldout_reads':7080,'old_train_images':0,
              'teacher_forward_images':15510,'student_probe_forward_images':15510,'student_probe_backward_images':4215,
              'teacher_broadcast_bytes_per_client_assumed':sum(v.numel()*v.element_size() for v in teacher.state_dict().values()),
              'test_images':14000,'test_opened_after_all_new_training':True,'branch_fit_reads':branch_reads,
              'new_branch_logical_communication_bytes':communications,
              'source_control_logical_communication_bytes':old_result['branch_logical_communication_bytes']['control'],
              'reused_control_raw_consistency_checked':True,
              'elapsed_seconds':time.time()-clock['started'],'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),
              'limitations':['single seed, T3 only; not final FCL or statistical significance',
                  'reused control and test were previously evaluated; not independently blind',
                  'history/current moments and client update dispersion do not guarantee old-class safety',
                  'teacher response on current images does not guarantee true old-class safety',
                  'theoretical added communication is not a network measurement']})
    finally:raw.close()


if __name__=='__main__':main()
