"""One fixed wire approximation at saved endpoints; no training or test reads."""
import gc,json,os,signal,time
from pathlib import Path
import torch
from torchvision.datasets import CIFAR100
from fcl.checkpoint import restore_rng
from fcl.data import loader,transforms
from scripts.spectra_train import initialize_rng,new_model
from scripts.matrix_tangent_diagnostic import ReadOnlyPilot
from research.fed_spectra.alignment_diagnostic import measure,select_probes
from research.fed_spectra.competition_diagnostic import measure_competition
from research.fed_spectra.matrix_tangent import tangent_candidate
from research.fed_spectra.all_seen_diagnostic import evaluate_candidate
from research.fed_spectra.heldout_round import GuardedLoader
from research.fed_spectra.rank8_transport import encode_decode,conditions
from research.fed_spectra.lifecycle import write


def run(root,seed,control,old,clock):
    initialize_rng(seed)
    x=torch.load(old/f'runs/seed{seed}/latest.pt',map_location='cpu',weights_only=False)
    pair=torch.load(old/f'runs/seed{seed}/pair_private.pt',map_location='cpu',weights_only=False)
    assert x['config']['seed']==seed and x['head_count']==4 and x['state']['task']==3
    assert x['state']['phase']=='local_done' and x['state']['round']==0 and len(x['state']['training_records'])==160
    cfg=json.loads(json.dumps(x['config']));cfg['output']=f'private/seed{seed}'
    manifest=json.loads((old/f'runs/seed{seed}/manifest_private.json').read_text())
    split=json.loads((old/f'runs/seed{seed}/split_private.json').read_text())
    train,validation=split['training'],split['validation']
    assert manifest['partitions']['train'][3]==train
    raw=CIFAR100(control/'data',train=True,download=False);assert list(raw.targets)==manifest['labels']['train']
    guard=GuardedLoader(loader,raw,sum(train,[]),sum(validation,[]))
    train_tf,test_tf=transforms();train_tf=train_tf.cuda();test_tf=test_tf.cuda()
    model=new_model(cfg,json.loads((control/'weights.json').read_text()))
    for _ in range(4):model.add_head(cfg['cpt'])
    model.load_state_dict(x['model'],strict=True)
    for name,a in model.adapters.items():a.merged=x['merged_flags'][name]
    assert all(not a.merged for a in model.adapters.values())
    p=ReadOnlyPilot(model,[raw,None],manifest,guard,train_tf,test_tf,cfg,root/cfg['output'],{'seed':seed},clock)
    p.storage_root=root;p.s=x['state'];restore_rng(x['rng']);del x
    origin,baseline,positive=pair['origin_B'],pair['baseline_B'],pair['positive_B']
    assert all(torch.equal(a.detach().cpu(),b) for a,b in zip(model.b_parameters(),origin))
    assert all(torch.equal(a,b) for a,b in zip(p.s['agg_B'],baseline))
    with torch.no_grad():model.rows[-1].copy_(pair['head'])
    p.get_teacher();hook=model.register_forward_pre_hook(guard.model_guard)
    heads=[h.detach().clone() for h in model.rows]
    def deploy(values):
        with torch.no_grad():
            for b,v in zip(model.b_parameters(),values):b.copy_(v)
    selection=select_probes(train,seed);assert not set(sum(selection,[]))&set(sum(validation,[]))
    deploy(baseline);full=[torch.zeros_like(b,dtype=torch.float64) for b in baseline];compressed=[v.clone() for v in full]
    records=[];grad_cost={'student_forward_images':0,'teacher_forward_images':0,'backward_images':0};svd_seconds=0.
    for k,ids in enumerate(selection):
        if not ids:continue
        probes=[[] for _ in train];probes[k]=ids
        m,weighted=measure(p,probes,True,full_client_gradients=True,all_seen_objective=True)
        weight=m['clients'][0]['weight']
        for key in grad_cost:grad_cost[key]+=m[key]
        for layer,(target,small,msg) in enumerate(zip(full,compressed,weighted)):
            client=(msg/weight).float()
            assert torch.equal(client.double()*weight,msg)
            then=time.perf_counter();reconstructed,rec=encode_decode(client);svd_seconds+=time.perf_counter()-then
            target.add_(msg);small.add_(reconstructed,alpha=weight)
            records.append(dict(rec,client=k,layer_QV=layer))
    full_ops=[tangent_candidate(o,b,v,g) for o,b,v,g in zip(origin,baseline,positive,full)]
    compressed_ops=[tangent_candidate(o,b,v,g) for o,b,v,g in zip(origin,baseline,positive,compressed)]
    gg=sum(float(g.square().sum()) for g in full);hh=sum(float(h.square().sum()) for h in compressed)
    relative_error=(sum(float((g-h).square().sum()) for g,h in zip(full,compressed))/max(gg,1e-60))**.5
    cosine=sum(float((g*h).sum()) for g,h in zip(full,compressed))/max((gg*hh)**.5,1e-60)
    del full,compressed,weighted,client,msg,reconstructed
    for a,b in zip(full_ops,compressed_ops):
        assert abs(a[1]['reference_angle']-b[1]['reference_angle'])<=1e-12
        assert a[1]['actual_FP32_norm_relative_error']<=5e-4 and b[1]['actual_FP32_norm_relative_error']<=5e-4
    prior=json.loads((old/f'reports/seed{seed}.json').read_text());refs={};evals=[];comp_evals=[];diffs=[]
    guard.allow_validation=True
    for name,probe,oldname in [('selection',selection,'selection'),('confirmation',validation,'validation')]:
        deploy(baseline);base=measure(p,probe)[0];bc=measure_competition(p,probe)[0]
        for value,ref in [(base,prior[f'{oldname}_baseline']),(bc,prior[f'{oldname}_baseline_competition'])]:
            diffs.append(max(abs(value['weighted'][k]-ref['weighted'][k]) for k in value['weighted']))
        refs[name]=(base,bc,base,bc);evals.append(base);comp_evals.append(bc)
    original=evaluate_candidate(p,selection,validation,baseline,[v[0] for v in full_ops],refs)
    for name,v in original['probes'].items():
        previous=prior['all_seen_candidate']['probes'][name]
        for key in ['objective','competition']:
            diffs.append(max(abs(v[key]['weighted'][k]-previous[key]['weighted'][k]) for k in v[key]['weighted']))
        if name=='confirmation':assert abs(v['all_seen_accuracy_delta_vs_baseline']-previous['all_seen_accuracy_delta_vs_baseline'])<=1e-10
        refs[name]=(refs[name][0],refs[name][1],v['objective'],v['competition'])
    assert max(diffs)<=1e-6,diffs
    result=evaluate_candidate(p,selection,validation,baseline,[v[0] for v in compressed_ops],refs)
    assert all(torch.equal(a.detach().cpu(),b) for a,b in zip(model.b_parameters(),baseline))
    assert all(torch.equal(a,b) for a,b in zip(heads,model.rows))
    assert guard.validation_grad_violations==guard.training_validation_accesses==0
    assert guard.validation_model_forward_images==10*sum(map(len,validation))
    fit=result['probes']['selection']
    cost={'student_forward_images':grad_cost['student_forward_images']+sum(v['student_forward_images'] for v in evals+comp_evals)+original['extra_student_forward_images']+result['extra_student_forward_images'],
      'teacher_forward_images':grad_cost['teacher_forward_images']+sum(v['teacher_forward_images'] for v in evals)+original['extra_teacher_forward_images']+result['extra_teacher_forward_images'],
      'student_backward_images':grad_cost['backward_images'],'client_SVDs':len(records),'SVD_seconds':svd_seconds}
    row={'seed':seed,'status':'COMPLETED','new_training_visits':0,'test_images_used':0,'new_checkpoints':0,
      'full_bytes':sum(v['full_bytes'] for v in records),'wire_bytes':sum(v['wire_bytes'] for v in records),
      'compression_records':records,'aggregate_gradient_relative_error':relative_error,'aggregate_gradient_cosine':cosine,
      'full_candidate':original,'compressed_candidate':result,'full_candidate_operators':[v[1] for v in full_ops],
      'compressed_candidate_operators':[v[1] for v in compressed_ops],'baseline_and_full_reproduction_max_difference':max(diffs),
      'selection_gate_passed':fit['combined_delta_vs_baseline']< -1e-6 and fit['original_objective_delta_vs_baseline']<=1e-6,
      'engineering_passed':True,'heads_unchanged':True,'baseline_restored':True,'validation_training_or_stats_accesses':0,
      'validation_grad_violations':0,'validation_student_forward_images_guard':guard.validation_model_forward_images,
      'cost':cost,'elapsed_from_common_start':time.time()-clock['started']}
    write(root/f'reports/seed{seed}.json',row)
    hook.remove();del p,model,pair,raw,guard,train_tf,test_tf;gc.collect();torch.cuda.empty_cache()
    return row


def main():
    root=Path(os.environ['PROJECT_ROOT']);o=json.loads((root/'reports/ORIGIN.json').read_text())
    clock={'started':o['started'],'deadline':o['started']+2400}
    assert json.loads((root/'reports/CPU_TESTS.json').read_text())['passed'] and not (root/'reports/RESULT.json').exists()
    def stop(*_):raise TimeoutError('Original diagnostic timeout or signal')
    signal.signal(signal.SIGTERM,stop)
    sources=json.loads((root/'sources_private.json').read_text());rows=[]
    for seed,source in enumerate(sources['controls']):
        write(root/'reports/PROGRESS.json',{'seed':seed,'phase':'loading','time':time.time()})
        rows.append(run(root,seed,Path(source),Path(sources['heldout']),clock))
    c=conditions(rows)
    write(root/'reports/RESULT.json',{'status':'COMPLETED','conditions':c,'rank8_local_fidelity_supported':all(c.values()),
      'new_training_visits':0,'test_images_used':0,'per_seed':rows,'elapsed_seconds':time.time()-clock['started'],
      'limitations':['Reused exploratory validation, not independent confirmation.','Endpoint compression fidelity is not full T5 benefit or old-class safety.','Logical payload and local SVD timing are not distributed network performance.']})


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        write(Path(os.environ['PROJECT_ROOT'])/'reports/FAILURE.json',{'error':repr(exc),'time':time.time()});raise
