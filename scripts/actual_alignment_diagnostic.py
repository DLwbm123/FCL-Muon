"""One bounded common-state round per registered seed; no test inference."""
import gc,json,os,signal,time
from pathlib import Path
import torch
from torchvision.datasets import CIFAR100
from fcl.checkpoint import atomic_save,restore_rng
from fcl.data import loader,transforms
from research.fed_spectra.actual_update import ActualUpdatePilot
from research.fed_spectra.alignment_diagnostic import measure,select_probes
from research.fed_spectra.lifecycle import write
from scripts.spectra_train import initialize_rng,new_model


class DiagnosticPilot(ActualUpdatePilot):
    def budget(self):
        super().budget()
        if time.time()-getattr(self,'storage_checked',0)>10:
            self.storage_checked=time.time();used=0
            for directory,_,files in os.walk(self.storage_root,followlinks=False):
                for name in files:
                    path=Path(directory)/name
                    try:
                        if not path.is_symlink():used+=path.stat().st_blocks*512
                    except FileNotFoundError:pass
            if used>=10_000_000_000:raise RuntimeError('Diagnostic write stop')
    def save(self):self.budget();super().save()
    def ids(self,split,*args):
        if split!='train':raise AssertionError('Test access prohibited')
        return super().ids(split,*args)
    def evaluate(self):raise AssertionError('Test inference prohibited')


def run_seed(root,seed,source,clock):
    initialize_rng(seed);x=torch.load(source/'runs/control/task3.pt',map_location='cpu',weights_only=False)
    assert x['schema']=='cosine-head-regmean-pilot-1' and x['config']['seed']==seed
    assert x['state']['task']==3 and x['state']['phase']=='task_start' and x['head_count']==3
    assert x['config']['actual_update']['operator']['eta']==0 and len(x['state']['training_records'])==150
    cfg=json.loads(json.dumps(x['config']));cfg['actual_update']['operator']['eta']=.1
    cfg.update(output=f'runs/seed{seed}',maximum_seconds=6900,storage_limit_bytes=12000000000,write_stop_bytes=10000000000)
    manifest=json.loads((source/cfg['manifest']).read_text());assert manifest['identity']=={k:cfg[k] for k in ['dataset','seed','alpha','clients']}
    # Keep only training partitions/labels in the diagnostic's data interface.
    manifest={'identity':manifest['identity'],'partitions':{'train':manifest['partitions']['train']},'labels':{'train':manifest['labels']['train']}}
    raw=CIFAR100(source/'data',train=True,download=False);assert list(raw.targets)==manifest['labels']['train']
    train_tf,test_tf=transforms();train_tf=train_tf.cuda();test_tf=test_tf.cuda()
    model=new_model(cfg,json.loads((source/'weights.json').read_text()))
    for _ in range(x['head_count']):model.add_head(cfg['cpt'])
    model.load_state_dict(x['model'],strict=True)
    for name,a in model.adapters.items():a.merged=x['merged_flags'][name]
    assert all(a.merged for a in model.adapters.values())
    p=DiagnosticPilot(model,[raw,None],manifest,loader,train_tf,test_tf,cfg,root/cfg['output'],{'diagnostic_seed':seed,'source_task':3},clock)
    p.storage_root=root;p.s=x['state'];restore_rng(x['rng']);del x
    probes=select_probes(manifest['partitions']['train'][3],seed)
    (p.out/'probe_ids_private.json').write_text(json.dumps(probes))
    initial_visits=sum(v['visited'] for v in p.s['training_records']);initial_teacher=p.s['anchor_forward_images'];initial_stats=sum(v['samples'] for v in p.s['stats_records'])
    while p.s['phase']!='local_done':
        assert p.s['task']==3 and p.s['round']==0;p.step()
        write(root/'reports/PROGRESS.json',{'seed':seed,'phase':p.s['phase'],'new_client_rounds':len(p.s['training_records'])-150,'new_training_visits':sum(v['visited'] for v in p.s['training_records'])-initial_visits,'time':time.time()})
    assert len(p.s['training_records'])==160 and sum(v['visited'] for v in p.s['training_records'])-initial_visits==25000
    baseline=[v.clone() for v in p.s['agg_B']];origin=[b.detach().cpu().clone() for b in p.model.b_parameters()]
    p.fuse();corrected=[b.detach().cpu().clone() for b in p.model.b_parameters()]
    assert p.s['task']==3 and p.s['round']==1
    head=p.model.rows[-1].detach().clone();old_heads=[h.detach().clone() for h in p.model.rows[:-1]]
    def deploy(values):
        with torch.no_grad():
            for b,v in zip(p.model.b_parameters(),values):b.copy_(v)
    deploy(baseline);base,grads=measure(p,probes,gradient=True)
    deploy(corrected);changed,_=measure(p,probes)
    deploy(baseline);repeat,_=measure(p,probes)
    assert all(abs(base['weighted'][k]-repeat['weighted'][k])<=1e-6 for k in base['weighted'])
    assert torch.equal(head,p.model.rows[-1]) and all(torch.equal(a,b) for a,b in zip(old_heads,p.model.rows[:-1]))
    delta=[b.double()-a.double() for a,b in zip(baseline,corrected)]
    dot=sum(float((g*d).sum()) for g,d in zip(grads,delta));gn=sum(float(g.square().sum()) for g in grads)**.5;dn=sum(float(d.square().sum()) for d in delta)**.5
    operator=p.s['actual_update_rounds'][-1]
    rr=[v for pair in operator['layers'].values() for v in pair]
    assert len(rr)==24 and all(v['actual_FP32_norm_relative_error']<=5e-4 for v in rr)
    p.budget();atomic_save({'baseline_B':baseline,'corrected_B':corrected,'origin_B':origin,'head':head.cpu(),'operator':operator},p.out/'pair_private.pt')
    result={'seed':seed,'status':'COMPLETED','source_tasks_reused':3,'new_client_rounds':10,'new_training_visits':25000,'source_read_only':True,
        'task':4,'round':1,'test_images_used':0,'probe_selection':'fixed up to 32 per current training client; training-internal, not holdout',
        'baseline':base,'corrected':changed,'repeated_baseline':repeat,
        'weighted_difference':{k:changed['weighted'][k]-base['weighted'][k] for k in base['weighted']},
        'gradient_dot_correction':dot,'gradient_norm':gn,'correction_norm':dn,'gradient_correction_cosine':dot/(gn*dn) if gn*dn else None,
        'operator':operator,'head_unchanged':True,'baseline_repeat_tolerance':1e-6,
        'new_training_teacher_forward_images':p.s['anchor_forward_images']-initial_teacher,
        'new_stats_forward_images':sum(v['samples'] for v in p.s['stats_records'])-initial_stats,
        'probe_student_forward_images':sum(v['student_forward_images'] for v in [base,changed,repeat]),
        'probe_teacher_forward_images':sum(v['teacher_forward_images'] for v in [base,changed,repeat]),
        'probe_backward_images':base['backward_images'],'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),
        'elapsed_seconds_from_common_start':time.time()-clock['started']}
    write(root/f'reports/seed{seed}.json',result)
    del p,model,raw,train_tf,test_tf;gc.collect();torch.cuda.empty_cache()
    return result


def main():
    root=Path(os.environ['PROJECT_ROOT']);origin=json.loads((root/'reports/ORIGIN.json').read_text())
    clock={'started':origin['started'],'deadline':origin['started']+6900}
    assert json.loads((root/'reports/CPU_TESTS.json').read_text())['passed']
    assert not (root/'reports/RESULT.json').exists()
    def stop(*_):raise TimeoutError('External diagnostic deadline or stop')
    signal.signal(signal.SIGTERM,stop)
    rows=[]
    for seed,source in enumerate(json.loads((root/'sources_private.json').read_text())):
        if time.time()>=clock['deadline']:raise TimeoutError('Common clock exhausted')
        torch.cuda.reset_peak_memory_stats();rows.append(run_seed(root,seed,Path(source),clock))
    worse=[r['seed'] for r in rows if r['weighted_difference']['loss']>1e-6 and r['gradient_dot_correction']>1e-6]
    better=[r['seed'] for r in rows if r['weighted_difference']['loss']< -1e-6]
    write(root/'reports/RESULT.json',{'status':'COMPLETED','seeds':[0,1,2],'new_training_visits':75000,'test_images_used':0,
        'same_point_misalignment_seeds':worse,'loss_improved_seeds':better,'local_misalignment_supported':len(worse)>=2,
        'training_objective_improvement_at_boundary':len(better)>=2,'elapsed_seconds':time.time()-clock['started'],
        'interpretation':'Local training-input diagnostic only. Does not explain generalization causally or overturn failed T5 gate.',
        'per_seed':rows})


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        write(Path(os.environ['PROJECT_ROOT'])/'reports/FAILURE.json',{'error':repr(exc),'time':time.time()});raise
