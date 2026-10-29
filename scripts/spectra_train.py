"""One fresh, bounded ten-task arm with T3 and T10 reload qualification."""
import gc,json,os,random,runpy,time
from pathlib import Path
import numpy as np
import torch,timm
from safetensors.torch import load_file
from torchvision.datasets import CIFAR100
from fcl.data import loader,transforms
from fcl.model import MuLoRAFL
from research.fed_spectra.pilot import SpectraPilot
from research.fed_spectra.actual_update import ActualUpdatePilot,global_shape_update
from research.fed_spectra.lifecycle import qualify,write


def initialize_rng(seed=0):
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)


def new_model(cfg,weights):
    backbone=timm.create_model(cfg['backbone'],pretrained=False,num_classes=0)
    backbone.load_state_dict({k:v for k,v in load_file(weights['path']).items() if not k.startswith('head.')},strict=True)
    return MuLoRAFL(backbone,100,cfg['rank'],cfg['cosine_scale']).cuda()


def real_preflight(cfg,weights,raw,manifest,test_tf):
    if cfg['method']=='actual_update_global':
        model=new_model(cfg,weights);model.add_head(10);model.eval()
        generator=torch.Generator().manual_seed(73129);receipts=[];original=[];changed=[]
        for a in model.adapters.values():
            d=a.A.shape[1];r=cfg['rank']
            u=torch.linalg.qr(torch.randn(d,r,generator=generator)).Q
            a.start(u.T.to(a.A))
            for b in (a.Bq,a.Bv):
                ds=[torch.randn(d,r,generator=generator,dtype=torch.float64)*.0001 for _ in range(2)]
                baseline=(.4*ds[0]+.6*ds[1]).float()
                identity,_=global_shape_update(ds,[.4,.6],baseline,eta=0.)
                assert torch.equal(identity.float(),baseline)
                value,record=global_shape_update(ds,[.4,.6],baseline,**cfg['actual_update']['operator'])
                assert abs(float(value.float().norm())-float(baseline.norm()))/float(baseline.norm())<1e-5
                original.append(baseline);changed.append(value.float());receipts.append(record)
                with torch.no_grad():b.copy_(baseline)
        ids=[i for group in manifest['partitions']['train'][0] for i in group][:4]
        x,_=next(iter(loader(raw[0],ids,4)));x=test_tf(x.cuda())
        with torch.no_grad():
            before=model(x)
            for b,v in zip(model.b_parameters(),original):b.copy_(v)
            same=model(x)
            assert torch.equal(before,same)
            for b,v in zip(model.b_parameters(),changed):b.copy_(v)
            after=model(x)
        assert torch.isfinite(after).all()
        result={'passed':True,'method':cfg['method'],'images':len(ids),'eta_zero_logits_exact':True,
            'corrected_logits_max_difference':float((after-before).abs().max()),
            'maximum_norm_relative_error':max(v['norm_relative_error'] for v in receipts),
            'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),
            'scope':'real ViT GPU forward with synthetic actual-increment inputs; full local learning tested in CPU lifecycle and formal run'}
        if cfg.get('direction_selection'):
            from research.fed_spectra.direction_diagnostic import reflected_proposal,choose_direction
            with torch.no_grad():
                for b,v in zip(model.b_parameters(),original):b.copy_(v)
            model.zero_grad(set_to_none=True)
            z=model(x);z.square().mean().backward()
            grads=[b.grad.detach().cpu().double().clone() for b in model.b_parameters()]
            assert all(torch.isfinite(g).all() for g in grads) and sum(float(g.norm()) for g in grads)>0
            negative,checks=reflected_proposal([torch.zeros_like(b) for b in original],original,changed)
            sign,dots=choose_direction(grads,original,changed,negative)
            with torch.no_grad():
                for b,v in zip(model.b_parameters(),changed if sign=='positive' else negative):b.copy_(v)
                assert torch.isfinite(model(x)).all()
            result.update(direction_preflight=True,selected_direction=sign,directional_derivatives=dots,
                          reflection_checks=checks,gradient_images=len(ids),
                          direction_scope='real ViT synthetic-increment logit-squared gradient and reflection; CE/KL lifecycle tested separately')
            del z,grads
        if cfg.get('spectral_tangent'):
            from research.fed_spectra.spectral_tangent import tangent_candidate
            with torch.no_grad():
                for b,v in zip(model.b_parameters(),original):b.copy_(v)
            model.zero_grad(set_to_none=True)
            z=model(x);z.square().mean().backward();checks=[]
            with torch.no_grad():
                for b,baseline,reference in zip(model.b_parameters(),original,changed):
                    u,_,vh=torch.linalg.svd(baseline.double(),full_matrices=False)
                    modes=(u.T@b.grad.detach().cpu().double()@vh.T).diag()
                    value,record=tangent_candidate(torch.zeros_like(baseline),baseline,reference,(u*modes)@vh)
                    record.pop('mode_gradient',None);checks.append(record);b.copy_(value)
                assert torch.isfinite(model(x)).all()
            assert any(v['identity_reason'] is None for v in checks)
            result.update(tangent_preflight=True,tangent_checks=checks,gradient_images=len(ids),
                tangent_scope='real ViT synthetic-increment projected logit-squared gradient; client-local CE/KL aggregation tested separately')
            del z
        if cfg.get('matrix_tangent'):
            from research.fed_spectra.matrix_tangent import tangent_candidate
            with torch.no_grad():
                for b,v in zip(model.b_parameters(),original):b.copy_(v)
            model.zero_grad(set_to_none=True)
            z=model(x);z.square().mean().backward();checks=[]
            with torch.no_grad():
                for b,baseline,reference in zip(model.b_parameters(),original,changed):
                    value,record=tangent_candidate(torch.zeros_like(baseline),baseline,reference,b.grad.detach().cpu().double())
                    checks.append(record);b.copy_(value)
                assert torch.isfinite(model(x)).all()
            assert any(v['identity_reason'] is None for v in checks)
            result.update(matrix_tangent_preflight=True,matrix_tangent_checks=checks,gradient_images=len(ids),
                matrix_scope='real ViT synthetic-increment logit-squared gradient; client-local CE/KL aggregation checked separately')
            model.zero_grad(set_to_none=True);del z
        del model;gc.collect();torch.cuda.empty_cache();return result
    from research.fed_spectra.math import gaussian_map,orthogonal_pool,rebase_adapter,fisher_coefficients
    model=new_model(cfg,weights);model.add_head(10);model.add_head(10);model.eval()
    for name,a in model.adapters.items():
        d=a.A.shape[1];u=orthogonal_pool([],d,cfg['rank'],99+int(name))
        a.start(u.T.to(a.A))
        with torch.no_grad():
            a.Bq.copy_(gaussian_map(d,cfg['rank'],199+int(name)).to(a.Bq)*.001)
            a.Bv.copy_(gaussian_map(d,cfg['rank'],299+int(name)).to(a.Bv)*.001)
    ids=[i for ids in manifest['partitions']['train'][0] for i in ids][:4]
    x,y=next(iter(loader(raw[0],ids,4)));x=test_tf(x.cuda())
    weights_qkv=[a.base.weight for a in model.adapters.values()]
    for w in weights_qkv:w.requires_grad_(True)
    z=model(x);coeff=fisher_coefficients(z[:,:10],71,2.)
    gradients=torch.autograd.grad((coeff*z[:,:10]/2).sum()/2,weights_qkv)
    norms=[float(g.norm()) for g in gradients]
    assert all(np.isfinite(v) and v>0 for v in norms)
    before=z.detach();del gradients,z
    receipts=[]
    for name,a in model.adapters.items():
        u=orthogonal_pool([],a.A.shape[1],cfg['rank'],399+int(name))
        receipts.extend(rebase_adapter(a,u.cuda()))
    with torch.no_grad():after=model(x)
    difference=float((before-after).abs().max())
    assert difference<.002 and all(v['maximum_weight_roundoff']<1e-6 for v in receipts)
    result={'passed':True,'images':4,'old_fisher_layer_gradient_norms':norms,
            'rebase_maximum_logit_difference':difference,'rebase':receipts,
            'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated()}
    del model;gc.collect();torch.cuda.empty_cache()
    return result


def main():
    root=Path(os.environ['PROJECT_ROOT']);arm=os.environ.get('ARM','full');report=root/'reports'/arm
    prefix=os.environ.get('CONFIG_PREFIX','spectra')
    seed=int(os.environ.get('EXPERIMENT_SEED','0'))
    if seed not in (0,1,2):raise ValueError('Unregistered seed')
    if prefix not in ('spectra','actual_update','actual_gate','actual_direction','actual_tangent','actual_matrix'):raise ValueError('Unknown configuration family')
    cfg=json.loads((root/f'configs/{prefix}_{arm}_seed{seed}.json').read_text())
    clock=json.loads((root/'reports/job.json').read_text());clock={k:clock[k] for k in ('started','deadline')}
    if time.time()>=clock['deadline']:raise TimeoutError('Original deadline exhausted')
    assert cfg['seed']==seed
    initialize_rng(seed)
    manifest=json.loads((root/cfg['manifest']).read_text());weights=json.loads((root/'weights.json').read_text())
    assert manifest['identity']=={k:cfg[k] for k in ('dataset','seed','alpha','clients')}
    raw=[CIFAR100(root/'data',train=t,download=False) for t in (True,False)]
    assert [list(d.targets) for d in raw]==[manifest['labels']['train'],manifest['labels']['test']]
    train_tf,test_tf=transforms();train_tf=train_tf.cuda();test_tf=test_tf.cuda()
    if os.environ.get('PREFLIGHT_ONLY')=='1':
        write(root/'reports/REAL_PREFLIGHT.json',real_preflight(cfg,weights,raw,manifest,test_tf));return
    source=json.loads((root/'reports/SOURCE.json').read_text())
    identity={'manifest':cfg['manifest'],'weight_identity':weights['sha256'],
              'source_revision':source['commit'],'method':cfg['method'],'purpose':'revised spectral study'}
    model=new_model(cfg,weights)
    pilot_class=ActualUpdatePilot if cfg['method']=='actual_update_global' else SpectraPilot
    if cfg.get('training_gate'):
        from research.fed_spectra.loss_gate import GatedActualUpdatePilot
        pilot_class=GatedActualUpdatePilot
    p=pilot_class(model,raw,manifest,loader,train_tf,test_tf,cfg,root/cfg['output'],identity,clock)
    if (p.out/'latest.pt').exists():
        assert os.environ.get('RESUME_RUN')=='1','No implicit restart';p.restore()
    torch.cuda.reset_peak_memory_stats()
    worker_started=time.time()
    def progress():
        write(report/'PROGRESS.json',{'time':time.time(),'method':cfg['method'],'task':p.s['task'],
              'round':p.s['round'],'phase':p.s['phase'],'next_client':p.s['next_client'],
              'client_rounds':len(p.s['training_records']),'training_visits':sum(v['visited'] for v in p.s['training_records']),
              'stages':[{'task':v['task']+1,'raw':v['all_seen'],'centered':v['centered']['all_seen_accuracy']} for v in p.s['stages']],
              'geometry_rounds':len(p.s['spectra_geometry']),'transport_rounds':len(p.s['spectra_rounds']),
              'actual_update_rounds':len(p.s.get('actual_update_rounds',[])),
              'accepted_gate_rounds':sum(v['accepted'] for v in p.s.get('training_gate_records',[])),
              'worker_started':worker_started,'original_clock':clock,
              'qualified_milestones':[n for n in cfg['milestones'] if (p.out/f'QUALIFICATION_T{n}.json').exists()]})
    progress()
    while p.step():
        n=p.s['task']
        if n in cfg['milestones'] and p.s['phase'] in ('task_start','complete') and not (p.out/f'QUALIFICATION_T{n}.json').exists():
            qualify(p,n)
            if os.environ.get('REFERENCE_OUTPUT'):
                os.environ.update(PREVIOUS_OUTPUT=os.environ['REFERENCE_OUTPUT'],CURRENT_OUTPUT=str(p.out),
                                  TASK_COUNT=str(n),PAIR_REPORT=f'reports/{arm}/PAIRED_VS_A_T{n}.json')
                runpy.run_path(str(root/'scripts/pair_boundary_outputs.py'),run_name='__main__')
        progress()
    assert (p.out/'COMPLETE.json').exists()
    write(report/'COMPLETE.json',{'qualified':True,'tasks':cfg['tasks'],'completed':time.time(),
                                'worker_elapsed_seconds':time.time()-worker_started})


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        root=Path(os.environ['PROJECT_ROOT']);arm=os.environ.get('ARM','full')
        report=root/'reports'/arm;report.mkdir(exist_ok=True)
        write(report/'FAILURE.json',{'time':time.time(),'error':repr(exc),'preflight':os.environ.get('PREFLIGHT_ONLY')=='1'})
        raise
