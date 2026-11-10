"""Real ViT, synthetic two-task proof of shared-trajectory/state isolation."""
import copy,gc,json,os
import numpy as np
import torch
from fcl.checkpoint import rng_state,restore_rng
from .runtime import setup,build,validate
from .frozen import step_frozen,restore_frozen
from research.adamw_t10.storage import write


def main():
    setup();root,f=validate();clock={k:f[k] for k in ('started','deadline')}
    generator=torch.Generator().manual_seed(772009)
    raw=[(torch.randn(3,224,224,generator=generator),torch.tensor(i)) for i in range(20)]
    parts=[[[i for i in range(t*10,(t+1)*10) if i%2==k] for k in range(2)] for t in range(2)]
    manifest={'labels':{s:list(range(20)) for s in ('train','test')},'partitions':{s:parts for s in ('train','test')}}
    def loader(raw,ids,batch,shuffle=False):
        ids=[ids[i] for i in (torch.randperm(len(ids)).tolist() if shuffle else range(len(ids)))]
        return [tuple(torch.stack(v) for v in zip(*(raw[j] for j in ids[i:i+batch]))) for i in range(0,len(ids),batch)]
    reference=None;results=[]
    for modes in [('identity','full','mean_only'),('identity',),('full',),('mean_only',)]:
        os.environ['ARM']='rrs_shared';p=build(clock,synthetic=([raw,raw],manifest,loader))
        p.cfg=copy.deepcopy(p.cfg);p.cfg.update(clients=2,epochs=1,rounds=1,batch=2,tasks=2,branch_modes=list(modes),primary_mode='full' if len(modes)>1 else modes[0],diagnostic_tasks=[])
        p.s['stat_branches']={mode:None for mode in modes}
        p.out=root/('src/runs/check_'+str(f.get('entry_revision',0))+'_'+('_'.join(modes)));p.out.mkdir(exist_ok=False)
        while not (p.s['task']==1 and p.s['phase']=='task_start'):p.step()
        p.step()
        if modes==('identity',):
            bases={k:a.A.detach().clone().requires_grad_(True) for k,a in p.model.adapters.items()}
            gram=[[(x.cuda(),y.cuda()) for x,y in loader(raw,parts[1][k],2)] for k in range(2)]
            support=[[gram[k][:2] for k in range(2)] for _ in range(2)]
            before=rng_state();args=(p.model,bases,support,gram,gram,[5,5],[3,3],'adamw',.0002,5,5,1,10)
            with torch.autograd.graph.save_on_cpu(pin_memory=False):
                tracked=p.functional_response(*args,track_response=True,**p.response_options)
                gradients=torch.autograd.grad(tracked['loss'],list(bases.values()))
            restore_rng(before);direct=p.functional_response(*args,track_response=False,**p.response_options)
            le=abs(float(tracked['loss'].detach()-direct['loss']));be=max(float((a.detach()-b.detach()).abs().max()) for a,b in zip(tracked['B'],direct['B']))
            if le>1e-5 or be>2e-6 or any(not torch.isfinite(g).all() for g in gradients):raise ArithmeticError('identity response mismatch')
            del tracked,direct,gradients;restore_rng(before)
            response=dict(loss_track_direct_max_abs=le,B_track_direct_max_abs=be,finite_gradients=True)
        while p.s['phase']!='complete':p.step()
        p.restore();p.s['task']=1;readout=p.evaluate(False);p.s['task']=2
        if readout!=p.s['stages'][-1]:raise ArithmeticError('branch recovery mismatch')
        current={k:v.detach().cpu().clone() for k,v in p.model.state_dict().items() if not k.startswith('lda_')}
        rng=rng_state()
        if reference is None:
            reference=dict(model=current,rng=rng,records=copy.deepcopy(p.s['training_records']),
                history=copy.deepcopy(p.s['history'].raw),readout=copy.deepcopy(readout),stats=copy.deepcopy(p.s['stat_branches']))
        else:
            if any(not torch.equal(v,current[k]) for k,v in reference['model'].items()):raise ArithmeticError('shared backbone/A/B/aux differs')
            if reference['records']!=p.s['training_records']:raise ArithmeticError('training records differ')
            if any(not torch.equal(v,p.s['history'].raw[k]) for k,v in reference['history'].items()):raise ArithmeticError('Patch history differs')
            if (not torch.equal(rng['torch'],reference['rng']['torch']) or any(not torch.equal(a,b) for a,b in zip(rng['cuda'],reference['rng']['cuda']))
                or rng['python']!=reference['rng']['python'] or not np.array_equal(rng['numpy'][1],reference['rng']['numpy'][1])):raise ArithmeticError('RNG differs')
            mode=modes[0]
            if readout['branch_readouts'][mode]!=reference['readout']['branch_readouts'][mode]:raise ArithmeticError('branch prediction differs')
            if any(not torch.equal(v,p.s['stat_branches'][mode][k]) for k,v in reference['stats'][mode].items()):raise ArithmeticError('branch historical statistics differ')
        results.append(dict(modes=list(modes),two_task_restore=True,shared_training_equal=True))
        del p,current;gc.collect();torch.cuda.empty_cache()
    # Exercise the evaluator-only diagnostic branches on the same synthetic
    # final anchor; these summaries never enter fit or selected acceptance.
    os.environ['ARM']='rrs_shared';p=build(clock,synthetic=([raw,raw],manifest,loader))
    p.cfg=copy.deepcopy(p.cfg);p.cfg.update(clients=2,epochs=1,rounds=1,batch=2,tasks=2,diagnostic_tasks=[2])
    p.out=root/('src/runs/check_diagnostic_'+str(f.get('entry_revision',0)));p.out.mkdir(exist_ok=False)
    while p.s['phase']!='complete':p.step()
    p.restore();p.s['task']=1;diag=p.evaluate(False);p.s['task']=2
    if set(diag['conditional_readouts'])!={'dense_current_scatter','local_current_coordinates','diagonal_metric','isotropic_metric','no_bias'}:raise ValueError('missing conditional readout')
    if set(diag['evaluator_only_transfer'])!={'identity','full','mean_only'}:raise ValueError('missing isolated old-test audit')
    if any(not torch.equal(v,p.model.state_dict()[k].cpu()) for k,v in reference['model'].items()):raise ArithmeticError('diagnostic altered training trajectory')
    del p;gc.collect();torch.cuda.empty_cache()
    os.environ['ARM']='frozen';p=build(clock,synthetic=([raw,raw],manifest,loader))
    p.cfg=copy.deepcopy(p.cfg);p.cfg.update(clients=2,batch=2,tasks=2,primary_mode='dense',branch_modes=['dense','compressed'],diagnostic_tasks=[])
    p.s['frozen_stats']={m:None for m in ('dense','compressed')};p.s['frozen_communication']={m:{} for m in ('dense','compressed')}
    p.out=root/('src/runs/check_frozen_'+str(f.get('entry_revision',0)));p.out.mkdir(exist_ok=False)
    for t in range(2):step_frozen(p,t)
    if any(torch.count_nonzero(b) for b in p.model.b_parameters()) or p.s['training_records']:raise ValueError('frozen trained parameters')
    write(root,root/'src/reports/preflight.json',dict(status='passed',real_dataset_images=0,
        shared_vs_three_standalone=results,backbone_A_B_aux_Patch_history_RNG_identical=True,
        independent_branch_statistics_and_predictions_equal=True,identity_short_response=response,
        frozen_two_task_restore=True,evaluator_only_diagnostics_exercised=True,
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()))


if __name__=='__main__':main()
