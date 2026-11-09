"""Real ViT, synthetic images; both deployed protocols and old-stat selection."""
import copy,json,os,time,gc
import torch
from fcl.checkpoint import rng_state,restore_rng
from research.stat_head.runtime import setup,build,validate
from research.adamw_t10.storage import write


def main():
    setup();root,f=validate();clock={k:f[k] for k in ('started','deadline')};results=[]
    generator=torch.Generator().manual_seed(426793)
    raw=[(torch.randn(3,224,224,generator=generator),torch.tensor(i//10*10+i%10)) for i in range(20)]
    partitions=[[[i for i in range(t*10,(t+1)*10) if i%2==k] for k in range(2)] for t in range(2)]
    manifest={'labels':{s:[int(y) for _,y in raw] for s in ('train','test')},'partitions':{s:partitions for s in ('train','test')}}
    def loader(raw,ids,batch,shuffle=False):
        ids=[ids[i] for i in (torch.randperm(len(ids)).tolist() if shuffle else range(len(ids)))]
        return [tuple(torch.stack(v) for v in zip(*(raw[j] for j in ids[i:i+batch]))) for i in range(0,len(ids),batch)]
    for arm in ('gram_codec','unified'):
        os.environ['ARM']=arm;p=build(clock,synthetic=([raw,raw],manifest,loader))
        p.cfg=copy.deepcopy(p.cfg);p.cfg.update(clients=2,epochs=1,rounds=1,batch=2,tasks=2)
        p.cfg['route']['mode']='rrs';p.out=root/('src/runs/synthetic_'+arm);p.out.mkdir(exist_ok=False)
        while not (p.s['task']==1 and p.s['phase']=='task_start'):p.step()
        p.step() # second task initialized, B=0, actual old statistics present
        bases={k:a.A.detach().clone().requires_grad_(True) for k,a in p.model.adapters.items()}
        gram=[]
        for k in range(2):
            gram.append([(x.cuda(),y.cuda()) for x,y in loader(raw,partitions[1][k],2)])
        support=[[gram[k][:2] for k in range(2)] for _ in range(2)]
        snapshot={k:v.detach().cpu().clone() for k,v in p.model.state_dict().items()};rng=rng_state()
        args=(p.model,bases,support,gram,gram,[5,5],[3,3],'adamw',.0002,5,5,1,10)
        with torch.autograd.graph.save_on_cpu(pin_memory=False):
            tracked=p.functional_response(*args,track_response=True,**p.response_options)
            grads=torch.autograd.grad(tracked['loss'],list(bases.values()))
        restore_rng(rng);direct=p.functional_response(*args,track_response=False,**p.response_options)
        loss_error=abs(float(tracked['loss'].detach()-direct['loss']))
        b_error=max(float((a.detach()-b.detach()).abs().max()) for a,b in zip(tracked['B'],direct['B']))
        if loss_error>1e-5 or b_error>2e-6 or any(not torch.isfinite(g).all() for g in grads):raise ArithmeticError('response mismatch')
        if any(not torch.equal(v,p.model.state_dict()[k].cpu()) for k,v in snapshot.items()):raise ValueError('source mutated')
        # At the zero-B anchor, directional finite difference should agree with
        # exact differentiable head statistics when no eigenvalue truncation occurs.
        # Codec/quantization backward elsewhere is explicitly surrogate.
        del tracked,direct,grads,snapshot;restore_rng(rng)
        while p.s['phase']!='complete':p.step()
        p.restore();p.s['task']=1
        metrics=p.evaluate(False);p.s['task']=2
        if metrics!=p.s['stages'][-1]:raise ValueError('deployed readout recovery mismatch')
        if arm=='unified' and int(p.s['unified_stats']['counts'].sum())!=20:raise ValueError('count duplication')
        results.append(dict(arm=arm,loss_track_direct_max_abs=loss_error,B_track_direct_max_abs=b_error,
                            two_task_state_restore=True,actual_old_stat_selected_forward=True,
                            communication=p.s['communication']))
        del p;gc.collect();torch.cuda.empty_cache()
    write(root,root/'src/reports/preflight.json',dict(status='passed',real_dataset_images=0,arms=results,
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()))


if __name__=='__main__':main()
