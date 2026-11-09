"""Real backbone, synthetic images only: formal round and differentiable response."""
import copy
import json
import time
import torch
from fcl.checkpoint import rng_state,restore_rng
from research.route_compare_20261102.response import federated_response
from .runtime import setup,build,validate
from .storage import write


def main():
    setup();root,f=validate();job=json.loads((root/'src/reports/job.json').read_text())
    g=torch.Generator().manual_seed(426793)
    raw=[(torch.randn(3,224,224,generator=g),torch.tensor(i%10)) for i in range(20)]
    manifest={'labels':{s:[int(y) for _,y in raw] for s in ('train','test')},
              'partitions':{s:[[list(range(10)),list(range(10,20))]] for s in ('train','test')}}
    def loader(raw,ids,batch,shuffle=False):
        ids=[ids[i] for i in (torch.randperm(len(ids)).tolist() if shuffle else range(len(ids)))]
        return [tuple(torch.stack(v) for v in zip(*(raw[j] for j in ids[i:i+batch]))) for i in range(0,len(ids),batch)]
    p=build({k:job[k] for k in ('started','deadline')},synthetic=([raw,raw],manifest,loader))
    p.cfg=copy.deepcopy(p.cfg);p.cfg.update(clients=2,epochs=1,rounds=1,batch=4)
    p.out=root/'src/runs/synthetic';p.out.mkdir(exist_ok=False)
    p.initialize_task();p.save()
    bases={k:a.A.detach().clone().requires_grad_(True) for k,a in p.model.adapters.items()}
    x=torch.randn(4,3,224,224,generator=g).cuda();y=torch.tensor([0,1,2,3],device='cuda')
    support=[[[(x,y)]*4 for _ in range(10)] for _ in range(2)];gram=[[(x,y)] for _ in range(10)]
    snapshot={k:v.detach().cpu().clone() for k,v in p.model.state_dict().items()};rng=rng_state()
    args=(p.model,bases,support,gram,gram,[32]*10,[2]*10,'adamw',.0002,5,5,0,10)
    with torch.autograd.graph.save_on_cpu(pin_memory=False):
        tracked=federated_response(*args,use_checkpoint=True,track_response=True)
        gradients=torch.autograd.grad(tracked['loss'],list(bases.values()))
    if any(not torch.isfinite(g).all() for g in gradients):raise ArithmeticError('nonfinite response derivative')
    restore_rng(rng);direct=federated_response(*args,use_checkpoint=False,track_response=False)
    error=max(float((a.detach()-b.detach()).abs().max()) for a,b in zip(tracked['B'],direct['B']))
    if error>2e-6 or abs(float(tracked['loss'].detach()-direct['loss']))>2e-6:raise ArithmeticError('functional endpoint mismatch')
    if any(not torch.equal(v,p.model.state_dict()[k].cpu()) for k,v in snapshot.items()):raise ValueError('source mutated')
    del tracked,direct,gradients,snapshot
    restore_rng(rng)
    while p.s['phase']!='evaluated':p.step()
    before=copy.deepcopy(p.s['stages'][-1]);state={k:v.detach().clone() for k,v in p.model.state_dict().items()}
    p.restore()
    if before!=p.evaluate(save_artifacts=False):raise ValueError('ViT readout restore mismatch')
    if any(not torch.equal(v,p.model.state_dict()[k]) for k,v in state.items()):raise ValueError('ViT state restore mismatch')
    write(root,root/'src/reports/preflight.json',{'status':'passed','real_dataset_images':0,
        'functional_clients':10,'short_updates':80,'head_fusions':2,'B_track_direct_max_abs':error,
        'formal_clients':2,'formal_synthetic_epoch_count':1,'both_readouts_and_full_restore':True,
        'no_proximal_module':True,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
        'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved()})


if __name__=='__main__':main()
