"""Prespecified D1/D2 on immutable task boundaries; no training or fitting."""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import timm
from torch.nn import functional as F
from torchvision.datasets import CIFAR100
from fcl.data import loader, transforms
from fcl.model import MuLoRAFL
from research.basic_repair.fcl_basic_repair_tools import center_task_logits, summarize


def distribution(a):
    a=np.asarray(a)
    return {'mean':float(a.mean()),'quantiles':np.quantile(a,[0,.1,.25,.5,.75,.9,1]).tolist()}


def statistics(scores,y):
    result=summarize(scores,y)
    n,c=scores.shape;t=c//10;pred=scores.argmax(1)
    cm=np.bincount(y*c+pred,minlength=c*c).reshape(c,c)
    result['confusion']=cm.tolist()
    result['per_class_accuracy']=(cm.diagonal()/cm.sum(1)).tolist()
    blocks=scores.reshape(n,t,10)
    own=np.arange(c)[None,:]//10==y[:,None]//10
    true=scores[np.arange(n),y]
    not_true=np.arange(c)[None,:]!=y[:,None]
    within=true-np.where(own & not_true,scores,-np.inf).max(1)
    cross=true-np.where(~own,scores,-np.inf).max(1) if t>1 else None
    result['score_distributions_by_true_task']=[]
    for k in range(t):
        select=y//10==k
        result['score_distributions_by_true_task'].append({
            'task':k,'within_margin':distribution(within[select]),
            'cross_margin':distribution(cross[select]) if t>1 else None,
            'block_mean':[distribution(blocks[select,j].mean(1)) for j in range(t)],
            'block_max':[distribution(blocks[select,j].max(1)) for j in range(t)]})
    return result


def paired(before,after,y):
    a=before.argmax(1);b=after.argmax(1)
    return [{'task':k,'corrected':int(((a!=y)&(b==y)&(y//10==k)).sum()),
             'broken':int(((a==y)&(b!=y)&(y//10==k)).sum()),
             'prediction_changed':int(((a!=b)&(y//10==k)).sum())} for k in range(before.shape[1]//10)]


def main():
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    root=Path(os.environ['PROJECT_ROOT']);base=Path(os.environ['EXPERIMENT_ROOT'])
    source=base/'head_regmean_pilot_20261025/runs/pilot';out=root/'runs/diagnostic'
    out.mkdir(parents=True,exist_ok=False)
    clock=json.loads((root/'reports/job.json').read_text())
    expected=json.loads((source/'metrics.json').read_text())
    mf=json.loads((base/'manifests/seq-cifar100_n10_a0.05_s0.json').read_text())
    data=CIFAR100(base/'data',train=False,download=False)
    assert list(data.targets)==mf['labels']['test']
    _,tf=transforms();tf=tf.cuda()
    results=[];identities=[];raw_matrix=[];center_matrix=[]
    for tasks in [1,2,3]:
        path=source/f'task{tasks}.pt';before=(path.stat().st_size,path.stat().st_mtime_ns)
        state=torch.load(path,map_location='cpu',weights_only=False);cfg=state['config']
        assert state['head_count']==tasks and state['state']['merges']==list(range(tasks))
        assert all(state['merged_flags'].values())
        model=MuLoRAFL(timm.create_model(cfg['backbone'],pretrained=False,num_classes=0),100,cfg['rank'],cfg['cosine_scale']).cuda()
        for _ in range(tasks):model.add_head(10)
        model.load_state_dict(state['model'],strict=True);model.eval()
        assert all(not torch.count_nonzero(b) for b in model.b_parameters())
        weights=torch.cat(list(model.rows));alternate=weights.detach().clone()
        if tasks>1:
            rd=torch.load(source/f'round_t{tasks-1}_r5.pt',map_location='cpu',weights_only=False)
            assert rd['task']==tasks-1 and rd['round']==4
            assert torch.equal(rd['head'],model.rows[-1].detach().cpu())
            alternate[-10:]=rd['raw_FedAvg_head'].cuda()
        ids=[i for task in mf['partitions']['test'][:tasks] for client in task for i in client]
        assert len(set(ids))==len(ids)==tasks*1000
        rows=[];alternates=[];labels=[]
        with torch.no_grad():
            for x,y in loader(data,ids,cfg['batch']):
                if time.time()>=clock['deadline']:raise TimeoutError('Original diagnostic deadline')
                z=F.normalize(model.backbone.forward_features(tf(x.cuda()))[:,0],dim=1)
                a=model.scale*F.linear(z,F.normalize(weights,dim=1))
                b=model.scale*F.linear(z,F.normalize(alternate,dim=1))
                assert torch.equal(a[:,:-10],b[:,:-10])
                rows.append(a.cpu().numpy());alternates.append(b.cpu().numpy());labels.append(y.numpy())
        scores=np.concatenate(rows).astype(np.float64);backup=np.concatenate(alternates).astype(np.float64);y=np.concatenate(labels)
        raw=statistics(scores,y);assert raw['confusion']==expected['stages'][tasks-1]['confusion']
        centered=center_task_logits(scores);cs=statistics(centered,y)
        result={'tasks':tasks,'D1_raw':raw,'D1_centered':cs,'D1_paired':paired(scores,centered,y),
                'qualification':{'raw_confusion_reproduced':True,'all_block_rankings_preserved':True,'old_logits_unchanged_D2':True}}
        if tasks>1:
            bc=center_task_logits(backup)
            result.update(D2_fedavg=statistics(backup,y),D2_fedavg_centered=statistics(bc,y),
                          D2_fedavg_to_regmean_paired=paired(backup,scores,y),
                          D2_centered_fedavg_to_regmean_paired=paired(bc,centered,y))
        np.savez(out/f'task{tasks}_private_logits.npz',logits=scores,fedavg_logits=backup,labels=y,indices=np.array(ids))
        assert (path.stat().st_size,path.stat().st_mtime_ns)==before
        identities.append({'task':tasks,'checkpoint':path.name,'identity':state['identity'],'size':before[0],'mtime_ns':before[1]})
        results.append(result)
        raw_matrix.append([v['all_seen_accuracy'] for v in raw['per_task']])
        center_matrix.append([v['all_seen_accuracy'] for v in cs['per_task']])
        print(json.dumps({'tasks':tasks,'raw':raw['all_seen_accuracy'],'centered':cs['all_seen_accuracy'],
                          'fedavg':result.get('D2_fedavg',{}).get('all_seen_accuracy')}),flush=True)
        del model,state
    forgetting=lambda m:sum(max(m[j][i] for j in range(i,3))-m[-1][i] for i in range(2))/2
    report={'protocol':'D1 fixed task-block logit centering; D2 same endpoint current-head replacement',
            'training_steps':0,'fitted_parameters':0,'test_forward_images':6000,'source_identities':identities,
            'stages':results,'D1_raw_matrix':raw_matrix,'D1_centered_matrix':center_matrix,
            'D1_raw_forgetting':forgetting(raw_matrix),'D1_centered_forgetting':forgetting(center_matrix),
            'limitations':['seed0 development evidence; no causal claim about head-only origin',
                            'D2 FedAvg on RegMean historical endpoints, not an independent training baseline',
                            'no task identity used for inference; labels only used for metrics']}
    (out/'RESULTS.json').write_text(json.dumps(report,indent=2))
    (out/'COMPLETE.json').write_text(json.dumps({'completed':time.time(),'qualified':True,'training_steps':0}))


if __name__=='__main__':main()
