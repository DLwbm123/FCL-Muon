"""Read-only original checkpoint diagnosis. No optimizer, head fitting or new head.

All crossed readouts use existing ORIGINAL client/server head tensors; they are
offline controls to account for loss, never deployed or saved as new models.
"""
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F


def measures(scores, labels):
    scores=scores.double(); labels=labels.long()
    pred=scores.argmax(1)
    confusion=torch.bincount(labels*10+pred,minlength=100).reshape(10,10)
    support=confusion.sum(1)
    c6=labels==6
    return {'n':len(labels),'accuracy':float((pred==labels).double().mean()),
            'ce':float(F.cross_entropy(scores,labels)),
            'per_class_accuracy':[float(confusion[i,i]/support[i]) if support[i] else None for i in range(10)],
            'confusion':confusion.tolist(),'class6_count':int(c6.sum()),
            'class6_to7':int(confusion[6,7]),
            'class6_margin6minus7_mean':float((scores[c6,6]-scores[c6,7]).mean()) if c6.any() else None}


def main():
    import timm
    from torchvision.datasets import CIFAR100
    from fcl.model import MuLoRAFL
    from fcl.data import loader,transforms
    root=Path(os.environ['EXPERIMENT_ROOT'])
    work=Path(os.environ['PROJECT_ROOT'])
    inputs=root/'basic_repair_20261025/runs/t1_original'
    out=work/'results';out.mkdir(exist_ok=False)
    files=sorted(list(inputs.glob('boundary_t0_r*.pt'))+list(inputs.glob('client_t0_*.pt')))
    assert len(files)==56
    signatures={p.name:(p.stat().st_size,p.stat().st_mtime_ns) for p in files}
    torch.set_num_threads(4);torch.manual_seed(0)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    manifest=json.loads((root/'manifests/seq-cifar100_n10_a0.05_s0.json').read_text())
    train=CIFAR100(root/'data',train=True,download=False)
    test=CIFAR100(root/'data',train=False,download=False)
    groups=manifest['partitions']['train'][0]
    train_ids=[i for ids in groups for i in ids]
    test_ids=[i for ids in manifest['partitions']['test'][0] for i in ids]
    assert len(train_ids)==len(set(train_ids))==5000 and len(test_ids)==1000
    offsets=np.cumsum([0]+list(map(len,groups)))
    _,tf=transforms();tf=tf.cuda()
    model=MuLoRAFL(timm.create_model('vit_base_patch16_224.augreg_in21k',pretrained=False,num_classes=0),100,32,16)
    model.add_head(10);model=model.eval().requires_grad_(False).cuda()
    @torch.inference_mode()
    def extract(raw,ids):
        z,y=[],[]
        for x,labels in loader(raw,ids,16):
            features=model.backbone.forward_features(tf(x.cuda()))[:,0]
            z.append(F.normalize(features,dim=1).cpu());y.append(labels)
        return torch.cat(z),torch.cat(y)
    @torch.inference_mode()
    def scores(z,w):
        return (16*F.linear(z.cuda(),F.normalize(w.cuda(),dim=1))).cpu()
    def metric(z,y,w):return measures(scores(z,w),y)
    began=time.time()
    report={'scope':'Original T1 checkpoints only; no parameter updates or replacement heads',
            'data':'All5000 current train and1000 test; deterministic original test transform for EVERY evaluated state',
            'warning':'Client-selected own-data scores are diagnostics, not a deployable global classifier. Crossed endpoints use original heads only.',
            'global':[],'rounds':[]}
    caches=[]
    for rd in range(6):
        s=torch.load(inputs/f'boundary_t0_r{rd}.pt',map_location='cpu',weights_only=False)
        assert s['task']==0 and s['round']==rd and s['initialized'] and s['head_count']==1
        model.load_state_dict(s['model'],strict=True)
        z,y=extract(train,train_ids);zt,yt=extract(test,test_ids);w=s['model']['rows.0'].clone()
        caches.append((z,y,zt,yt,w))
        record={'boundary_round':rd,'train':metric(z,y,w),'test':metric(zt,yt,w)}
        report['global'].append(record)
        # Private diagnostic cache, never used by training or later-task replay.
        torch.save({'z':z,'y':y,'zt':zt,'yt':yt},out/f'global_r{rd}_features.pt')
        (out/'summary.json').write_text(json.dumps(report,indent=2))
        print('global',rd,record['train']['ce'],record['test']['accuracy'],flush=True)
        del s
    for rd in range(5):
        before=caches[rd];after=caches[rd+1]
        # T1 base and A are fixed throughout all communication rounds.
        s=torch.load(inputs/f'boundary_t0_r{rd}.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(s['model'],strict=True);del s
        agg_head=torch.zeros_like(after[4]);agg_b=[torch.zeros_like(b,device='cpu') for b in model.b_parameters()]
        rows=[];p=[len(ids)/5000 for ids in groups]
        contribution=[]
        for k,ids in enumerate(groups):
            local=torch.load(inputs/f'client_t0_r{rd}_k{k}.pt',map_location='cpu',weights_only=False)
            assert local['task']==0 and local['round']==rd and local['client']==k
            with torch.no_grad():
                for b,v in zip(model.b_parameters(),local['B']):b.copy_(v)
                model.rows[0].copy_(local['head'])
            zl,yl=extract(train,ids);zlt,ylt=extract(test,test_ids)
            lo,hi=offsets[k:k+2]
            za,ya=after[0][lo:hi],after[1][lo:hi]
            assert torch.equal(ya,yl)
            record={'client':k,'N':len(ids),'weight':p[k],
                    'class_counts':torch.bincount(yl,minlength=10).tolist(),
                    'own_train':{
                        'before':metric(before[0][lo:hi],yl,before[4]),
                        'local_encoder_local_head':metric(zl,yl,local['head']),
                        'global_encoder_local_head':metric(za,yl,local['head']),
                        'local_encoder_global_head':metric(zl,yl,after[4]),
                        'after':metric(za,yl,after[4])},
                    'all_test':{
                        'local_encoder_local_head':metric(zlt,ylt,local['head']),
                        'global_encoder_local_head':metric(after[2],after[3],local['head']),
                        'local_encoder_global_head':metric(zlt,ylt,after[4])},
                    'head_row_norms':local['head'].norm(dim=1).tolist()}
            rows.append(record)
            agg_head.add_(local['head'],alpha=p[k])
            for b,v in zip(agg_b,local['B']):b.add_(v,alpha=p[k])
            # Exact additive decomposition of observed global class6-vs7 margin.
            # Shared denominators are the ORIGINAL aggregate head's row norms.
            norms=after[4].double().norm(dim=1)
            term=16*p[k]*(local['head'][6].double()/norms[6]-local['head'][7].double()/norms[7])
            mtrain=after[0][after[1]==6].double()@term
            mtest=after[2][after[3]==6].double()@term
            contribution.append({'client':k,'class6_train_count':record['class_counts'][6],
                                 'mean_train_margin_contribution':float(mtrain.mean()),
                                 'mean_test_margin_contribution':float(mtest.mean())})
            print('client',rd,k,record['own_train']['local_encoder_local_head']['ce'],record['own_train']['after']['ce'],flush=True)
        s=torch.load(inputs/f'boundary_t0_r{rd+1}.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(s['model'],strict=True);del s
        head_error=float((agg_head-after[4]).abs().max())
        b_error=max(float((b-v.cpu()).abs().max()) for b,v in zip(agg_b,model.b_parameters()))
        assert head_error<1e-6 and b_error<1e-6
        pooled={name:{m:sum(p[k]*rows[k]['own_train'][name][m] for k in range(10)) for m in ['accuracy','ce']}
                for name in rows[0]['own_train']}
        assert abs(pooled['after']['ce']-report['global'][rd+1]['train']['ce'])<1e-6
        local_ce=pooled['local_encoder_local_head']['ce']
        global_ce=pooled['after']['ce']
        enc_part=pooled['global_encoder_local_head']['ce']-local_ce
        head_part=global_ce-pooled['global_encoder_local_head']['ce']
        assert abs(enc_part+head_part-(global_ce-local_ce))<1e-10
        actual=sum(c['mean_test_margin_contribution'] for c in contribution)
        reported=report['global'][rd+1]['test']['class6_margin6minus7_mean']
        assert abs(actual-reported)<1e-5
        entry={'round':rd+1,'clients':rows,'sample_weighted_own_train':pooled,
               'loss_decomposition_encoder_then_head':{'local_to_global_total':global_ce-local_ce,'encoder_part':enc_part,'head_part':head_part},
               'loss_decomposition_head_then_encoder':{'head_part':pooled['local_encoder_global_head']['ce']-local_ce,'encoder_part':global_ce-pooled['local_encoder_global_head']['ce']},
               'weighted_local_all_test':{m:sum(p[k]*rows[k]['all_test']['local_encoder_local_head'][m] for k in range(10)) for m in ['accuracy','ce']},
               'head_aggregation_max_abs_error':head_error,'B_aggregation_max_abs_error':b_error,
               'class6_margin_exact_components':contribution}
        report['rounds'].append(entry)
        (out/'summary.json').write_text(json.dumps(report,indent=2))
    assert all((p.stat().st_size,p.stat().st_mtime_ns)==signatures[p.name] for p in files)
    report['checkpoint_signatures_unchanged']=True
    report['elapsed_seconds']=time.time()-began
    report['parameters_updated']=False
    report['new_head_constructed']=False
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    (out/'COMPLETE.json').write_text(json.dumps({'completed':time.time(),'global_states':6,'local_states':50,'elapsed_seconds':report['elapsed_seconds']}))


if __name__=='__main__':main()
