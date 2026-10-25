"""One authorized T1 round; reuse cosine endpoints, train linear once.

No history_spectral import, no changes to the faithful trainer or checkpoints.
"""
import copy
import json
import math
import os
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from fcl.model import MuLoRAFL
from fcl.math import ClientMuon
from fcl.checkpoint import atomic_save, restore_rng, rng_state
from research.head_regmean.solver import aggregate_heads


class HeadVariant(MuLoRAFL):
    def forward(self,x):
        if self.head_type=='cosine':return super().forward(x)
        return F.linear(self.backbone.forward_features(x)[:,0],torch.cat(list(self.rows)))


def measures(logits, y):
    logits=logits.double();pred=logits.argmax(1)
    cm=torch.bincount(y*10+pred,minlength=100).reshape(10,10)
    return {'n':len(y),'accuracy':float((pred==y).double().mean()),
            'ce':float(F.cross_entropy(logits,y)),
            'per_class_accuracy':(cm.diag()/cm.sum(1)).tolist(),'confusion':cm.tolist()}


def main():
    import timm
    from torchvision.datasets import CIFAR100
    from fcl.data import loader,transforms
    root=Path(os.environ['PROJECT_ROOT']);original=Path(os.environ['EXPERIMENT_ROOT'])
    inputs=original/'basic_repair_20261025/runs/t1_original'
    protocol=json.loads((root/'research/head_regmean/PROTOCOL.json').read_text())
    out=root/'runs';out.mkdir(exist_ok=False)
    started=time.time();torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    source_files=[inputs/'boundary_t0_r0.pt',inputs/'boundary_t0_r1.pt']+sorted(inputs.glob('client_t0_r0_k*.pt'))
    signatures={p.name:(p.stat().st_size,p.stat().st_mtime_ns) for p in source_files}
    assert len(source_files)==12
    initial=torch.load(source_files[0],map_location='cpu',weights_only=False)
    assert initial['task']==0 and initial['round']==0 and initial['initialized'] and initial['head_count']==1
    assert not initial['history'].committed and len(initial['pending'])==12
    cfg=initial['config'];assert all(cfg[k]==protocol[k] for k in ['epochs','batch','rank'])
    assert cfg['rounds']==protocol['scheduler_planned_rounds']==5 and cfg['lr']==protocol['B_lr']
    m=json.loads((original/'manifests'/protocol['manifest']).read_text())
    assert m['identity']=={'dataset':'seq-cifar100','seed':0,'alpha':.05,'clients':10}
    groups=m['partitions']['train'][0];sizes=list(map(len,groups));probs=[n/sum(sizes) for n in sizes]
    train_ids=[i for group in groups for i in group];test_ids=[i for group in m['partitions']['test'][0] for i in group]
    assert len(train_ids)==len(set(train_ids))==5000 and len(test_ids)==1000
    train=CIFAR100(original/'data',train=True,download=False);test=CIFAR100(original/'data',train=False,download=False)
    train_tf,test_tf=transforms();train_tf=train_tf.cuda();test_tf=test_tf.cuda()
    model=HeadVariant(timm.create_model('vit_base_patch16_224.augreg_in21k',pretrained=False,num_classes=0),100,32,16).cuda()
    model.add_head(10);model.eval()
    for a in model.adapters.values():a.merged=False
    log=(out/'events.jsonl').open('x',buffering=1)
    def event(kind,**data):
        entry={'kind':kind,'time':time.time(),**data};log.write(json.dumps(entry)+'\n');print(kind,json.dumps(data),flush=True)
    def load_endpoint(endpoint):
        with torch.no_grad():
            for b,v in zip(model.b_parameters(),endpoint['B']):b.copy_(v)
            model.rows[0].copy_(endpoint['head'])
    def load_initial():model.load_state_dict(initial['model'],strict=True)
    @torch.no_grad()
    def extract(raw,ids):
        features=[];labels=[]
        for x,y in loader(raw,ids,16):
            features.append(model.backbone.forward_features(test_tf(x.cuda()))[:,0].cpu());labels.append(y)
        return torch.cat(features),torch.cat(labels)
    def score(z,w,kind):
        if kind=='cosine':return 16*F.linear(F.normalize(z,dim=1),F.normalize(w,dim=1))
        return F.linear(z,w)
    report={'protocol':protocol,'arms':{},'training_run_count':1,'cosine_training_reused':True,
            'warning':'one round only; no full-task or forgetting conclusion; all variants reported, no test-selected head'}
    # Exactly one NEW local-training pass through all clients. No Gram/eval pass
    # interleaved with training; restore original boundary RNG after construction.
    linear_dir=out/'linear';linear_dir.mkdir();load_initial();model.head_type='linear'
    restore_rng(initial['rng']);client_states=copy.deepcopy(initial['client_states'])
    records=[];bs=model.b_parameters()
    for k,ids in enumerate(groups):
        load_initial();dl=loader(train,ids,16,True);steps=0;seen=0;loss_sum=0
        steps_per_round=5*len(dl)
        momentum=[v.cuda() for v in client_states[k]['m']]
        opt=ClientMuon(bs,momentum,[v.cuda() for v in client_states[k]['c']],
                      [v.cuda() for v in initial['server_c']],False)
        headopt=torch.optim.AdamW([model.rows[0]],lr=.001,weight_decay=0)
        client_started=time.time();first_lr=None
        for ep in range(5):
            for x,y in dl:
                lr=.0002*.5*(1+math.cos(math.pi*steps/(5*steps_per_round-1)))
                if first_lr is None:first_lr=lr
                model.zero_grad(set_to_none=True)
                loss=F.cross_entropy(model(train_tf(x.cuda())),y.cuda())
                if not torch.isfinite(loss):raise ArithmeticError('Nonfinite training loss')
                loss.backward();opt.step(lr);headopt.step()
                loss_sum+=float(loss)*len(y);seen+=len(y);steps+=1
        assert steps==steps_per_round and seen==5*len(ids)
        assert all(torch.equal(a.A.cpu(),initial['model'][f'backbone.blocks.{i}.attn.qkv.A']) for i,a in model.adapters.items())
        endpoint={'B':[b.detach().cpu().clone() for b in bs], 'head':model.rows[0].detach().cpu().clone(),
                  'task':0,'round':0,'client':k,'head_type':'linear','bias':False}
        atomic_save(endpoint,linear_dir/f'client_{k}.pt')
        client_states[k]['m']=[v.cpu() for v in momentum];client_states[k]['c']=[v.cpu().clone() for v in momentum]
        rec={'client':k,'N':len(ids),'visited':seen,'steps':steps,'lr_first':first_lr,'lr_last':lr,
             'mean_augmented_training_ce':loss_sum/seen,'seconds':time.time()-client_started}
        records.append(rec);event('linear_client',**rec)
    end_rng=rng_state()
    # B and raw head only; same trained endpoints support both aggregators.
    for kind in ['cosine','linear']:
        model.head_type=kind;load_initial()
        arm_dir=out/kind;arm_dir.mkdir(exist_ok=True)
        heads=[];grams=[];agg_b=[torch.zeros_like(b,device='cpu') for b in bs];local_scores=[]
        for k,ids in enumerate(groups):
            p=inputs/f'client_t0_r0_k{k}.pt' if kind=='cosine' else linear_dir/f'client_{k}.pt'
            endpoint=torch.load(p,map_location='cpu',weights_only=False);load_endpoint(endpoint)
            assert endpoint['task']==0 and endpoint['round']==0 and endpoint['client']==k
            z,y=extract(train,ids);w=endpoint['head'];heads.append(w)
            features=F.normalize(z,dim=1) if kind=='cosine' else z
            g=features.double().T@features.double();grams.append(g)
            for a,b in zip(agg_b,endpoint['B']):a.add_(b,alpha=probs[k])
            local_scores.append({'client':k,'n':len(ids),'local_train_ce':float(F.cross_entropy(score(z,w,kind),y))})
            event('head_gram',head_type=kind,client=k,samples=len(ids),upload_bytes=g.numel()*8)
        reg,avg,certificate=aggregate_heads(heads,grams,sizes,kind=='cosine',alpha=.5)
        atomic_save({'raw_grams':grams,'sizes':sizes,'head_type':kind},arm_dir/'grams_private.pt')
        if kind=='cosine':
            original_after=torch.load(inputs/'boundary_t0_r1.pt',map_location='cpu',weights_only=False)
            load_initial()
            with torch.no_grad():
                for b,v in zip(bs,agg_b):b.copy_(v)
            # Original trainer accumulates FP32 weights, solver comparator used FP64.
            exactavg=torch.zeros_like(heads[0])
            for w,p in zip(heads,probs):exactavg.add_(w,alpha=p)
            assert torch.equal(exactavg,original_after['model']['rows.0'])
            assert all(torch.equal(b.cpu(),original_after['model'][name]) for name,b in model.named_parameters() if name.endswith(('Bq','Bv')))
            avg=exactavg
            next_state=copy.deepcopy(original_after);del original_after
        else:
            next_state=copy.deepcopy(initial);next_state.update(round=1,client_states=client_states,rng=end_rng,scheduler={'round':1})
            next_state['server_c']=[sum(client_states[k]['c'][j]*probs[k] for k in range(10)) for j in range(len(bs))]
        load_initial()
        with torch.no_grad():
            for b,v in zip(bs,agg_b):b.copy_(v)
        z,y=extract(train,train_ids);zt,yt=extract(test,test_ids)
        results={}
        for name,w in [('fedavg',avg),('regmean',reg)]:
            results[name]={'train':measures(score(z,w,kind),y),'test':measures(score(zt,w,kind),yt)}
        with torch.no_grad():model.rows[0].copy_(reg)
        next_state['model']={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        next_state.update(head_type=kind,head_bias=False,head_aggregation='constrained_regmean' if kind=='cosine' else 'regmean',
                          alternate_fedavg_head=avg,regmean_certificate=certificate,protocol=protocol,
                          original_start=float(os.environ['EXPERIMENT_STARTED']),auto_resume_allowed=False)
        assert next_state['round']==1 and next_state['task']==0 and next_state['initialized']
        assert not next_state['history'].committed and len(next_state['pending'])==12
        # Frozen base / A must be unchanged; B must not be prematurely merged.
        for name,value in next_state['model'].items():
            if not name.endswith(('Bq','Bv')) and not name.startswith('rows.'):
                assert torch.equal(value,initial['model'][name]),name
        atomic_save(next_state,arm_dir/'round1.pt')
        # Independent reload of saved head/model with direct full-test agreement.
        loaded=torch.load(arm_dir/'round1.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(loaded['model'],strict=True);zr,yr=extract(test,test_ids)
        reload_result=measures(score(zr,loaded['model']['rows.0'],kind),yr)
        assert reload_result['confusion']==results['regmean']['test']['confusion']
        assert abs(reload_result['ce']-results['regmean']['test']['ce'])<1e-6
        report['arms'][kind]={'results':results,'certificate':certificate,'local_train':local_scores,
                             'training':'reused original first-round endpoints' if kind=='cosine' else records,
                             'checkpoint_reload_verified':True,'BA_merge_count':0,'history_commit_count':0,
                             'extra_head_statistics_upload_bytes':10*768*768*8,
                             'head_download_bytes':10*10*768*4,
                             'extra_statistic_forward_samples':5000}
        (out/'results.json').write_text(json.dumps(report,indent=2));event('arm_complete',head_type=kind,results=results,certificate=certificate)
    assert all((p.stat().st_size,p.stat().st_mtime_ns)==signatures[p.name] for p in source_files)
    report.update(elapsed_seconds=time.time()-started,original_inputs_unchanged=True,history_spectral_resumed=False)
    (out/'results.json').write_text(json.dumps(report,indent=2))
    (out/'COMPLETE.json').write_text(json.dumps({'completed':time.time(),'elapsed_seconds':report['elapsed_seconds'],
                                              'new_training_clients':10,'new_optimizer_epochs_per_client':5,
                                              'cosine_clients_reused':10,'task_complete':False,'round_complete':True},indent=2))


if __name__=='__main__':main()
