import argparse
import copy
import hashlib
import json
import math
import os
import random
import signal
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import timm
from safetensors.torch import load_file
from .data import load_raw,manifest,loader,transforms,current_training_indices
from .model import MODELS
from .math import rrs,History,ClientMuon,spectrum
from .checkpoint import atomic_save,rng_state,restore_rng

STOP=False

def stop(*_):
    global STOP
    STOP=True


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
    return h.hexdigest()


def main(extension=None):
    p=argparse.ArgumentParser()
    p.add_argument('--config',default=os.environ.get('RUN_CONFIG'))
    p.add_argument('--resume',default=os.environ.get('RESUME'))
    p.add_argument('--fork-task1',default=os.environ.get('FORK_TASK1'))
    args=p.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    assert cfg['method'] in MODELS
    root=Path(os.environ.get('PROJECT_ROOT','.')).resolve()
    out=root/cfg['output'];out.mkdir(parents=True,exist_ok=True)
    if extension and (root/'reports/USER_PAUSE.json').exists():
        raise RuntimeError('Research experiments are paused by the user for Pro review')
    source_digest=hashlib.sha256()
    if extension:
        for path in sorted((Path(__file__).parent).glob('*.py')):
            source_digest.update(path.name.encode());source_digest.update(path.read_bytes())
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    torch.set_num_threads(4)
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed'])
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    raw,cpt,classes=load_raw(cfg['dataset'],root/'data')
    mf=root/'manifests'/f"{cfg['dataset']}_n{cfg['clients']}_a{cfg['alpha']}_s{cfg['seed']}.json"
    m=manifest(raw,cpt,classes,cfg,mf)
    wmeta=json.loads((root/'weights.json').read_text())
    assert digest(wmeta['path'])==wmeta['sha256']
    backbone=timm.create_model('vit_base_patch16_224.augreg_in21k',pretrained=False,num_classes=0)
    weights=load_file(wmeta['path'])
    weights={k:v for k,v in weights.items() if not k.startswith('head.')}
    backbone.load_state_dict(weights,strict=True)
    model=MODELS[cfg['method']](backbone,classes,cfg['rank'],cfg.get('cosine_scale',16)).cuda()
    train_tf,test_tf=transforms();train_tf=train_tf.cuda();test_tf=test_tf.cuda()
    hist=History();pending={};client_states={};server_c=[];matrix=[];stage_micro=[]
    task=0;round_idx=0;initialized=False
    start=time.time();original_start=start
    log=(out/'events.jsonl').open('a',buffering=1)
    def event(kind,**data):
        log.write(json.dumps({'kind':kind,'time':time.time(),**data})+'\n')
        print(kind,json.dumps(data),flush=True)
    if extension:extension.attach(model,raw,m,train_tf,test_tf,event,root)
    if args.resume or args.fork_task1:
        s=torch.load(args.resume or args.fork_task1,map_location='cpu',weights_only=False)
        if args.fork_task1:
            assert extension and s['task']==1 and not s['initialized'] and s['head_count']==1
            for key in ['dataset','seed','alpha','clients','tasks','rounds','epochs','batch','rank','lr','cosine_scale','direction_provider','train_batches','test_batches','stats_batches','history_stats_batches']:
                assert s['config'].get(key)==cfg.get(key),('Fork protocol mismatch',key)
        else:assert s['config']==cfg
        assert s['manifest_hash']==digest(mf) and s['weight_hash']==wmeta['sha256']
        for _ in range(s['head_count']):model.add_head(cpt)
        model.load_state_dict(s['model']);hist=s['history'];pending=s['pending'];client_states=s['client_states'];server_c=s['server_c']
        task=s['task'];round_idx=s['round'];initialized=s['initialized'];matrix=s['matrix'];stage_micro=s['stage_micro'];original_start=s['original_start']
        for a in model.adapters.values():a.merged=not initialized
        if extension:extension.restore_state(s['research'])
        restore_rng(s['rng'])
    measure_spectrum=extension.spectrum if extension else spectrum
    model.eval()
    event('start',config=cfg,weight=wmeta,manifest=str(mf),trainable_B=sum(p.numel() for p in model.b_parameters()),
          matrices={f'blocks.{i}.attn.{q}':list(getattr(a,'B'+q).shape) for i,a in model.adapters.items() for q in ['q','v']},
          A_bytes=sum(a.A.numel()*a.A.element_size() for a in model.adapters.values()),
          operator={'name':'Takezawa scaled NS' if cfg['method']=='mulora_fl_fedmuon' else 'MuLoRA Eq.3 SSB',
                    'dimension_multiplier':cfg['method']=='mulora_fl_fedmuon','beta':.9,'nesterov':False,'NS_steps':5,
                    'coefficients':[3.4445,-4.7750,2.0315],'normalization_epsilon':1e-7},
          cosine_scale=cfg.get('cosine_scale',16),precision='FP32 training, FP64 statistics and RRS')
    def save():
        state={'config':cfg,'model':model.state_dict(),'head_count':len(model.rows),'history':hist,'pending':pending,
               'client_states':client_states,'server_c':server_c,'task':task,'round':round_idx,'initialized':initialized,
               'matrix':matrix,'stage_micro':stage_micro,'rng':rng_state(),'manifest_hash':digest(mf),
               'weight_hash':wmeta['sha256'],'original_start':original_start,'scheduler':{'round':round_idx},
               'research':extension.checkpoint_state() if extension else None,'checkpoint_position':{'epoch':0,'minibatch':0},
               'source_revision':cfg.get('source_revision','baseline-preserved'),
               'source_hash':source_digest.hexdigest() if extension else None,
               'config_hash':hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()}
        atomic_save(state,out/'latest.pt')
        if extension and cfg.get('export_task1') and task==1 and not initialized and not (out/'task1_shared.pt').exists():
            # Hard link retains the exact boundary snapshot; no copy during export.
            os.link(out/'latest.pt',out/'task1_shared.pt')
    def budget():
        if time.time()-original_start>85000:return False
        # New project resources shared with runs; conservative full-root allocated bytes.
        total=sum(p.stat().st_blocks*512 for p in root.rglob('*') if p.is_file() and not p.is_symlink())
        return total<45_000_000_000
    def indices(split,t,k=None):
        clients=m['partitions'][split][t]
        return clients[k] if k is not None else [i for client in clients for i in client]
    for t in range(task,cfg['tasks']):
        task=t
        if not initialized:
            began=time.time();model.add_head(cpt);pending={}
            for a in model.adapters.values():
                a.collect=True;a.tokens=0
            stat_ids=current_training_indices(m,t)
            selected=set(stat_ids[:cfg['stats_batches']*cfg['batch']]) if cfg.get('stats_batches') else None
            total_stats={i:torch.zeros(a.A.shape[1],a.A.shape[1],dtype=torch.float64) for i,a in model.adapters.items()}
            stat_samples=0;stat_tokens={i:0 for i in model.adapters}
            # Explicit client-local raw sums; server never reweights by N.
            for k in range(cfg['clients']):
                client_ids=indices('train',t,k)
                if selected is not None:client_ids=[i for i in client_ids if i in selected]
                for a in model.adapters.values():
                    a.stat=torch.zeros(a.A.shape[1],a.A.shape[1],dtype=torch.float64,device='cuda');a.tokens=0
                with torch.no_grad():
                    for x,y in loader(raw[0],client_ids,cfg['batch']):
                        model.backbone.forward_features(test_tf(x.cuda()))
                for i,a in model.adapters.items():
                    total_stats[i].add_(a.stat.cpu());stat_tokens[i]+=a.tokens
                stat_samples+=len(client_ids)
                event('statistics_communication',task=t,client=k,samples=len(client_ids),
                      raw_moments_upload_bytes=sum(v.numel()*8 for v in total_stats.values()),
                      shared_A_download_bytes=sum(a.A.numel()*4 for a in model.adapters.values()))
            for i,a in model.adapters.items():a.stat=total_stats[i];a.tokens=stat_tokens[i]
            event('statistics',task=t,seconds=time.time()-began,samples=stat_samples,tokens={i:a.tokens for i,a in model.adapters.items()})
            began=time.time()
            for i,a in model.adapters.items():
                a.collect=False;pending[i]=a.stat.cpu();a.stat=None
                basis,details=rrs(pending[i].cuda(),hist.raw.get(i,torch.zeros_like(pending[i])).cuda(),cfg['rank'])
                a.start(basis);event('rrs',task=t,layer=i,**details)
            event('rrs_end',task=t,seconds=time.time()-began)
            bs=model.b_parameters()
            client_states={k:{'m':[torch.zeros_like(b,device='cpu') for b in bs],'c':[torch.zeros_like(b,device='cpu') for b in bs]} for k in range(cfg['clients'])}
            server_c=[torch.zeros_like(b,device='cpu') for b in bs]
            round_idx=0;initialized=True
            if extension:extension.begin_task(t)
            save()
        bs=model.b_parameters()
        for r in range(round_idx,cfg['rounds']):
            if STOP or not budget():save();event('budget_or_signal_stop',task=t,round=r);return
            began=time.time();torch.cuda.reset_peak_memory_stats()
            global_bs=[b.detach().clone() for b in bs];global_head=model.rows[-1].detach().clone()
            sizes=[len(indices('train',t,k)) for k in range(cfg['clients'])]
            probs=[n/sum(sizes) for n in sizes]
            agg=[torch.zeros_like(b) for b in bs];agghead=torch.zeros_like(global_head);newc=[torch.zeros_like(b,device='cpu') for b in bs]
            for k in range(cfg['clients']):
                with torch.no_grad():
                    for b,g in zip(bs,global_bs):b.copy_(g)
                    model.rows[-1].copy_(global_head)
                ids=indices('train',t,k)
                if cfg.get('train_batches'):ids=ids[:cfg['train_batches']*cfg['batch']]
                dl=loader(raw[0],ids,cfg['batch'],True)
                steps_per_round=cfg['epochs']*len(dl);steps=0;loss_sum=0.;seen=0
                state=client_states[k]
                momentum=[v.cuda() for v in state['m']];old_c=[v.cuda() for v in state['c']]
                opt=extension.optimizer(bs,momentum,old_c,[v.cuda() for v in server_c],k) if extension else ClientMuon(bs,momentum,old_c,[v.cuda() for v in server_c],cfg['method']=='mulora_fl_fedmuon')
                headopt=torch.optim.AdamW([model.rows[-1]],lr=.001,weight_decay=0)
                lr_values=[];first_update=None;delta_diag=[];client_started=time.time()
                for ep in range(cfg['epochs']):
                    for x,y in dl:
                        progress=(r*steps_per_round+steps)/max(cfg['rounds']*steps_per_round-1,1)
                        lr=cfg['lr']*.5*(1+math.cos(math.pi*progress));lr_values.append(lr)
                        model.zero_grad(set_to_none=True)
                        logits=model(train_tf(x.cuda()))[:,t*cpt:(t+1)*cpt]
                        loss=F.cross_entropy(logits,y.cuda()-t*cpt)
                        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                        loss.backward();updates=opt.step(lr);headopt.step()
                        if first_update is None:
                            first_update=measure_spectrum(updates[0])
                            delta_diag=[measure_spectrum(u) for u in updates]
                        loss_sum+=loss.item()*len(y);seen+=len(y);steps+=1
                if extension:opt.flush(t,r)
                state['m']=[v.cpu() for v in momentum]
                state['c']=[v.clone() for v in state['m']]
                with torch.no_grad():
                    for j,b in enumerate(bs):
                        if not torch.isfinite(b).all():raise RuntimeError('Nonfinite B')
                        agg[j].add_(b,alpha=probs[k]);newc[j].add_(state['c'][j],alpha=probs[k])
                    agghead.add_(model.rows[-1],alpha=probs[k])
                event('client',task=t,round=r,client=k,N=sizes[k],visited=seen,steps=steps,lr=lr_values,
                      loss=loss_sum/seen,training_seconds=time.time()-client_started,first_delta_B=first_update,
                      first_step_delta_B_all_matrices=delta_diag,local_cumulative_B=measure_spectrum(bs[0]),
                      local_cumulative_B_all_matrices=[measure_spectrum(b) for b in bs],
                      communication_bytes={'B_upload':sum(b.numel()*4 for b in bs),'B_download':sum(b.numel()*4 for b in bs),
                      'control_each_direction':sum(b.numel()*4 for b in bs) if opt.corrected else 0,'head_each_direction':global_head.numel()*4})
            with torch.no_grad():
                for b,a in zip(bs,agg):b.copy_(a)
                model.rows[-1].copy_(agghead)
            aggregate_started=time.time()
            server_c=newc;round_idx=r+1;save()
            event('round',task=t,round=r,seconds=time.time()-began,aggregate_B=measure_spectrum(bs[0]),aggregate_B_all_matrices=[measure_spectrum(b) for b in bs],
                  checkpoint_seconds=time.time()-aggregate_started,persistent_tensor_bytes=sum(p.numel()*p.element_size() for p in model.state_dict().values())+sum(v.numel()*8 for v in hist.raw.values())+sum(v.numel()*8 for v in pending.values())+sum(v.numel()*4 for state in client_states.values() for key in ['m','c'] for v in state[key])+sum(v.numel()*4 for v in server_c),peak_gpu_bytes=torch.cuda.max_memory_allocated(),checkpoint_bytes=(out/'latest.pt').stat().st_size)
        # Evaluate stage before merge, using all seen logits for every test task.
        began=time.time();row=[];counts=[];corrects=[];confusion=[];group_results=[]
        with torch.no_grad():
            for i in range(t+1):
                test_ids=indices('test',i)
                if cfg.get('test_batches'):test_ids=test_ids[:cfg['test_batches']*cfg['batch']]
                correct=total=old_to_new=old_to_old_wrong=0
                client_ids={idx:k for k,ids in enumerate(m['partitions']['test'][i]) for idx in ids}
                group_counts={k:[0,0] for k in range(cfg['clients'])};offset=0
                for x,y in loader(raw[1],test_ids,cfg['batch']):
                    pred=model(test_tf(x.cuda())).argmax(-1).cpu()
                    correct+=int((pred==y).sum());total+=len(y)
                    if extension:
                        if i<t:
                            old_to_new+=int((pred>=t*cpt).sum());old_to_old_wrong+=int(((pred!=y)&(pred<t*cpt)).sum())
                        for pos,idx in enumerate(test_ids[offset:offset+len(y)]):
                            k=client_ids[idx];group_counts[k][0]+=int(pred[pos]==y[pos]);group_counts[k][1]+=1
                        offset+=len(y)
                row.append(correct/total);counts.append(total);corrects.append(correct)
                if extension:
                    confusion.append({'task':i,'old_to_new':old_to_new,'old_to_old_wrong':old_to_old_wrong,'total':total})
                    group_results.append(group_counts)
        matrix.append(row);stage_micro.append(sum(corrects)/sum(counts))
        metrics={'matrix':matrix,'stage_micro':stage_micro,'final_micro':stage_micro[-1],
                 'final_task_macro':sum(row)/len(row),'mean_stage_micro':sum(stage_micro)/len(stage_micro),
                 'average_forgetting':sum(max(matrix[j][i] for j in range(i,t+1))-row[i] for i in range(t))/t if t else 0.,
                 'old_micro':sum(corrects[:-1])/sum(counts[:-1]) if t else None,'new_accuracy':row[-1]}
        if extension:
            metrics['old_to_new_error_rate']=sum(c['old_to_new'] for c in confusion)/sum(counts[:-1]) if t else None
            metrics['old_to_old_error_rate']=sum(c['old_to_old_wrong'] for c in confusion)/sum(counts[:-1]) if t else None
            metrics['client_seen_accuracy']={k:sum(g[k][0] for g in group_results)/max(sum(g[k][1] for g in group_results),1) for k in range(cfg['clients'])}
        (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
        event('evaluation',task=t,seconds=time.time()-began,**metrics)
        if extension:extension.before_merge(t)
        for a in model.adapters.values():a.merge()
        if extension:extension.after_merge(t)
        hist.commit(t,pending);pending={};client_states={};server_c=[]
        task=t+1;round_idx=0;initialized=False;save()
    (out/'COMPLETE.json').write_text(json.dumps({'end':time.time(),'elapsed':time.time()-original_start,'config':cfg}))
    event('complete',elapsed=time.time()-original_start)

if __name__=='__main__':main()
