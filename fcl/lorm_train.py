"""Pinned official LoRM methods, manifest-driven three-task comparison harness."""
import argparse,copy,json,os,random,signal,time
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from lightning_fabric import Fabric
from .lorm_upstream import load_lorm
from .data import load_raw,loader,transforms,Indexed
from .checkpoint import atomic_save,rng_state,restore_rng
from .train import digest

STOP=False

def stop(*_):
    global STOP
    STOP=True


def cpu_tree(value):
    if torch.is_tensor(value):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    return value


FIELDS=['cur_A','cur_B','old_delta','run_weights_gram','run_gram','optimization_dict','head',
        'cur_task','cur_offset','cpt','cur_round','cur_train_matrix','is_server']
COUNTERS=['cur_task','cur_offset','cpt','cur_round','cur_train_matrix']


def pack_server(server):
    return {'network':cpu_tree(server.network.state_dict()),
            'attributes':{name:cpu_tree(getattr(server,name)) for name in FIELDS if hasattr(server,name)}}


def restore_models(server,clients,snapshot,counters):
    server.network.load_state_dict(snapshot['network'])
    for name,value in snapshot['attributes'].items():setattr(server,name,value)
    for client,counter in zip(clients,counters):
        client.network.load_state_dict(snapshot['network'])
        for name,value in counter.items():setattr(client,name,value)
        # Prior client state is overwritten by server broadcast at next round;
        # end_task collection needs the same historical temporal mean as the server.
        client.old_delta=server.old_delta


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default=os.environ.get('RUN_CONFIG'));p.add_argument('--resume',default=os.environ.get('RESUME'))
    args=p.parse_args();cfg=json.loads(Path(args.config).read_text())
    root=Path(os.environ['PROJECT_ROOT']);out=root/cfg['output'];out.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    torch.set_num_threads(4);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed'])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    # Original main.py default; does not alter the MuLoRA/FedMuon runs.
    torch.set_float32_matmul_precision('medium')
    raw,cpt,classes=load_raw(cfg['dataset'],root/'data')
    mf=root/cfg['manifest'];m=json.loads(mf.read_text())
    assert m['identity']=={k:cfg[k] for k in ['dataset','seed','alpha','clients']}
    wm=json.loads((root/'weights.json').read_text());assert digest(wm['path'])==wm['sha256']
    LoRM,ViT=load_lorm()
    fabric=Fabric(accelerator='cuda',devices=1,precision=cfg['precision']);fabric.launch()
    network=ViT(pretrained=False,num_classes=classes)
    sd={k:v for k,v in load_file(wm['path']).items() if not k.startswith('head.')}
    missing,extra=network.model.load_state_dict(sd,strict=False)
    assert set(missing)=={'head.weight','head.bias'} and not extra
    def build():
        model=LoRM(fabric,copy.deepcopy(network),'cuda:0',**cfg['official'])
        model.to('cpu');return model
    server=build();clients=[build() for _ in range(cfg['clients'])]
    train_tf,test_tf=transforms();train_tf.cuda();test_tf.cuda()
    for client in clients:client.augment=train_tf;client.test_transform=test_tf
    matrix=[];stage_micro=[];task=0;round_idx=0;initialized=False
    clock=out/'clock.json'
    if clock.exists():original_start=json.loads(clock.read_text())['started']
    else:
        previous=out/'events.jsonl'
        first=json.loads(previous.read_text().splitlines()[0])['time'] if previous.exists() and previous.stat().st_size else time.time()
        original_start=float(os.environ.get('RUN_STARTED',first))
        clock.write_text(json.dumps({'started':original_start}))
    log=(out/'events.jsonl').open('a',buffering=1)
    def event(kind,**kw):
        record={'kind':kind,'time':time.time(),**kw};log.write(json.dumps(record)+'\n');print(kind,json.dumps(kw),flush=True)
    if args.resume:
        s=torch.load(args.resume,map_location='cpu',weights_only=False)
        assert s['config']==cfg and s['manifest_hash']==digest(mf) and s['weight_hash']==wm['sha256']
        restore_models(server,clients,s['server'],s['client_counters'])
        task=s['task'];round_idx=s['round'];initialized=s['initialized'];matrix=s['matrix'];stage_micro=s['stage_micro'];original_start=s['original_start']
        if s['scaler'] is not None:fabric._precision.scaler.load_state_dict(s['scaler'])
        restore_rng(s['rng'])
    event('start',config=cfg,manifest_hash=digest(mf),weight_hash=wm['sha256'],adapted_matrices=len(server.lora_keys),
          matrix_shapes={k:[list(server.cur_B[k].shape),list(server.cur_A[k].shape)] for k in server.lora_keys},
          head='upstream linear weight+bias; current-only CE; RegMean head weight / weighted bias',
          upstream_commit='6d57b5d361f743f3ff0cd1371409bd7a1158de64')
    def save():
        scaler=getattr(fabric._precision,'scaler',None)
        atomic_save({'config':cfg,'server':pack_server(server),'client_counters':[{n:getattr(c,n) for n in COUNTERS} for c in clients],
                     'task':task,'round':round_idx,'initialized':initialized,'matrix':matrix,'stage_micro':stage_micro,
                     'original_start':original_start,'rng':rng_state(),'manifest_hash':digest(mf),'weight_hash':wm['sha256'],
                     'scaler':scaler.state_dict() if scaler else None},out/'latest.pt')
    def get_loader(split,t,k=None,shuffle=False):
        ids=m['partitions'][split][t][k] if k is not None else [i for group in m['partitions'][split][t] for i in group]
        cap=cfg.get('train_batches' if split=='train' else 'test_batches')
        if cap:ids=ids[:cap*cfg['batch']]
        dl=loader(raw[0 if split=='train' else 1],ids,cfg['batch'],shuffle)
        dl.dataset.data=np.array(ids)  # upstream get_client_info only needs sample count
        return fabric.setup_dataloaders(dl)
    def evaluate(t):
        server.to(server.device);server.eval();row=[];counts=[];corrects=[]
        with torch.no_grad():
            for i in range(t+1):
                correct=total=0
                for x,y in get_loader('test',i):
                    pred=server(test_tf(x))[:,:cpt*(t+1)].argmax(-1)
                    correct+=int((pred==y).sum());total+=len(y)
                row.append(correct/total);counts.append(total);corrects.append(correct)
        return row,counts,corrects
    for t in range(task,cfg['tasks']):
        task=t
        if not initialized:
            server.begin_task(cpt)
            for client in clients:client.begin_task(cpt)
            round_idx=0;initialized=True
        for rd in range(round_idx,cfg['rounds']):
            if STOP or time.time()-original_start>=85000:save();event('budget_or_signal_stop');return
            began=time.time();torch.cuda.reset_peak_memory_stats()
            server.to(server.device);server.begin_round_server();server_info=server.get_server_info();info=[]
            for k,client in enumerate(clients):
                dl=get_loader('train',t,k,True);client.to(client.device)
                client.begin_round_client(dl,server_info)
                frozen=client.cur_A if client.cur_train_matrix=='B' else client.cur_B
                frozen_before={key:v.detach().clone() for key,v in frozen.items()}
                steps=visited=0;loss_sum=0.;ct=time.time()
                for ep in range(cfg['epochs']):
                    for x,y in dl:
                        loss=client.observe(x,y)
                        if not np.isfinite(loss):raise RuntimeError('Nonfinite LoRM loss')
                        steps+=1;visited+=len(y);loss_sum+=loss*len(y)
                    client.end_epoch()
                assert all(torch.equal(frozen[key],value) for key,value in frozen_before.items()),'Frozen LoRM factor changed'
                del frozen_before
                client.end_round_client(dl);client.to('cpu');info.append(client.get_client_info(dl))
                event('client',task=t,round=rd,client=k,N=len(dl.dataset),visited=visited,steps=steps,loss=loss_sum/visited,
                      train_matrix=client.cur_train_matrix,lr=cfg['official']['lr'],seconds=time.time()-ct,
                      gram_collection_passes=cfg['official']['regmean_rounds'])
            server.end_round_server(info);del info,server_info
            round_idx=rd+1
            event('round',task=t,round=rd,seconds=time.time()-began,peak_gpu_bytes=torch.cuda.max_memory_allocated(),checkpoint_bytes=(out/'latest.pt').stat().st_size if (out/'latest.pt').exists() else None)
            row,counts,corrects=evaluate(t)
            event('round_evaluation',task=t,round=rd,task_accuracies=row,all_seen_micro=sum(corrects)/sum(counts))
            save()
        server_info=server.get_server_info();info=[]
        for k,client in enumerate(clients):
            dl=get_loader('train',t,k,True);client.to(client.device)
            info.append(client.end_task_client(dl,server_info));client.to('cpu')
        server.end_task_server(info);del info,server_info
        row,counts,corrects=evaluate(t);matrix.append(row);stage_micro.append(sum(corrects)/sum(counts))
        metrics={'matrix':matrix,'stage_micro':stage_micro,'final_micro':stage_micro[-1],
                 'final_task_macro':sum(row)/len(row),'mean_stage_micro':sum(stage_micro)/len(stage_micro),
                 'average_forgetting':sum(max(matrix[j][i] for j in range(i,t+1))-row[i] for i in range(t))/t if t else 0.,
                 'old_micro':sum(corrects[:-1])/sum(counts[:-1]) if t else None,'new_accuracy':row[-1]}
        (out/'metrics.json').write_text(json.dumps(metrics,indent=2));event('evaluation',task=t,**metrics)
        task=t+1;round_idx=0;initialized=False;save()
    (out/'COMPLETE.json').write_text(json.dumps({'config':cfg,'end':time.time(),'elapsed':time.time()-original_start}))
    event('complete',elapsed=time.time()-original_start)

if __name__=='__main__':main()
