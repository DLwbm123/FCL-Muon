"""Cosine Head-RegMean continuation, isolated from faithful/v1/history trainers."""
import contextlib
import copy
import json
import math
import os
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from fcl.checkpoint import atomic_save, rng_state, restore_rng
from fcl.math import History, ClientMuon, rrs
from .solver import aggregate_heads


@contextlib.contextmanager
def observe(model):
    state=rng_state(); modes={m:m.training for m in model.modules()}
    try:
        model.eval()
        with torch.no_grad():yield
    finally:
        for m,mode in modes.items():m.training=mode
        restore_rng(state)


class Pilot:
    def __init__(self,model,raw,manifest,loader,train_tf,test_tf,config,out,identity,clock):
        self.model=model;self.device=next(model.parameters()).device
        self.raw=raw;self.manifest=manifest;self.loader=loader
        self.train_tf=train_tf;self.test_tf=test_tf;self.cfg=config
        self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True)
        self.identity=identity;self.clock=clock
        self.s={'task':0,'round':0,'phase':'task_start','history':History(),'pending':{},
                'client_states':{},'matrix':[],'stages':[],'merges':[],
                'communication':{},'certificates':[],'training_records':[],
                'stats_records':[],'next_client':0,'heads':[],'grams':[],'sizes':[],'agg_B':[]}
        self.model.eval()

    def event(self,kind,**values):
        with (self.out/'events.jsonl').open('a') as f:
            f.write(json.dumps({'kind':kind,'time':time.time(),'task':self.s['task'],
                'round':self.s['round'],**values})+'\n')

    def ids(self,split,t,k=None):
        groups=self.manifest['partitions'][split][t]
        return groups[k] if k is not None else [i for g in groups for i in g]

    def charge(self,key,value):self.s['communication'][key]=self.s['communication'].get(key,0)+int(value)

    def budget(self):
        if time.time()>=self.clock['deadline']:raise TimeoutError('Original experiment deadline reached')

    def save(self):
        atomic_save({'schema':'cosine-head-regmean-pilot-1','config':self.cfg,
            'identity':self.identity,'clock':self.clock,'state':self.s,
            'model':{k:v.detach().cpu() for k,v in self.model.state_dict().items()},
            'head_count':len(self.model.rows),'merged_flags':{i:a.merged for i,a in self.model.adapters.items()},
            'rng':rng_state()},self.out/'latest.pt')

    def restore(self):
        x=torch.load(self.out/'latest.pt',map_location='cpu',weights_only=False)
        if x['schema']!='cosine-head-regmean-pilot-1' or x['config']!=self.cfg or x['identity']!=self.identity or x['clock']!=self.clock:
            raise ValueError('Recovery identity/config/original clock mismatch')
        for _ in range(x['head_count']-len(self.model.rows)):self.model.add_head(self.cfg['cpt'])
        self.model.load_state_dict(x['model'],strict=True);self.s=x['state']
        for i,a in self.model.adapters.items():a.merged=x['merged_flags'][i]
        merged=self.s['phase'] in ('task_start','merged','committed','complete')
        if any(a.merged!=merged for a in self.model.adapters.values()):raise ValueError('Merge/phase mismatch')
        if self.s['phase']=='complete' and len(self.s['history'].committed)!=self.cfg['tasks']:raise ValueError('Missing history')
        restore_rng(x['rng'])

    def import_round1(self,x):
        expected={'task':0,'round':1,'initialized':True,'head_count':1,
            'head_type':'cosine','head_bias':False,'head_aggregation':'constrained_regmean'}
        if any(x.get(k)!=v for k,v in expected.items()):raise ValueError('Not a cosine RegMean R1 boundary')
        for k in ['dataset','seed','alpha','clients','rounds','epochs','batch','rank','lr','cosine_scale']:
            if x['config'][k]!=self.cfg[k]:raise ValueError('Source protocol mismatch: '+k)
        if x['history'].committed or len(x['pending'])!=len(self.model.adapters) or x['scheduler']['round']!=1:
            raise ValueError('R1 task/history/scheduler mismatch')
        if x['manifest_hash']!=self.identity['manifest'] or x['weight_hash']!=self.identity['weights']:
            raise ValueError('Source data/weight mismatch')
        if not x['regmean_certificate']['deployed_fp32']['passed']:raise ValueError('Unqualified source head')
        self.model.add_head(self.cfg['cpt']);self.model.load_state_dict(x['model'],strict=True)
        for a in self.model.adapters.values():a.merged=False
        self.s.update(round=1,phase='round_start',history=copy.deepcopy(x['history']),
            pending=copy.deepcopy(x['pending']),client_states=copy.deepcopy(x['client_states']))
        self.s['imported_R1']={'certificate':x['regmean_certificate'],'source_original_start':x['original_start'],
            'reused_training_clients':self.cfg['clients'],'head_aggregation':'constrained_regmean'}
        restore_rng(x['rng']);self.save()

    def initialize_task(self):
        t=self.s['task'];self.model.add_head(self.cfg['cpt'])
        totals={i:torch.zeros_like(a.A.T@a.A,device='cpu',dtype=torch.float64) for i,a in self.model.adapters.items()}
        with torch.no_grad():
            try:
                for k in range(self.cfg['clients']):
                    ids=self.ids('train',t,k)
                    for a in self.model.adapters.values():
                        a.collect=True;a.stat=torch.zeros(a.A.shape[1],a.A.shape[1],device=self.device,dtype=torch.float64);a.tokens=0
                    for x,y in self.loader(self.raw[0],ids,self.cfg['batch']):
                        self.budget();self.check_labels(y)
                        self.model.backbone.forward_features(self.test_tf(x.to(self.device)))
                    for i,a in self.model.adapters.items():totals[i]+=a.stat.cpu()
                    self.charge('RRS_raw_upload',sum(g.numel()*8 for g in totals.values()))
            finally:
                for a in self.model.adapters.values():a.collect=False;a.stat=None
        for i,a in self.model.adapters.items():
            basis,info=rrs(totals[i].to(self.device),self.s['history'].raw.get(i,torch.zeros_like(totals[i])).to(self.device),self.cfg['rank'])
            a.start(basis);self.event('RRS',layer=i,**info)
        self.s['pending']=totals
        self.s['client_states']={k:{'m':[torch.zeros_like(b,device='cpu') for b in self.model.b_parameters()]} for k in range(self.cfg['clients'])}
        self.charge('A_broadcast',self.cfg['clients']*sum(a.A.numel()*4 for a in self.model.adapters.values()))
        self.s['stats_records'].append({'task':t,'kind':'RRS','samples':len(self.ids('train',t))})
        self.s.update(round=0,phase='round_start')

    def check_labels(self,y):
        low=self.s['task']*self.cfg['cpt']
        if bool(((y<low)|(y>=low+self.cfg['cpt'])).any()):raise ValueError('Old/future data in current task')

    def local_gram(self,k):
        d=self.model.backbone.num_features;g=torch.zeros(d,d,dtype=torch.float64,device=self.device);n=0
        with observe(self.model):
            for x,y in self.loader(self.raw[0],self.ids('train',self.s['task'],k),self.cfg['batch']):
                self.budget();self.check_labels(y)
                z=F.normalize(self.model.backbone.forward_features(self.test_tf(x.to(self.device)))[:,0],dim=1).double()
                g.add_(z.T@z);n+=len(y)
        if n!=len(self.ids('train',self.s['task'],k)) or not torch.isfinite(g).all():raise ValueError('Invalid Gram pass')
        self.charge('head_Gram_upload',g.numel()*8)
        self.s['stats_records'].append({'task':self.s['task'],'round':self.s['round'],'client':k,'kind':'local_endpoint_Gram','samples':n})
        return g.cpu()

    def train_client(self):
        s=self.s;k=s['next_client'];t=s['task'];r=s['round'];bs=self.model.b_parameters()
        global_b=[b.detach().clone() for b in bs];global_head=self.model.rows[-1].detach().clone()
        old_heads=[v.detach().clone() for v in self.model.rows[:-1]]
        ids=self.ids('train',t,k);n=len(ids)
        if n:
            dl=self.loader(self.raw[0],ids,self.cfg['batch'],True);count=self.cfg['epochs']*len(dl)
            momentum=[v.to(self.device) for v in s['client_states'][k]['m']]
            opt=ClientMuon(bs,momentum,[],[],False)
            headopt=torch.optim.AdamW([self.model.rows[-1]],lr=.001,weight_decay=0)
            steps=seen=0;loss_sum=0.
            for _ in range(self.cfg['epochs']):
                for x,y in dl:
                    self.budget();self.check_labels(y)
                    lr=self.cfg['lr']*.5*(1+math.cos(math.pi*(r*count+steps)/max(self.cfg['rounds']*count-1,1)))
                    self.model.zero_grad(set_to_none=True)
                    logits=self.model(self.train_tf(x.to(self.device)))[:,t*self.cfg['cpt']:(t+1)*self.cfg['cpt']]
                    loss=F.cross_entropy(logits,y.to(self.device)-t*self.cfg['cpt'])
                    if not torch.isfinite(loss):raise ArithmeticError('Nonfinite CE')
                    loss.backward();opt.step(lr);headopt.step();steps+=1;seen+=len(y);loss_sum+=float(loss.detach())*len(y)
            if steps!=count or seen!=n*self.cfg['epochs']:raise AssertionError('Optimizer budget')
            s['client_states'][k]={'m':[v.cpu() for v in momentum]}
            # This is the local endpoint encoder, BEFORE B FedAvg. Observation
            # restores RNG/modes so the extra pass does not change next-client RNG.
            gram=self.local_gram(k)
            if any(not torch.equal(a,b) for a,b in zip(old_heads,self.model.rows[:-1])):raise AssertionError('Old head changed')
            with torch.no_grad():
                for dst,b in zip(s['agg_B'],bs):dst.add_(b.detach().cpu(),alpha=n/len(self.ids('train',t)))
            s['heads'].append(self.model.rows[-1].detach().cpu().clone());s['grams'].append(gram);s['sizes'].append(n)
            rec={'task':t,'round':r,'client':k,'N':n,'visited':seen,'steps':steps,'lr_last':lr,'loss':loss_sum/seen}
            s['training_records'].append(rec);self.event('client',**rec)
            self.charge('B_upload',sum(b.numel()*4 for b in bs));self.charge('current_head_upload',global_head.numel()*4)
        with torch.no_grad():
            for b,v in zip(bs,global_b):b.copy_(v)
            self.model.rows[-1].copy_(global_head)
        s['next_client']+=1
        s['phase']='local_done' if s['next_client']==self.cfg['clients'] else 'clients'

    def fuse(self):
        s=self.s
        head,raw,cert=aggregate_heads(s['heads'],s['grams'],s['sizes'],True,alpha=.5)
        if not torch.isfinite(head).all() or any(not torch.isfinite(b).all() for b in s['agg_B']):raise ArithmeticError('Nonfinite aggregate')
        with torch.no_grad():
            for b,v in zip(self.model.b_parameters(),s['agg_B']):b.copy_(v)
            self.model.rows[-1].copy_(head)
        s['certificates'].append({'task':s['task'],'round':s['round'],**cert})
        atomic_save({'task':s['task'],'round':s['round'],'B':s['agg_B'],'head':head,
            'raw_FedAvg_head':raw,'certificate':cert},self.out/f"round_t{s['task']}_r{s['round']+1}.pt")
        self.event('aggregate',certificate=cert)
        s.update(round=s['round']+1,heads=[],grams=[],sizes=[],agg_B=[],next_client=0,phase='round_start')

    def evaluate(self):
        t=self.s['task'];c=self.cfg['cpt'];n=(t+1)*c;cm=torch.zeros(n,n,dtype=torch.int64);restricted=[]
        with observe(self.model):
            for i in range(t+1):
                hit=total=0
                for x,y in self.loader(self.raw[1],self.ids('test',i),self.cfg['batch']):
                    self.budget();logits=self.model(self.test_tf(x.to(self.device))).cpu();pred=logits.argmax(1)
                    cm+=torch.bincount(y*n+pred,minlength=n*n).reshape(n,n)
                    hit+=int((logits[:,i*c:(i+1)*c].argmax(1)+i*c==y).sum());total+=len(y)
                restricted.append(hit/total)
        totals=[int(cm[i*c:(i+1)*c].sum()) for i in range(t+1)];hits=[int(cm.diag()[i*c:(i+1)*c].sum()) for i in range(t+1)]
        row=[a/b for a,b in zip(hits,totals)];matrix=self.s['matrix']+[row]
        return {'task':t,'all_seen':sum(hits)/sum(totals),'per_task':row,'restricted':restricted,
            'old_accuracy':sum(hits[:-1])/sum(totals[:-1]) if t else None,'new_accuracy':row[-1],
            'forgetting':sum(max(matrix[j][i] for j in range(i,t+1))-row[i] for i in range(t))/t if t else None,
            'old_to_new_error':float(cm[:t*c,t*c:].sum())/sum(totals[:-1]) if t else None,
            'new_to_old_error':float(cm[t*c:,:t*c].sum())/totals[-1],
            'confusion':cm.tolist()}

    def step(self):
        self.budget();s=self.s;p=s['phase'];t=s['task']
        if p=='complete':return False
        if p=='task_start':self.initialize_task()
        elif p=='round_start':
            if s['round']==self.cfg['rounds']:s['phase']='ready_evaluation'
            else:
                s.update(next_client=0,heads=[],grams=[],sizes=[],agg_B=[torch.zeros_like(b,device='cpu') for b in self.model.b_parameters()],phase='clients')
                self.charge('B_download',self.cfg['clients']*sum(b.numel()*4 for b in self.model.b_parameters()))
                self.charge('current_head_download',self.cfg['clients']*self.model.rows[-1].numel()*4)
        elif p=='clients':self.train_client()
        elif p=='local_done':self.fuse()
        elif p=='ready_evaluation':
            metrics=self.evaluate();s['matrix'].append(metrics['per_task']);s['stages'].append(metrics);s['phase']='evaluated'
            (self.out/'metrics.json').write_text(json.dumps({'matrix':s['matrix'],'stages':s['stages']},indent=2));self.event('evaluation',**metrics)
        elif p=='evaluated':
            if t in s['merges']:raise ValueError('Repeated merge')
            self.charge('task_end_B_download',self.cfg['clients']*sum(b.numel()*4 for b in self.model.b_parameters()))
            self.charge('task_end_head_download',self.cfg['clients']*self.model.rows[-1].numel()*4)
            for a in self.model.adapters.values():a.merge()
            s['merges'].append(t);s['phase']='merged'
        elif p=='merged':s['history'].commit(t,s['pending']);s['pending']={};s['phase']='committed'
        elif p=='committed':
            s.update(task=t+1,round=0,client_states={},phase='complete' if t+1==self.cfg['tasks'] else 'task_start')
        else:raise ValueError('Unknown phase')
        self.save()
        if p=='committed':
            dest=self.out/f'task{t+1}.pt'
            if not dest.exists():os.link(self.out/'latest.pt',dest)
        return True

    def run(self):
        while self.step():pass
