"""Paired aggregation study. Both arms inherit the same production learner."""
import json
import numpy as np
import torch
from torch.nn import functional as F
from fcl.checkpoint import atomic_save, rng_state
from .pilot import Pilot, observe
from research.basic_repair.fcl_basic_repair_tools import center_task_logits, summarize


class ConfirmationPilot(Pilot):
    def local_gram(self,k):
        # The pure FedAvg arm has no response-matching statistic or solve.
        # RegMean's observation restores RNG, so this does not unpair training.
        if self.cfg['aggregation']=='fedavg':return None
        return super().local_gram(k)

    def fuse(self):
        s=self.s
        if s['task']==0 and s['round']==0 and self.cfg.get('retain_first_round_endpoints',True):
            atomic_save({'heads':s['heads'],'B':s['agg_B'],
                         'A':[a.A.cpu() for a in self.model.adapters.values()],
                         'sizes':s['sizes'],'rng':rng_state()},self.out/'first_round_endpoints.pt')
        if self.cfg['aggregation']=='regmean':return super().fuse()
        raw=torch.zeros_like(s['heads'][0])
        for w,n in zip(s['heads'],s['sizes']):raw.add_(w,alpha=n/sum(s['sizes']))
        assert torch.isfinite(raw).all() and all(torch.isfinite(b).all() for b in s['agg_B'])
        with torch.no_grad():
            for b,v in zip(self.model.b_parameters(),s['agg_B']):b.copy_(v)
            self.model.rows[-1].copy_(raw)
        atomic_save({'task':s['task'],'round':s['round'],'B':s['agg_B'],'head':raw,
                     'raw_FedAvg_head':raw,'aggregation':'fedavg'},self.out/f"round_t{s['task']}_r{s['round']+1}.pt")
        self.event('aggregate',aggregation='fedavg')
        s.update(round=s['round']+1,heads=[],grams=[],sizes=[],agg_B=[],next_client=0,phase='round_start')

    def evaluate(self):
        t=self.s['task'];c=self.cfg['cpt'];n=(t+1)*c;rows=[];ys=[]
        with observe(self.model):
            for i in range(t+1):
                for x,y in self.loader(self.raw[1],self.ids('test',i),self.cfg['batch']):
                    self.budget();rows.append(self.model(self.test_tf(x.to(self.device))).cpu().numpy());ys.append(y.numpy())
        scores=np.concatenate(rows).astype(np.float64);y=np.concatenate(ys)
        raw=summarize(scores,y,c);centered=summarize(center_task_logits(scores,c),y,c)
        cm=np.bincount(y*n+scores.argmax(1),minlength=n*n).reshape(n,n)
        row=[v['all_seen_accuracy'] for v in raw['per_task']];matrix=self.s['matrix']+[row]
        ids=[i for k in range(t+1) for i in self.ids('test',k)]
        np.savez(self.out/f'task{t+1}_private_logits.npz',logits=scores,labels=y,indices=np.array(ids))
        return {'task':t,'all_seen':raw['all_seen_accuracy'],'per_task':row,
                'restricted':[v['restricted_accuracy'] for v in raw['per_task']],
                'old_accuracy':raw['old_all_seen_accuracy'],'new_accuracy':raw['current_all_seen_accuracy'],
                'forgetting':sum(max(matrix[j][i] for j in range(i,t+1))-row[i] for i in range(t))/t if t else None,
                'old_to_new_error':raw['old_to_current'],'new_to_old_error':raw['current_to_old'],
                'confusion':cm.tolist(),'centered':centered}
