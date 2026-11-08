"""Unchanged RoutePilot training; add paired fixed readouts and storage guard."""
import os
import torch
from pathlib import Path
from fcl.checkpoint import rng_state
from research.head_regmean.pilot import observe
from research.route_compare_20261102.pilot import RoutePilot
from .storage import allocated, lock, save


def centered_logits(logits, cpt):
    blocks = logits.reshape(len(logits), -1, cpt)
    return (blocks-blocks.mean(dim=-1, keepdim=True)).reshape_as(logits)


def metrics(cm, cpt, previous):
    t = cm.shape[0]//cpt-1
    totals = [int(cm[i*cpt:(i+1)*cpt].sum()) for i in range(t+1)]
    hits = [int(cm.diag()[i*cpt:(i+1)*cpt].sum()) for i in range(t+1)]
    row = [a/b for a,b in zip(hits,totals)]
    matrix = previous[:t]+[row]
    return dict(task=t, all_seen=sum(hits)/sum(totals), per_task=row,
                old_accuracy=sum(hits[:-1])/sum(totals[:-1]) if t else None,
                new_accuracy=row[-1],
                forgetting=sum(max(matrix[j][i] for j in range(i,t+1))-row[i] for i in range(t))/t if t else None,
                old_to_new_error=float(cm[:t*cpt,t*cpt:].sum())/sum(totals[:-1]) if t else None,
                new_to_old_error=float(cm[t*cpt:,:t*cpt].sum())/totals[-1], confusion=cm.tolist())


class TenTaskPilot(RoutePilot):
    def budget(self):
        super().budget()
        # Leave time for an orderly exit before the independent outer timeout.
        if __import__('time').time() >= self.clock['deadline']-120:
            raise TimeoutError('saving/termination reserve reached')

    def save(self):
        root = Path(os.environ['STUDY_ROOT'])
        save(root, self.out/'latest.pt', {
            'schema':'cosine-head-regmean-pilot-1', 'config':self.cfg,
            'identity':self.identity, 'clock':self.clock, 'state':self.s,
            'model':{k:v.detach().cpu() for k,v in self.model.state_dict().items()},
            'head_count':len(self.model.rows),
            'merged_flags':{i:a.merged for i,a in self.model.adapters.items()}, 'rng':rng_state()})

    def evaluate(self, save_artifacts=True):
        t=self.s['task'];c=self.cfg['cpt'];n=(t+1)*c
        raw=torch.zeros(n,n,dtype=torch.int64);center=raw.clone();restricted=[]
        all_logits=[];all_labels=[]
        with observe(self.model):
            for i in range(t+1):
                hit=total=0
                for x,y in self.loader(self.raw[1],self.ids('test',i),self.cfg['batch']):
                    self.budget();logits=self.model(self.test_tf(x.to(self.device))).cpu()
                    if not torch.isfinite(logits).all():raise ArithmeticError('nonfinite test logits')
                    raw+=torch.bincount(y*n+logits.argmax(1),minlength=n*n).reshape(n,n)
                    center+=torch.bincount(y*n+centered_logits(logits,c).argmax(1),minlength=n*n).reshape(n,n)
                    hit+=int((logits[:,i*c:(i+1)*c].argmax(1)+i*c==y).sum());total+=len(y)
                    all_logits.append(logits);all_labels.append(y)
                restricted.append(hit/total)
        result=metrics(raw,c,self.s['matrix']);result['restricted']=restricted
        previous=[s['centered']['per_task'] for s in self.s['stages'][:t]]
        result['centered']=metrics(center,c,previous)
        if save_artifacts:
            save(Path(os.environ['STUDY_ROOT']),self.out/f'logits_T{t+1}.pt',
                 {'logits':torch.cat(all_logits),'labels':torch.cat(all_labels),'task':t})
        return result
