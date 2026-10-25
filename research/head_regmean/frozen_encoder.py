"""Bounded T2 mechanism experiment: freeze the encoder, train only the new head."""
import torch
from torch.nn import functional as F
from .confirmation import ConfirmationPilot
from .head_loss import center_blocks


class FrozenEncoderPilot(ConfirmationPilot):
    expected_first=None
    first_batch_verified=False

    def freeze_encoder(self):
        self.model.backbone.requires_grad_(False)
        for row in self.model.rows[:-1]:row.requires_grad_(False)
        self.model.rows[-1].requires_grad_(True)
        if any(torch.count_nonzero(b) for b in self.model.b_parameters()):
            raise ValueError('Frozen experiment must start from zero unmerged B')

    def charge(self,key,value):
        if key in ('B_download','B_upload','task_end_B_download'):return
        super().charge(key,value)

    def backward_loss(self,x,y):
        if any(v.requires_grad for v in self.model.backbone.parameters()):
            raise ValueError('Encoder is not frozen')
        with torch.no_grad():z=F.normalize(self.model.backbone.forward_features(x)[:,0],dim=-1)
        u=F.normalize(torch.cat(list(self.model.rows)),dim=-1)
        loss=F.cross_entropy(center_blocks(self.model.scale*F.linear(z,u),self.cfg['cpt']),y)
        if not torch.isfinite(loss):raise ArithmeticError('Nonfinite head CE')
        loss.backward()
        assert all(v.grad is None for v in self.model.backbone.parameters())
        assert all(v.grad is None for v in self.model.rows[:-1])
        if self.expected_first is not None:
            e=self.expected_first
            assert torch.equal(x.cpu(),e['x']) and torch.equal(y.cpu(),e['y'])
            assert torch.equal(self.model.rows[-1].grad.cpu(),e['head'])
            self.first_batch_verified=True;self.expected_first=None
            self.event('actual_first_batch_verified',encoder_frozen=True,b_optimizer_steps=0)
        return loss

    def train_client(self):
        s=self.s;k=s['next_client'];t=s['task'];r=s['round']
        global_head=self.model.rows[-1].detach().clone()
        old_heads=[v.detach().clone() for v in self.model.rows[:-1]]
        ids=self.ids('train',t,k);n=len(ids)
        if n:
            dl=self.loader(self.raw[0],ids,self.cfg['batch'],True)
            headopt=torch.optim.AdamW([self.model.rows[-1]],lr=.001,weight_decay=0)
            steps=seen=0;loss_sum=0.
            for _ in range(self.cfg['epochs']):
                for x,y in dl:
                    self.budget();self.check_labels(y);self.model.zero_grad(set_to_none=True)
                    loss=self.backward_loss(self.train_tf(x.to(self.device)),y.to(self.device))
                    headopt.step();steps+=1;seen+=len(y);loss_sum+=float(loss.detach())*len(y)
            assert steps==self.cfg['epochs']*len(dl) and seen==n*self.cfg['epochs']
            gram=self.local_gram(k)
            assert all(torch.equal(a,b) for a,b in zip(old_heads,self.model.rows[:-1]))
            assert all(not torch.count_nonzero(b) for b in self.model.b_parameters())
            # agg_B remains exactly zero; inherited fuse/merge has no encoder update.
            s['heads'].append(self.model.rows[-1].detach().cpu().clone())
            s['grams'].append(gram);s['sizes'].append(n)
            rec={'task':t,'round':r,'client':k,'N':n,'visited':seen,'steps':steps,
                 'head_steps':steps,'B_steps':0,'lr_last':0.,'head_lr':.001,'loss':loss_sum/seen}
            s['training_records'].append(rec);self.event('client',**rec)
            self.charge('current_head_upload',global_head.numel()*4)
        with torch.no_grad():self.model.rows[-1].copy_(global_head)
        s['next_client']+=1
        s['phase']='local_done' if s['next_client']==self.cfg['clients'] else 'clients'
