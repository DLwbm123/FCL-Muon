import torch
from torch import nn
from torch.nn import functional as F


class QVAdapter(nn.Module):
    def __init__(self, linear, rank):
        super().__init__()
        self.base = linear
        d = linear.in_features
        self.register_buffer('A', torch.zeros(rank,d))
        self.Bq = nn.Parameter(torch.zeros(d,rank))
        self.Bv = nn.Parameter(torch.zeros(d,rank))
        self.merged = True
        self.collect = False
        self.stat = None
        self.tokens = 0
    def forward(self, x):
        if self.collect:
            patches = x.detach()[:,1:,:].reshape(-1,x.shape[-1]).double()
            self.stat.add_(patches.T@patches)
            self.tokens += patches.shape[0]
        y = self.base(x)
        low = F.linear(x,self.A)
        dq,dv = F.linear(low,self.Bq),F.linear(low,self.Bv)
        q,k,v = y.chunk(3,dim=-1)
        return torch.cat((q+dq,k,v+dv),dim=-1)
    @torch.no_grad()
    def start(self,a):
        if not self.merged:
            raise ValueError('Unmerged previous task')
        self.A.copy_(a)
        self.Bq.zero_(); self.Bv.zero_()
        self.merged=False
    @torch.no_grad()
    def merge(self):
        if self.merged:
            raise ValueError('Duplicate merge')
        d=self.Bq.shape[0]
        self.base.weight[:d].add_(self.Bq@self.A)
        self.base.weight[2*d:].add_(self.Bv@self.A)
        self.Bq.zero_();self.Bv.zero_()
        self.merged=True


class ContinualViT(nn.Module):
    def __init__(self, backbone, classes, rank=32, scale=16):
        super().__init__()
        self.backbone=backbone
        self.backbone.requires_grad_(False)
        self.adapters={}
        for i,b in enumerate(backbone.blocks):
            a=QVAdapter(b.attn.qkv,rank)
            b.attn.qkv=a
            # Non-owning lookup avoids duplicate state_dict aliases.
            self.adapters[str(i)]=a
        self.rows=nn.ParameterList()
        self.classes=classes
        self.scale=scale
    def add_head(self,cpt):
        for row in self.rows:
            row.requires_grad_(False)
        row=nn.Parameter(torch.empty(cpt,self.backbone.num_features,device=next(self.parameters()).device))
        nn.init.normal_(row,std=0.02)
        self.rows.append(row)
    def b_parameters(self):
        return [p for a in self.adapters.values() for p in (a.Bq,a.Bv)]
    def forward(self,x):
        features=self.backbone.forward_features(x)[:,0]
        weights=torch.cat(list(self.rows))
        return self.scale*F.linear(F.normalize(features,dim=-1),F.normalize(weights,dim=-1))


class MuLoRAFL(ContinualViT):
    method='mulora_fl'

class MuLoRAFLFedMuon(ContinualViT):
    method='mulora_fl_fedmuon'

MODELS={c.method:c for c in (MuLoRAFL,MuLoRAFLFedMuon)}
