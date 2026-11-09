import torch
from torch.nn import functional as F
from fcl.model import MuLoRAFL


class StatisticsViT(MuLoRAFL):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);d=self.backbone.num_features
        self.register_buffer('lda_weight',torch.zeros(self.classes,d))
        self.register_buffer('lda_bias',torch.zeros(self.classes))
        self.register_buffer('lda_classes',torch.tensor(0,dtype=torch.int64))

    def forward(self,x):
        n=int(self.lda_classes)
        if not n:return super().forward(x)
        z=F.normalize(self.backbone.forward_features(x)[:,0],dim=-1)
        return F.linear(z,self.lda_weight[:n],self.lda_bias[:n])

    def deploy(self,W,beta):
        with torch.no_grad():
            n=len(W);self.lda_weight[:n].copy_(W.float());self.lda_bias[:n].copy_(beta.float());self.lda_classes.fill_(n)
