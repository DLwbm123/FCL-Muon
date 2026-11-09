import copy
import torch
from torch.nn import functional as F
from research.rrs_compression.pilot import CompressedPilot
from research.rrs_compression.codec import encode,decode
from research.head_regmean.pilot import observe
from .statistics import projection,local_packet,merge
from .response import federated_response


class HeadPilot(CompressedPilot):
    functional_response=staticmethod(federated_response)

    @property
    def response_options(self):
        return dict(protocol=self.cfg['head_protocol'],old=self.s.get('stat_anchor'),
                    projection=self.R.to(self.device),head_rank=self.cfg['head_rank'])

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.R=projection(self.model.backbone.num_features,self.cfg['transport_rank'])

    def initialize_task(self):
        self.s['stat_anchor']=copy.deepcopy(self.s.get('unified_stats'))
        self.s['anchor_features']={}
        super().initialize_task()

    def backward_loss(self,x,y):
        if self.cfg['head_protocol']=='gram_codec':return super().backward_loss(x,y)
        z=F.normalize(self.model.backbone.forward_features(x)[:,0],dim=1)
        logits=self.cfg['cosine_scale']*F.linear(z,F.normalize(self.model.rows[-1],dim=1))
        loss=F.cross_entropy(logits,y-self.s['task']*self.cfg['cpt'])
        if not torch.isfinite(loss):raise ArithmeticError('nonfinite CE')
        loss.backward();return loss

    def local_gram(self,k):
        if self.cfg['head_protocol']=='unified':return torch.empty(0)
        dense=super().local_gram(k)
        packet=encode(dense,rank=self.cfg['head_rank']);reconstructed,record=decode(packet)
        self.charge('head_Gram_upload',-dense.numel()*8)
        self.charge('head_Gram_compressed_upload',len(packet))
        record.update(task=self.s['task'],round=self.s['round'],client=k)
        self.s.setdefault('head_codec_records',[]).append(record)
        return reconstructed

    def _features(self,k):
        zs=[];ys=[]
        for x,y in self.loader(self.raw[0],self.ids('train',self.s['task'],k),self.cfg['batch']):
            self.budget();self.check_labels(y)
            zs.append(F.normalize(self.model.backbone.forward_features(self.test_tf(x.to(self.device)))[:,0],dim=1).cpu())
            ys.append(y-self.s['task']*self.cfg['cpt'])
        return torch.cat(zs),torch.cat(ys)

    def fuse(self):
        if self.cfg['head_protocol']=='gram_codec':return super().fuse()
        s=self.s;bs=self.model.b_parameters()
        with observe(self.model):
            if not s['anchor_features']:
                snapshot=[b.clone() for b in bs]
                try:
                    for b in bs:b.zero_()
                    for k in range(self.cfg['clients']):s['anchor_features'][k]=self._features(k)
                finally:
                    for b,v in zip(bs,snapshot):b.copy_(v)
            for b,v in zip(bs,s['agg_B']):b.copy_(v)
            total=sum(s['sizes'])
            self.model.rows[-1].copy_(sum(h*(n/total) for h,n in zip(s['heads'],s['sizes'])))
            packets=[]
            for k in range(self.cfg['clients']):
                z,y=self._features(k);z0,y0=s['anchor_features'][k]
                if not torch.equal(y,y0):raise ValueError('anchor order changed')
                packet,wire,record=local_packet(z0,z,y,self.cfg['cpt'],self.R,self.cfg['head_rank'])
                packets.append(packet);self.charge('unified_statistics_upload',len(wire))
                record.update(task=s['task'],round=s['round'],client=k)
                s.setdefault('head_codec_records',[]).append(record)
            stats,W,beta,cert=merge(packets,s['stat_anchor'],self.R)
            if int(stats['counts'].sum())!=(s['task']+1)*len(self.ids('train',s['task'])):
                raise AssertionError('unique class count multiplied across rounds')
            s['unified_stats']={k:v.detach().cpu() for k,v in stats.items()}
            self.model.deploy(W.to(self.device),beta.to(self.device))
            self.charge('all_seen_head_download',self.cfg['clients']*(W.numel()+beta.numel())*4)
        s['certificates'].append(dict(task=s['task'],round=s['round'],**cert))
        self.event('aggregate',certificate=cert)
        s.update(round=s['round']+1,heads=[],grams=[],sizes=[],agg_B=[],next_client=0,phase='round_start')
