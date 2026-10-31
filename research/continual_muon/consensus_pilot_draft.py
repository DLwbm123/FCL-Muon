"""Unregistered multiround Muon-consensus integration; no real-data run."""
import time
import torch
from fcl.math import mulora_ssb
from fcl.checkpoint import rng_state
from scripts.real_boundary_worker import same_rng
from research.head_regmean.pilot import Pilot
from research.continual_muon.consensus_aggregation_draft import online_whitener,finish_online


class ConsensusPilot(Pilot):
    def __init__(self,*args,consensus_variant='full',**kwargs):
        if consensus_variant not in ('full','history_neutral','server_polar'):
            raise ValueError('unknown consensus variant')
        self.consensus_variant=consensus_variant
        super().__init__(*args,**kwargs)

    def _check_variant(self):
        if self.s.get('consensus_variant')!=self.consensus_variant:raise ValueError('checkpoint variant mismatch')

    def train_client(self):
        self._check_variant()
        if self.s['task']!=2 or self.s['phase']!='clients':raise ValueError('unexpected local lifecycle')
        started=time.time();self._consensus_origin=[x.detach().cpu().clone() for x in self.model.b_parameters()]
        self._consensus_origin_seconds=time.time()-started
        try:return super().train_client()
        finally:del self._consensus_origin;del self._consensus_origin_seconds

    def local_gram(self,k):
        before_rng=rng_state()
        result=super().local_gram(k)
        if not hasattr(self,'_consensus_origin') or k!=self.s['next_client']:
            raise ValueError('consensus capture not attached to local endpoint')
        started=time.time();t=self.s['task'];r=self.s['round'];groups=self.manifest['partitions']['train'][t]
        if not groups[k] or any(not g for g in groups):raise ValueError('empty client needs explicit protocol')
        weights=len(groups[k])/sum(map(len,groups));bs=self.model.b_parameters()
        state=self.s.setdefault('consensus_moments',{'round':r,'client_count':0,'weight':0.,
            'mean_updates':[torch.zeros_like(b,device='cpu',dtype=torch.float64) for b in bs],
            'consensus_sums':[torch.zeros_like(b,device='cpu',dtype=torch.float64) for b in bs],
            'valid_history':[True for _ in bs],'local_ssb_calls':[0 for _ in bs],
            'consensus_seconds':0.,'origin_capture_seconds':0.})
        if state['round']!=r or state['client_count']!=k or len(state['mean_updates'])!=len(bs):
            raise ValueError('accumulation order or checkpoint mismatch')
        with torch.no_grad():
            for j,(mean,acc,b,origin) in enumerate(zip(state['mean_updates'],state['consensus_sums'],bs,self._consensus_origin)):
                self.budget();d=b.detach().cpu().double()-origin.double();mean.add_(d,alpha=weights)
                if self.consensus_variant=='server_polar':continue
                name=list(self.model.adapters)[j//2];a=self.model.adapters[name].A.detach().cpu().double()
                h=a@self.s['history'].raw[name].double()@a.T
                whiten=online_whitener(h,use_history=(self.consensus_variant=='full'))
                if whiten is None:state['valid_history'][j]=False;continue
                acc.add_(mulora_ssb((d@whiten).float()).double(),alpha=weights);state['local_ssb_calls'][j]+=1
        state['weight']+=weights;state['client_count']+=1;state['consensus_seconds']+=time.time()-started
        state['origin_capture_seconds']+=self._consensus_origin_seconds
        if not same_rng(before_rng,rng_state()):raise ValueError('online consensus changed training RNG')
        return result

    def fuse(self):
        self._check_variant()
        if self.s['task']!=2 or self.s['phase']!='local_done':raise ValueError('unexpected fuse lifecycle')
        r=self.s['round'];state=self.s.get('consensus_moments')
        if (not state or state['round']!=r or state['client_count']!=self.cfg['clients'] or
                abs(state['weight']-1)>1e-12):raise ValueError('incomplete online consensus')
        origin=[x.detach().cpu().clone() for x in self.model.b_parameters()]
        if len(origin)!=2*len(self.model.adapters) or len(state['mean_updates'])!=len(origin):
            raise ValueError('adapter count mismatch')
        super().fuse();base=[x.detach().cpu().clone() for x in self.model.b_parameters()]
        started=time.time();deployed=[];records=[]
        for j,(o,b,mean,q,valid,calls) in enumerate(zip(origin,base,state['mean_updates'],state['consensus_sums'],state['valid_history'],state['local_ssb_calls'])):
            self.budget();name=list(self.model.adapters)[j//2];a=self.model.adapters[name].A.detach().cpu().double()
            h=a@self.s['history'].raw[name].double()@a.T
            if not valid:
                value,info=b.clone(),{'accepted':False,'reason':'empty_history','ssb_calls':calls}
            else:
                if self.consensus_variant=='server_polar':q=b.double()-o.double()
                value,info=finish_online(o,b,mean,q,h,local_ssb_calls=calls)
            if not info['accepted'] and not torch.equal(value,b):raise ValueError('fallback differs from FedAvg')
            deployed.append(value);records.append(info)
        with torch.no_grad():
            for p,value in zip(self.model.b_parameters(),deployed):p.copy_(value)
        self.s.setdefault('consensus_records',[]).append({'round':r,'consensus_variant':self.consensus_variant,'layers':records,
            'origin_capture_seconds':state['origin_capture_seconds'],'local_consensus_seconds':state['consensus_seconds'],
            'server_consensus_seconds':time.time()-started,'additional_update_upload_bytes':0,
            'additional_upload_assumption':'server sees each already-uploaded B and shares common basis',
            'network_measured_bytes':None})
        del self.s['consensus_moments']
