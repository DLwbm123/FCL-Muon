"""Unregistered Muon model-update mode-gate integration; no real-data run."""
import time
import torch
from research.head_regmean.pilot import Pilot
from research.continual_muon.server_mode_gate_draft import dispersion_from_moments,gated_parameter


class ModeGatePilot(Pilot):
    """Collect update second moments at actual local endpoints before restore."""
    def __init__(self,*args,mode_variant='full',**kwargs):
        if mode_variant not in ('full','no_dispersion','history_neutral'):
            raise ValueError('unknown mode gate variant')
        self.mode_variant=mode_variant
        super().__init__(*args,**kwargs)

    def _check_variant(self):
        if self.s.get('mode_variant')!=self.mode_variant:raise ValueError('checkpoint mode variant mismatch')

    def train_client(self):
        self._check_variant()
        if self.s['task']!=2 or self.s['phase']!='clients':raise ValueError('unexpected local lifecycle')
        started=time.time();self._mode_origin=[x.detach().cpu().clone() for x in self.model.b_parameters()]
        self._mode_origin_seconds=time.time()-started
        try:
            return super().train_client()
        finally:
            del self._mode_origin;del self._mode_origin_seconds

    def local_gram(self,k):
        result=super().local_gram(k)
        if not hasattr(self,'_mode_origin') or k!=self.s['next_client']:
            raise ValueError('mode capture not attached to actual local endpoint')
        start=time.time();t=self.s['task'];r=self.s['round']
        groups=self.manifest['partitions']['train'][t]
        if not groups[k] or any(not g for g in groups):raise ValueError('empty client needs explicit protocol')
        w=len(groups[k])/sum(map(len,groups));bs=self.model.b_parameters()
        state=self.s.setdefault('mode_moments',{'round':r,'client_count':0,'weight':0.,
            'sums':[torch.zeros_like(b,device='cpu',dtype=torch.float64) for b in bs],
            'seconds':[torch.zeros(b.shape[1],b.shape[1],device='cpu',dtype=torch.float64) for b in bs],
            'seconds_compute_time':0.,'origin_capture_seconds':0.})
        if state['round']!=r or state['client_count']!=k or len(state['sums'])!=len(bs):
            raise ValueError('mode accumulation order or checkpoint mismatch')
        with torch.no_grad():
            for acc,second,b,origin in zip(state['sums'],state['seconds'],bs,self._mode_origin):
                d=b.detach().cpu().double()-origin.double()
                acc.add_(d,alpha=w);second.add_(d.T@d,alpha=w)
        state['weight']+=w;state['client_count']+=1;state['seconds_compute_time']+=time.time()-start
        state['origin_capture_seconds']+=self._mode_origin_seconds
        return result

    def fuse(self):
        self._check_variant()
        if self.s['task']!=2 or self.s['phase']!='local_done':raise ValueError('unexpected fuse lifecycle')
        r=self.s['round'];state=self.s.get('mode_moments')
        if (not state or state['round']!=r or state['client_count']!=self.cfg['clients'] or
                abs(state['weight']-1)>1e-12):raise ValueError('incomplete update moments')
        origin=[x.detach().cpu().clone() for x in self.model.b_parameters()]
        if len(origin)!=2*len(self.model.adapters) or len(state['sums'])!=len(origin):
            raise ValueError('adapter count mismatch')
        super().fuse()
        base=[x.detach().cpu().clone() for x in self.model.b_parameters()]
        start=time.time();deployed=[];records=[]
        for j,(o,b,mean,second) in enumerate(zip(origin,base,state['sums'],state['seconds'])):
            self.budget();name=list(self.model.adapters)[j//2]
            a=self.model.adapters[name].A.detach().cpu().double()
            history=a@self.s['history'].raw[name].double()@a.T
            current=a@self.s['pending'][name].double()@a.T
            actual=(b.double()-o.double());err=float(torch.linalg.vector_norm(actual-mean))
            scale=max(float(torch.linalg.vector_norm(mean)),1.)
            if err>1e-6*scale:raise ValueError('mean actual update inconsistent with received moments')
            dispersion=dispersion_from_moments(mean,second)
            gate_dispersion=torch.zeros_like(dispersion) if self.mode_variant=='no_dispersion' else dispersion
            gate_history=torch.eye(history.shape[0],dtype=history.dtype) if self.mode_variant=='history_neutral' else None
            value,info=gated_parameter(o,b,history,current,gate_dispersion,gate_history=gate_history)
            info.update(aggregate_difference_l2=err,dispersion_trace=float(torch.trace(dispersion)))
            if not info['accepted'] and not torch.equal(value,b):raise ValueError('fallback differs from FedAvg')
            deployed.append(value);records.append(info)
        with torch.no_grad():
            for p,value in zip(self.model.b_parameters(),deployed):p.copy_(value)
        self.s.setdefault('mode_gate_records',[]).append({'round':r,'mode_variant':self.mode_variant,'layers':records,
            'local_second_moment_seconds':state['seconds_compute_time'],
            'origin_capture_seconds':state['origin_capture_seconds'],'server_gate_seconds':time.time()-start,
            'theoretical_client_second_moment_bytes':self.cfg['clients']*sum(v.numel()*v.element_size() for v in state['seconds']),
            'network_measured_bytes':None})
        del self.s['mode_moments']
