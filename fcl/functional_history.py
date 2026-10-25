"""Exact centered sufficient statistics; Q/V share only their identical input C."""
import torch
from .spectral_qp import checked_psd


class InputHistory:
    schema_version=1
    def __init__(self,din,dout,center='initial',energy_floor=1e-12):
        self.C=torch.zeros(din,din,dtype=torch.float64)
        self.risks={p:{'R':torch.zeros(dout,din,dtype=torch.float64),'e':0.,'z':0.} for p in ['q','v']}
        self.n=0;self.center=center;self.energy_floor=energy_floor;self.registered=set();self.moved=set()
        self.metadata={'schema_version':1,'token_policy':'patch_only_no_CLS','transform':'upstream deterministic test-form on current train',
                       'dtype':'torch.float64','normalization':'max(anchor_energy, energy_floor*n*dout)'}
    def recenter(self,task,deltas,new_center):
        if task in self.moved:raise ValueError('Repeated historical recenter')
        for p,d in deltas.items():
            d=d.detach().cpu().double();risk=self.risks[p]
            value=risk['e']+float((risk['R']*d).sum())+.5*float(((d@self.C)*d).sum())
            tolerance=1e-10*max(risk['z'],abs(risk['e']),1e-30)
            if value < -tolerance:raise ValueError('Negative raw historical residual energy')
            risk['e']=value  # Preserve tiny rounding negatives, do not hide them.
            risk['R']=risk['R']+d@self.C
        self.center=new_center;self.moved.add(task)
    def register(self,task,c,energies,n,center):
        if task not in self.moved or center!=self.center:raise ValueError('Must recenter old history before registration')
        if task in self.registered:raise ValueError('Repeated historical registration')
        if n<0:raise ValueError('Negative token count')
        self.C+=c.detach().cpu().double();self.n+=n
        for p in self.risks:self.risks[p]['z']+=float(energies[p])
        self.registered.add(task)
    def project(self,a,p):
        risk=self.risks[p];a=a.detach().cpu().double()
        if self.n==0:return {'H':a@self.C@a.T,'F':risk['R']@a.T,'q0':0.,'Z':0.,'floor_active':False,'n':0}
        floor=self.energy_floor*self.n*risk['R'].shape[0];z=max(risk['z'],floor)
        h,_,correction=checked_psd(a@self.C@a.T/z)
        return {'H':h,'F':risk['R']@a.T/z,'q0':risk['e']/z,'Z':z,'floor_active':risk['z']<floor,'n':self.n,'psd_correction':correction}


def risk_value(core,b):
    return core['q0']+(core['F']*b).sum()+.5*((b@core['H'])*b).sum()


def common_core(banks,bases,client_weights=None):
    """Uniform nonempty historical clients; a_l=1/(2*number of blocks)."""
    output=[];metadata=[];layer_weight=1/(2*len(bases))
    for name,a in bases.items():
        active=[k for k in banks if banks[k][name].n>0]
        weights={k:1/len(active) for k in active} if client_weights is None else client_weights
        for p in ['q','v']:
            projected=[(k,banks[k][name].project(a,p)) for k in active]
            zeroH=torch.zeros(a.shape[0],a.shape[0],dtype=torch.float64)
            dout=banks[next(iter(banks))][name].risks[p]['R'].shape[0]
            zeroF=torch.zeros(dout,a.shape[0],dtype=torch.float64)
            core={'H':zeroH,'F':zeroF,'q0':0.}
            for k,c in projected:
                factor=layer_weight*weights[k];core['H']+=factor*c['H'];core['F']+=factor*c['F'];core['q0']+=factor*c['q0']
                metadata.append({'block':name,'projection':p,'client':k,'Z':c['Z'],'floor_active':c['floor_active'],'n':c['n'],'client_weight':weights[k],'layer_weight':layer_weight})
            output.append(core)
    return output,metadata
