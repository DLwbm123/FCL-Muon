"""Research lifecycle hooks for the existing FCIL trainer and data manifests."""
import hashlib,json,time
import torch
import torch.nn.functional as F
from .functional_history import InputHistory,common_core
from .history_optimizer import HistoryOptimizer,update_geometry
from .spectral_qp import SolverConfig,compact_polar
from .checkpoint import rng_state,restore_rng
from .data import loader,current_training_indices


class HistoricalCoordinator:
    def __init__(self,cfg):
        self.cfg=cfg;self.method=cfg['method'];self.provider=cfg.get('direction_provider','mulora')
        if self.provider not in ['mulora','fedmuon']:raise ValueError('Unknown direction provider')
        self.solver=SolverConfig(**cfg.get('solver',{}));self.rho_factor=cfg.get('rho_factor',1.)
        self.lam=cfg.get('lambda_history',0.);self.calibrated=False;self.banks={};self.cores=[];self.caches={};self.task=-1
        self.audit_enabled=cfg.get('paired_audit',True);self.calibration={};self.basis_identity=None;self.audited_tasks=set()
    def attach(self,model,raw,manifest,train_tf,test_tf,event,root):
        self.model=model;self.raw=raw;self.manifest=manifest;self.train_tf=train_tf;self.test_tf=test_tf;self.event=event;self.root=root
        if self.cfg.get('lora_scale',1)!=1:raise ValueError('Only unit LoRA scale is supported')
        if not self.banks:
            self.banks={k:{i:InputHistory(a.A.shape[1],a.Bq.shape[0],energy_floor=self.cfg.get('energy_floor',1e-12)) for i,a in model.adapters.items()} for k in range(self.cfg['clients'])}
    def spectrum(self,x):return update_geometry(x)
    def basis_hash(self):
        value=hashlib.sha256()
        for a in self.model.adapters.values():value.update(a.A.detach().cpu().numpy().tobytes())
        return value.hexdigest()
    def checkpoint_state(self):
        return {'banks':self.banks,'cores':[{k:(v.cpu() if torch.is_tensor(v) else v) for k,v in c.items()} for c in self.cores],
                'caches':{k:[v.cpu() if v is not None else None for v in values] for k,values in self.caches.items()},
                'task':self.task,'lambda':self.lam,'calibrated':self.calibrated,'calibration':self.calibration,
                'schema_version':1,'audited_tasks':self.audited_tasks,'basis_identity':self.basis_identity,'center_ids':{k:{i:b.center for i,b in banks.items()} for k,banks in self.banks.items()},
                'position':'communication_round_boundary; epoch0/minibatch0; default sampler uses checkpointed main RNG'}
    def restore_state(self,state):
        self.audited_tasks=state['audited_tasks'];self.banks=state['banks'];self.cores=[{k:(v.cuda() if torch.is_tensor(v) else v) for k,v in c.items()} for c in state['cores']]
        self.caches={k:[v.cuda() if v is not None else None for v in values] for k,values in state['caches'].items()}
        self.basis_identity=state['basis_identity']
        if self.basis_identity is not None:assert self.basis_identity==self.basis_hash(),'Solver basis identity mismatch'
        self.task=state['task'];self.lam=state['lambda'];self.calibrated=state['calibrated'];self.calibration=state['calibration']
    def begin_task(self,t):
        started=time.perf_counter();self.task=t;self.basis_identity=self.basis_hash()
        cores,metadata=common_core(self.banks,{i:a.A for i,a in self.model.adapters.items()})
        self.cores=[{k:(v.cuda().float() if torch.is_tensor(v) else v) for k,v in c.items()} for c in cores]
        self.caches={k:[None]*len(self.cores) for k in self.banks}
        fingerprint=hashlib.sha256()
        for c in cores:
            for name in ['H','F']:fingerprint.update(c[name].numpy().tobytes())
            fingerprint.update(str(c['q0']).encode())
        self.event('historical_core',task=t,core_hash=fingerprint.hexdigest(),shared_identically_by_all_clients=True,
                   metadata=metadata,seconds=time.perf_counter()-started,
                   client_upload_bytes=sum((a.A.shape[0]**2+a.Bq.numel()+1)*8 for a in self.model.adapters.values())*2*len(self.banks),
                   server_broadcast_bytes=sum((c['H'].numel()+c['F'].numel()+1)*4 for c in cores)*len(self.banks))
        if t>0 and not self.calibrated and self.cfg.get('lambda_mode')=='training_curvature_reference' and self.method!='mulora_fl_exact_polar':self.calibrate(t)
    def calibrate(self,t):
        saved=rng_state();before=time.perf_counter();ratios=[];batches=[]
        cpt=self.manifest['cpt'];allowed=set(current_training_indices(self.manifest,t))
        try:
            # Fixed current train identities, deterministic transform; no update to B,
            # head, momentum, controls, history, cache or training sampler.
            for k in range(min(self.cfg['clients'],self.cfg.get('calibration_clients',2))):
                ids=self.manifest['partitions']['train'][t][k]
                for number,(x,y) in enumerate(loader(self.raw[0],ids,self.cfg['batch'],False)):
                    if number>=self.cfg.get('calibration_batches',2):break
                    used=ids[number*self.cfg['batch']:(number+1)*self.cfg['batch']];assert set(used)<=allowed
                    self.model.zero_grad(set_to_none=True)
                    loss=F.cross_entropy(self.model(self.test_tf(x.cuda()))[:,t*cpt:(t+1)*cpt],y.cuda()-t*cpt);loss.backward()
                    gain=curve=0.
                    for p,core in zip(self.model.b_parameters(),self.cores):
                        m=.1*p.grad.float();eta=self.cfg['lr']*(max(p.shape)**.5 if self.provider=='fedmuon' else 1.)
                        d=-eta*compact_polar(m)[0];gain-=float((m*d).sum());curve+=.5*float(((d@core['H'])*d).sum())
                    if curve>0 and gain>0:ratios.append(gain/curve)
                    batches.append({'client':k,'indices':used,'gain':gain,'curvature_cost':curve,'valid_denominator':curve>0})
            if not ratios:raise RuntimeError('Training-only lambda calibration has no nondegenerate batch')
            reference=float(torch.tensor(ratios,dtype=torch.float64).median())
            self.lam=reference*self.cfg.get('lambda_multiplier',1.);self.calibrated=True
            self.calibration={'task':t,'reference':reference,'multiplier':self.cfg.get('lambda_multiplier',1.),'lambda':self.lam,
                              'selection':'predeclared training-scale diagnostic, not validation-selected or optimal lambda','batches':batches}
            self.event('lambda_calibration',seconds=time.perf_counter()-before,**self.calibration)
        finally:self.model.zero_grad(set_to_none=True);restore_rng(saved)
    def optimizer(self,bs,momentum,old_c,server_c,client):return HistoryOptimizer(bs,momentum,old_c,server_c,self,client)
    def before_merge(self,t):
        ids=self.manifest['partitions']['train'][t][0][:self.cfg['batch']]
        assert set(ids)<=set(current_training_indices(self.manifest,t))
        self.probe_input=self.test_tf(next(iter(loader(self.raw[0],ids,self.cfg['batch'])))[0].cuda())
        with torch.no_grad():self.probe_output=self.model(self.probe_input).clone()
        center=f"{self.cfg['output']}:task{t}:merged"
        for k,banks in self.banks.items():
            for i,a in self.model.adapters.items():banks[i].recenter(t,{'q':a.Bq@a.A,'v':a.Bv@a.A},center)
    @torch.no_grad()
    def after_merge(self,t):
        started=time.perf_counter()
        torch.testing.assert_close(self.model(self.probe_input),self.probe_output,atol=5e-5,rtol=5e-5)
        del self.probe_input,self.probe_output
        for k,banks in self.banks.items():
            ids=self.manifest['partitions']['train'][t][k]
            cap=self.cfg.get('history_stats_batches')
            if cap:ids=ids[:cap*self.cfg['batch']]
            if not set(ids)<=set(current_training_indices(self.manifest,t)):raise ValueError('Non-current training data in history registration')
            for a in self.model.adapters.values():
                assert a.merged and not torch.count_nonzero(a.Bq) and not torch.count_nonzero(a.Bv)
                a.collect=True;a.tokens=0;a.stat=torch.zeros(a.A.shape[1],a.A.shape[1],device='cuda',dtype=torch.float64)
            for x,y in loader(self.raw[0],ids,self.cfg['batch']):self.model.backbone.forward_features(self.test_tf(x.cuda()))
            for i,a in self.model.adapters.items():
                a.collect=False;c=a.stat;d=a.Bq.shape[0]
                energies={p:float(((a.base.weight[s:s+d].double()@c)*a.base.weight[s:s+d].double()).sum()) for p,s in [('q',0),('v',2*d)]}
                banks[i].register(t,c,energies,a.tokens,f"{self.cfg['output']}:task{t}:merged")
                a.stat=None
            self.event('history_registration',task=t,client=k,samples=len(ids),tokens_per_block={i:b.n for i,b in banks.items()},
                       new_anchor_residual_contribution=0.,old_history_recentered_before_new_C=True)
        self.caches={k:[None]*len(self.cores) for k in self.banks}
        state_bytes=sum(b.C.numel()*8+sum(r['R'].numel()*8+16 for r in b.risks.values()) for bs in self.banks.values() for b in bs.values())
        self.event('history_boundary',task=t,seconds=time.perf_counter()-started,bank_bytes=state_bytes,adapter_merge_forward_check='passed')
