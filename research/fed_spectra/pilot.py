"""Shared production lifecycle with AdamW local learning and spectral increments.

Control/MVP keep RRS. Full uses gradient sketches and a persistent input-space
Fisher state. No SSB, boundary repair, FCCA, old images, or test-side selection.
"""
import copy
import math
import time
import torch
from torch.nn import functional as F
from research.head_regmean.confirmation import ConfirmationPilot
from research.aggregation_utility.pilot import passive_probe
from .math import (barycenter, candidate_pool, fisher_coefficients, gaussian_map,
                   orthogonal_pool, rebase_adapter, select_subspace, signed_disagreement,
                   sym, transport_increments, unit_trace, update_history)


class AdamAdapter:
    def __init__(self, parameters, settings):
        self.optimizer=torch.optim.AdamW(parameters,lr=0.,betas=tuple(settings['adam_betas']),
                                        eps=settings['adam_eps'],weight_decay=0.)

    def step(self, lr):
        for group in self.optimizer.param_groups:group['lr']=lr
        self.optimizer.step()


def probe_ids(indices, task, client, count, seed):
    ids=sorted(indices);generator=torch.Generator().manual_seed(seed+10007*task+101*client)
    order=torch.randperm(len(ids),generator=generator).tolist()
    return [ids[i] for i in order[:count]]


class SpectraPilot(ConfirmationPilot):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.cfg['method'] not in ('controlled_adam','fed_spectra_mvp','fed_spectra_full'):
            raise ValueError('Unknown spectral method')
        self.teacher=None
        self.s.update(spectra_history={},spectra_rounds=[],spectra_geometry=[],spectra_probes=[],
                      spectra_commits=[],spectra_targets={},spectra_pools={},spectra_endpoints=[],
                      teacher_state=None,anchor_forward_images=0,anchor_forward_calls=0,
                      local_ce_sum=0.,local_kl_sum=0.)

    @property
    def settings(self):return self.cfg['fed_spectra']

    @property
    def spectral(self):return self.cfg['method']!='controlled_adam'

    def restore(self):
        super().restore();self.teacher=None

    def adapter_optimizer(self,parameters,state):
        # All arms use identical per-client/per-round Adam resets. No diagonal
        # second moment is incorrectly rotated when Full changes coordinates.
        return AdamAdapter(parameters,self.settings)

    def initialize_task(self):
        s=self.s;t=s['task'];self.teacher=None
        if self.cfg['method']!='fed_spectra_full':
            super().initialize_task()
        else:
            self.model.add_head(self.cfg['cpt'])
            for name,adapter in self.model.adapters.items():
                previous=adapter.A.detach().cpu().double().T
                basis=orthogonal_pool([previous],previous.shape[0],self.cfg['rank'],
                                      self.settings['probe_seed']+int(name))
                adapter.start(basis.T.to(adapter.A))
            s['pending']={}
            s['client_states']={k:{'m':[torch.zeros_like(b,device='cpu') for b in self.model.b_parameters()]}
                                for k in range(self.cfg['clients'])}
            self.charge('A_broadcast',self.cfg['clients']*sum(a.A.numel()*4 for a in self.model.adapters.values()))
            s.update(round=0,phase='round_start')
        for name,adapter in self.model.adapters.items():
            d=adapter.A.shape[1]
            s['spectra_history'].setdefault(name,torch.zeros(d,d,dtype=torch.float64))
        s['spectra_probe_ids']=[probe_ids(self.ids('train',t,k),t,k,self.settings['probe_images'],
                                         self.settings['probe_seed']) for k in range(self.cfg['clients'])]
        s['prepared_round']=None;s['history_probe_committed']=None;s['spectra_endpoints']=[]
        s['teacher_state']={k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()} if t>0 else None

    def get_teacher(self):
        if self.teacher is None:
            if self.s['teacher_state'] is None:raise ValueError('Missing task-start anchor')
            self.teacher=copy.deepcopy(self.model)
            self.teacher.load_state_dict(self.s['teacher_state'],strict=True)
            self.teacher.requires_grad_(False);self.teacher.eval()
        return self.teacher

    def backward_loss(self,x,y):
        t=self.s['task'];c=self.cfg['cpt'];z=self.model(x)
        ce=F.cross_entropy(z[:,t*c:(t+1)*c],y-t*c)
        kl=torch.zeros((),device=z.device)
        if t and self.settings['anchor_weight']:
            temperature=self.settings['anchor_temperature']
            with torch.no_grad():target=(self.get_teacher()(x)[:,:t*c]/temperature).softmax(-1)
            kl=F.kl_div((z[:,:t*c]/temperature).log_softmax(-1),target,reduction='batchmean')*temperature**2
            self.s['anchor_forward_images']+=len(y);self.s['anchor_forward_calls']+=1
        loss=ce+self.settings['anchor_weight']*kl
        if not torch.isfinite(loss):raise ArithmeticError('Nonfinite CE/KL')
        loss.backward()
        self.s['local_ce_sum']+=float(ce.detach())*len(y)
        self.s['local_kl_sum']+=float(kl.detach())*len(y)
        return loss

    def train_client(self):
        ce,kl=self.s['local_ce_sum'],self.s['local_kl_sum'];before=len(self.s['training_records'])
        super().train_client()
        if len(self.s['training_records'])>before:
            rec=self.s['training_records'][-1]
            rec.update(current_ce=(self.s['local_ce_sum']-ce)/rec['visited'],
                       old_kl=(self.s['local_kl_sum']-kl)/rec['visited'],adapter_optimizer='AdamW_round_reset')

    def local_gram(self,k):
        if self.spectral:
            self.s['spectra_endpoints'].append({'client':k,'B':[b.detach().cpu().clone() for b in self.model.b_parameters()]})
        return super().local_gram(k)

    def probe_gradients(self, archive=False):
        """Full gradients stay client-private; only sketches/small Grams upload.

        This single-process simulator caches private client gradients in RAM
        while the server broadcasts P. They never enter public artifacts or
        server messages. No extra images are read beyond current train probes.
        """
        s=self.s;t=s['task'];c=self.cfg['cpt'];entries=[]
        started=time.perf_counter();images=back_images=calls=back_calls=0
        weights=[a.base.weight for a in self.model.adapters.values()]
        names=list(self.model.adapters)
        old_count=(t+1)*c if archive else t*c
        with passive_probe(self.model):
            for w in weights:w.requires_grad_(True)
            for k,ids in enumerate(s['spectra_probe_ids']):
                if not ids:continue
                n=len(ids);cur={};old={}
                for name,a in self.model.adapters.items():
                    d=a.A.shape[1]
                    cur[name]=[torch.zeros(d,d) for _ in range(2)]
                    old[name]=[torch.zeros(d,d) for _ in range(2)]
                for batch_index,(x,y) in enumerate(self.loader(self.raw[0],ids,self.cfg['batch'])):
                    self.budget();self.check_labels(y);y=y.to(self.device)
                    logits=self.model(self.test_tf(x.to(self.device)))
                    if not archive:
                        loss=F.cross_entropy(logits[:,t*c:(t+1)*c],y-t*c,reduction='sum')/n
                        gs=torch.autograd.grad(loss,weights,retain_graph=old_count>0)
                        for name,g in zip(names,gs):
                            d=g.shape[1];cur[name][0].add_(g[:d].detach().cpu());cur[name][1].add_(g[2*d:].detach().cpu())
                        back_calls+=1;back_images+=len(y)
                    if old_count:
                        seed=self.settings['probe_seed']+1000003*t+1009*s['round']+31*k+batch_index+(70000019 if archive else 0)
                        coeff=fisher_coefficients(logits[:,:old_count],seed,self.settings['anchor_temperature'])
                        score=(coeff*logits[:,:old_count]/self.settings['anchor_temperature']).sum()/math.sqrt(n)
                        gs=torch.autograd.grad(score,weights)
                        for name,g in zip(names,gs):
                            d=g.shape[1];old[name][0].add_(g[:d].detach().cpu());old[name][1].add_(g[2*d:].detach().cpu())
                        back_calls+=1;back_images+=len(y)
                    images+=len(y);calls+=1
                if any(not torch.isfinite(g).all() for group in (cur,old) for vv in group.values() for g in vv):
                    raise ArithmeticError('Nonfinite private probe gradient')
                entries.append({'client':k,'n':len(self.ids('train',t,k)),'current':cur,'old':old})
        record={'task':t+1,'round':s['round']+1,'kind':'archive_all_seen_fisher' if archive else 'round_current_old',
                'forward_images':images,'backward_images':back_images,'forward_calls':calls,'backward_calls':back_calls,
                'old_classes':old_count,'seconds':time.perf_counter()-started,'current_train_only':True}
        s['spectra_probes'].append(record)
        return entries

    def prepare_round(self):
        s=self.s;cfg=self.settings;t=s['task'];r=s['round'];started=time.perf_counter()
        entries=self.probe_gradients();weights=torch.tensor([v['n'] for v in entries],dtype=torch.float64);weights/=weights.sum()
        geometry={};targets={};pools={}
        for name,a in self.model.adapters.items():
            previous=a.A.detach().cpu().double().T;d=previous.shape[0]
            omega=gaussian_map(d,cfg['sketch_dim'],cfg['probe_seed']+7919*int(name))
            old_history=s['spectra_history'][name]
            if self.cfg['method']=='fed_spectra_full':
                sketches=[]
                for item in entries:
                    gs=[g.double() for g in item['current'][name]]
                    scale=math.sqrt(sum(float(g.square().sum()) for g in gs))
                    sketch=torch.cat([g.T@omega for g in gs],dim=1)/max(scale,1e-24)
                    sketches.append(sketch.float().double())
                self.charge('spectra_candidate_sketch_upload',len(entries)*d*2*cfg['sketch_dim']*4)
                pool=candidate_pool(previous,old_history,sketches,weights,cfg['candidate_dim'],cfg['candidate_current_rank'],
                                    cfg['probe_seed']+997*t+37*r+int(name)).float().double()
            else:
                pool=previous
            pools[name]=pool
            self.charge('spectra_candidate_pool_broadcast',len(entries)*pool.numel()*4)
            currents=[];histories=[];signed=[]
            for item in entries:
                ec=[g.double()@pool for g in item['current'][name]]
                eo=[g.double()@pool for g in item['old'][name]]
                gram=sum(g.T@g for g in ec);old_gram=sum(g.T@g for g in eo)
                scale=math.sqrt(max(float(gram.trace()),1e-24))
                currents.append(unit_trace(gram));histories.append(unit_trace(old_gram))
                signed.append((torch.cat([omega.T@g for g in ec],dim=0)/scale).float().double())
            m=pool.shape[1]
            self.charge('spectra_small_grams_upload',len(entries)*2*m*m*8)
            self.charge('spectra_signed_sketch_upload',len(entries)*2*cfg['sketch_dim']*m*4)
            consensus,bary=barycenter(currents,weights,cfg['barycenter_type'],cfg['barycenter_ridge'],
                                      cfg['barycenter_iterations'],cfg['barycenter_tolerance'])
            disagreement=signed_disagreement(signed,weights)
            old_current=sum(float(p)*h for p,h in zip(weights,histories))
            history=update_history(old_history,pool,old_current,cfg['history_ema'])
            s['spectra_history'][name]=history
            old_exposure=float((previous.T@history@previous).trace())
            if self.cfg['method']=='fed_spectra_full':
                new,selection=select_subspace(pool,consensus,history,disagreement,self.cfg['rank'],cfg['lambda_old'],cfg['lambda_dis'])
                rebase=rebase_adapter(a,new.to(a.A.device))
                self.charge('spectra_shared_A_broadcast',len(entries)*a.A.numel()*4)
                # Nonzero residuals change frozen Q/V weights at the clients too.
                # Send old B and the new projected B: all clients know old/new A.
                self.charge('spectra_residual_factor_broadcast',len(entries)*4*d*self.cfg['rank']*4)
            else:
                selection={'history_exposure':old_exposure};rebase=[]
            u=a.A.detach().cpu().double().T
            coordinates=torch.linalg.solve(pool.T@pool,pool.T@u)
            targets[name]={'current':sym(coordinates.T@consensus@coordinates),
                           'history':sym(u.T@history@u),'disagreement':sym(coordinates.T@disagreement@coordinates)}
            geometry[name]={'barycenter':bary,'selection':selection,'rebase':rebase,
                            'old_history_exposure':old_exposure,'signed_disagreement_trace':float(disagreement.trace()),
                            'projector_distance':float((u@u.T-previous@previous.T).norm()),
                            'pool_previous_residual':float((previous-pool@torch.linalg.solve(pool.T@pool,pool.T@previous)).norm()),
                            'orthogonal_error':float((u.T@u-torch.eye(u.shape[1])).norm()),
                            'old_fisher_nonzero_clients':sum(float(v.trace())>1e-24 for v in histories),
                            'zero_current_clients':sum(float(v.trace())<=1e-24 for v in currents)}
            if geometry[name]['orthogonal_error']>1e-5 or geometry[name]['pool_previous_residual']>1e-5:
                raise ArithmeticError('Invalid deployed subspace')
            if any(v['maximum_weight_roundoff']>1e-6 for v in rebase):
                raise ArithmeticError('Rebase changed the effective weight beyond FP32 tolerance')
        s['spectra_targets']=targets;s['spectra_pools']=pools
        s['spectra_geometry'].append({'task':t+1,'round':r+1,'layers':geometry,'seconds':time.perf_counter()-started})
        s['prepared_round']=(t,r)
        self.event('spectra_geometry',barycenter_fallbacks=sum(v['barycenter']['fallback'] for v in geometry.values()),
                   maximum_rebase_roundoff=max([v['maximum_weight_roundoff'] for g in geometry.values() for v in g['rebase']] or [0.]))

    def commit_fisher(self):
        s=self.s;entries=self.probe_gradients(archive=True)
        weights=torch.tensor([v['n'] for v in entries],dtype=torch.float64);weights/=weights.sum()
        record={}
        for name,pool in s['spectra_pools'].items():
            matrices=[]
            for item in entries:
                projected=[g.double()@pool for g in item['old'][name]]
                matrices.append(unit_trace(sum(g.T@g for g in projected)))
            mean=sum(float(p)*g for p,g in zip(weights,matrices))
            s['spectra_history'][name]=update_history(s['spectra_history'][name],pool,mean,self.settings['history_ema'])
            self.charge('spectra_task_end_fisher_upload',len(entries)*pool.shape[1]**2*8)
            record[name]={'trace':float(s['spectra_history'][name].trace()),
                          'nonzero_clients':sum(float(v.trace())>1e-24 for v in matrices)}
        s['spectra_commits'].append({'task':s['task']+1,'layers':record})
        s['history_probe_committed']=s['task']

    def fuse(self):
        if not self.spectral:return super().fuse()
        s=self.s;started=time.perf_counter();r=s['round'];t=s['task']
        origin=[b.detach().cpu().clone() for b in self.model.b_parameters()]
        baseline=[b.clone() for b in s['agg_B']]
        entries=s['spectra_endpoints'];expected=[k for k in range(self.cfg['clients']) if self.ids('train',t,k)]
        if [v['client'] for v in entries]!=expected:raise ValueError('Spectral endpoint/client mismatch')
        weights=torch.tensor(s['sizes'],dtype=torch.float64);weights/=weights.sum()
        deployed=[];layers={}
        for i,name in enumerate(self.model.adapters):
            target=s['spectra_targets'][name];records=[]
            for part in range(2):
                index=2*i+part
                ds=[v['B'][index].double()-origin[index].double() for v in entries]
                new,record=transport_increments(ds,weights,target['current'],target['history'],target['disagreement'],self.settings)
                old=sum(float(p)*d for p,d in zip(weights,ds))
                # An identity transport reproduces exact original FP32 FedAvg.
                value=origin[index].clone() if record['zero_update'] else (baseline[index].double()+new-old).float()
                if not torch.isfinite(value).all():raise ArithmeticError('Nonfinite deployed spectral increment')
                actual=value.double()-origin[index].double()
                record['actual_FP32_norm']=float(actual.norm())
                record['actual_FP32_history']=float((actual.T@actual@target['history']).trace())
                deployed.append(value);records.append(record)
            layers[name]=records
        super().fuse()
        with torch.no_grad():
            for b,v in zip(self.model.b_parameters(),deployed):b.copy_(v)
        s['spectra_rounds'].append({'task':t+1,'round':r+1,'layers':layers,'seconds':time.perf_counter()-started})
        s['spectra_endpoints']=[]
        self.event('spectra_aggregate',global_norm_caps=sum(v['global_scale']<1 for vv in layers.values() for v in vv))

    def step(self):
        s=self.s;phase=s['phase']
        if self.spectral and phase=='round_start' and s['round']<self.cfg['rounds']:
            if s.get('prepared_round')!=(s['task'],s['round']):self.prepare_round()
        if self.spectral and phase=='evaluated' and s.get('history_probe_committed')!=s['task']:
            self.commit_fisher()
        result=super().step()
        if phase=='committed':
            self.teacher=None;s['teacher_state']=None
            # The state transition was saved by the base lifecycle; save the
            # released anchor as well, so final/next-task reload has no stale copy.
            self.save()
        return result
