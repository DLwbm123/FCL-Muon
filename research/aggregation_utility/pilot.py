"""Task-start supervised subspace selection around the unchanged Head-RegMean trainer."""
import contextlib
import time
import torch
from torch.nn import functional as F
from fcl.checkpoint import rng_state, restore_rng, atomic_save
from research.head_regmean.confirmation import ConfirmationPilot
from .selection import candidate_pool, quantities, scalar_record, fit_search, certify


def split_indices(indices, labels, task, client, cpt, seed, fraction):
    """Stratified deterministic local split; singletons remain in fit, never duplicated."""
    fit, check = [], []
    generator = torch.Generator().manual_seed(seed+10007*task+101*client)
    for label in range(task*cpt, (task+1)*cpt):
        ids = sorted(i for i in indices if int(labels[i]) == label)
        if not ids:
            continue
        permutation = torch.randperm(len(ids), generator=generator).tolist()
        shuffled = [ids[i] for i in permutation]
        n_check = min(len(ids)-1, max(1, int(len(ids)*fraction))) if len(ids)>1 else 0
        check.extend(shuffled[:n_check]);fit.extend(shuffled[n_check:])
    assert set(fit).isdisjoint(check) and sorted(fit+check)==sorted(indices)
    return sorted(fit), sorted(check)


@contextlib.contextmanager
def passive_probe(model):
    state = rng_state()
    modes = {m:m.training for m in model.modules()}
    flags = {p:p.requires_grad for p in model.parameters()}
    try:
        model.eval()
        for p in flags:
            p.requires_grad_(False)
        yield
    finally:
        model.zero_grad(set_to_none=True)
        for p,flag in flags.items():
            p.requires_grad_(flag)
        for m,mode in modes.items():
            m.training=mode
        restore_rng(state)


@contextlib.contextmanager
def zero_probe(model, pools):
    """Only temporary zero B_probe tensors require gradients; forward stays unchanged."""
    probes, handles = {}, []
    try:
        for name, adapter in model.adapters.items():
            d, width = pools[name].shape
            probes[name] = [torch.zeros(d,width,device=pools[name].device,requires_grad=True) for _ in range(2)]
            def hook(_module, inputs, output, key=name):
                low = inputs[0] @ pools[key]
                q,k,v = output.chunk(3,dim=-1)
                bq,bv = probes[key]
                return torch.cat((q+F.linear(low,bq), k, v+F.linear(low,bv)),dim=-1)
            handles.append(adapter.register_forward_hook(hook))
        yield probes
    finally:
        for handle in handles:
            handle.remove()


class UtilityPilot(ConfirmationPilot):
    def initialize_task(self):
        super().initialize_task()
        if self.s['task']==0 or self.cfg['selection_mode']=='prefix':
            return
        t=self.s['task'];settings=self.cfg['selection'];started=time.time()
        modes={'rrs':'federated','central':'central','federated':'federated','adamw':'adamw'}
        mode=modes[self.cfg['selection_mode']]
        self.event('selection_start',selection_mode=self.cfg['selection_mode'])
        heads=[r.detach().clone() for r in self.model.rows]
        pools, z0s, histories = {}, {}, {}
        original_bases={i:a.A.detach().clone() for i,a in self.model.adapters.items()}
        for name,a in self.model.adapters.items():
            h=self.s['history'].raw[name].to(self.device)
            pools[name],z0s[name],_=candidate_pool(self.s['pending'][name].to(self.device),h,a.A,settings['candidate_width'])
            histories[name]=pools[name].double().T @ h.double() @ pools[name].double()
        splits=[split_indices(self.ids('train',t,k),self.manifest['labels']['train'],t,k,self.cfg['cpt'],
                              settings['split_seed'],settings['check_fraction']) for k in range(self.cfg['clients'])]
        fit_counts=torch.tensor([len(v[0]) for v in splits],dtype=torch.float32,device=self.device)
        check_counts=torch.tensor([len(v[1]) for v in splits],dtype=torch.float32,device=self.device)
        if float(fit_counts.sum())<=0 or float(check_counts.sum())<=0:
            raise ValueError('Empty fit/check split')
        fit_weights=fit_counts/fit_counts.sum();check_weights=check_counts/check_counts.sum()
        with passive_probe(self.model):
            prototype,proto_counts=self.prototypes(splits)
            fit=self.projected_gradients(pools,prototype,splits,0)
            check=self.projected_gradients(pools,prototype,splits,1)
            self.charge('candidate_pool_broadcast',self.cfg['clients']*sum(p.numel()*p.element_size() for p in pools.values()))
            self.charge('prototype_head_broadcast',self.cfg['clients']*prototype.numel()*prototype.element_size())
            records={}
            selected={}
            # No check gradients enter fit_search. Exactly one candidate is checked per layer.
            for name in pools:
                self.budget()
                if self.cfg['selection_mode']=='rrs':
                    q=quantities(z0s[name],fit[name],fit_weights,histories[name],mode)
                    rec={'accepted':False,'reason':'original_RRS_control','baseline':scalar_record(q)}
                    selected[name]=original_bases[name]
                else:
                    z,search=fit_search(fit[name],fit_weights,histories[name],z0s[name],mode,settings,self.budget)
                    selected[name],rec=certify(pools[name],z,z0s[name],fit[name],fit_weights,check[name],check_weights,
                                               histories[name],mode,settings)
                    rec['search']=search
                    if not rec['accepted']:
                        selected[name]=original_bases[name]  # exact original bytes on rejection
                records[name]=rec
                self.event('selection_layer',layer=name,accepted=rec['accepted'])
            private={'task':t,'mode':self.cfg['selection_mode'],'splits':splits,
                     'prototype':prototype.cpu(),'prototype_counts':proto_counts.cpu(),
                     'fit_weights':fit_weights.cpu(),'check_weights':check_weights.cpu(),
                     'pools':{k:v.cpu() for k,v in pools.items()},'history_projected':{k:v.cpu() for k,v in histories.items()},
                     'fit_gradients':{k:v.cpu() for k,v in fit.items()},'check_gradients':{k:v.cpu() for k,v in check.items()},
                     'original_bases':{k:v.cpu() for k,v in original_bases.items()},
                     'selected_bases':{k:v.cpu() for k,v in selected.items()},'records':records}
            atomic_save(private,self.out/f'selection_t{t+1}_private.pt')
        with torch.no_grad():
            for name,a in self.model.adapters.items():a.A.copy_(selected[name])
        assert all(torch.equal(before,after) for before,after in zip(heads,self.model.rows))
        assert all(not torch.count_nonzero(b) for b in self.model.b_parameters())
        record={'task':t,'mode':self.cfg['selection_mode'],'fit_samples':int(fit_counts.sum()),
                'check_samples':int(check_counts.sum()),'prototype_counts':proto_counts.cpu().tolist(),
                'fit_client_counts':fit_counts.cpu().tolist(),'check_client_counts':check_counts.cpu().tolist(),
                'accepted_layers':sum(v['accepted'] for v in records.values()),'layers':records,
                'seconds':time.time()-started,'rng_restored':True,'formal_head_unchanged':True}
        self.s.setdefault('selection_records',[]).append(record)
        self.event('selection_complete',accepted_layers=record['accepted_layers'],seconds=record['seconds'])

    def prototypes(self,splits):
        d=self.model.backbone.num_features;cpt=self.cfg['cpt'];t=self.s['task']
        sums=torch.zeros(cpt,d,dtype=torch.float64,device=self.device)
        counts=torch.zeros(cpt,dtype=torch.int64,device=self.device)
        with torch.no_grad():
            for k,(fit,_) in enumerate(splits):
                for x,y in self.loader(self.raw[0],fit,self.cfg['batch']):
                    self.budget();self.check_labels(y)
                    z=F.normalize(self.model.backbone.forward_features(self.test_tf(x.to(self.device)))[:,0],dim=1).double()
                    y=y.to(self.device)-t*cpt
                    # Deterministic class reductions avoid CUDA index_add atomics.
                    for label in range(cpt):
                        mask=y==label
                        sums[label]+=z[mask].sum(0);counts[label]+=mask.sum()
                self.charge('prototype_statistics_upload',sums.numel()*8+counts.numel()*8)
        if not bool((counts>0).all()) or not torch.isfinite(sums).all():
            raise ValueError('Missing or invalid fit class prototype')
        self.s.setdefault('probe_records',[]).append({'task':t,'kind':'prototype_fit','samples':int(counts.sum())})
        return F.normalize(sums/counts[:,None],dim=1).float(),counts

    def projected_gradients(self,pools,prototype,splits,part):
        values={name:[] for name in pools};t=self.s['task'];cpt=self.cfg['cpt']
        with zero_probe(self.model,pools) as probes:
            for k,split in enumerate(splits):
                ids=split[part]
                for pair in probes.values():
                    for p in pair:p.grad=None
                loss_sum=0.
                for x,y in self.loader(self.raw[0],ids,self.cfg['batch']):
                    self.budget();self.check_labels(y)
                    z=self.model.backbone.forward_features(self.test_tf(x.to(self.device)))[:,0]
                    logits=self.cfg['cosine_scale']*F.linear(F.normalize(z,dim=1),prototype)
                    loss=F.cross_entropy(logits,y.to(self.device)-t*cpt,reduction='sum')
                    if not torch.isfinite(loss):raise ArithmeticError('Nonfinite probe loss')
                    (loss/len(ids)).backward();loss_sum+=float(loss.detach())
                for name,pair in probes.items():
                    gradients=[p.grad.detach().clone() if ids else torch.zeros_like(p) for p in pair]
                    if any(not torch.isfinite(g).all() for g in gradients):raise ArithmeticError('Nonfinite projected gradient')
                    values[name].append(torch.stack(gradients))
                self.s.setdefault('probe_records',[]).append({'task':t,'kind':'fit_gradient' if part==0 else 'check_gradient',
                                                              'client':k,'samples':len(ids),'mean_loss':loss_sum/len(ids) if ids else None})
                self.event('probe_client',split='fit' if part==0 else 'check',client=k,samples=len(ids))
        result={name:torch.stack(v).detach() for name,v in values.items()}
        self.charge('probe_fit_gradient_upload' if part==0 else 'probe_check_gradient_upload',
                    sum(v.numel()*v.element_size() for v in result.values()))
        return result
