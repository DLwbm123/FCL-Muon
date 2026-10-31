"""Unregistered mainline integration draft; do not run before preregistration."""
import math
import time
import torch
from research.head_regmean.pilot import Pilot
from research.continual_muon.actual_step_server_draft import actual_step, deployed_parameters
from research.continual_muon.dual_fit_gradients import gradients
from research.continual_muon.muon_proxy_draft import history_damped_direction


class MainlinePilot(Pilot):
    """Adds one actual-update constraint after a genuine client Muon round."""
    def fuse(self):
        if self.s['task']!=2 or self.s['phase']!='local_done':
            raise ValueError('mainline draft only supports a complete current-task round')
        kcount=self.cfg['clients'];round_no=self.s['round']
        groups=self.manifest['partitions']['train'][2]
        if (len(groups)!=kcount or set(self.s['client_states'])!=set(range(kcount)) or
                any(not group for group in groups)):
            raise ValueError('client source incomplete')
        before=[v.detach().cpu().clone() for v in self.model.b_parameters()]
        if len(before)!=2*len(self.model.adapters):raise ValueError('adapter tensor count mismatch')
        super().fuse() # actual B aggregation and head fusion; no optimizer step here
        baseline=[v.detach().cpu().clone() for v in self.model.b_parameters()]
        updates=[b.double()-a.double() for a,b in zip(before,baseline)]
        pairs=[(sorted(group)[:32],sorted(group)[:32]) for group in groups]
        probe_rng=__import__('fcl.checkpoint',fromlist=['rng_state']).rng_state()
        ga,gc,cost=gradients(self,pairs,0)
        from scripts.real_boundary_worker import same_rng
        if not same_rng(probe_rng,__import__('fcl.checkpoint',fromlist=['rng_state']).rng_state()):
            raise ValueError('server probe changed training RNG')
        count=sum(map(len,groups));weights=[len(g)/count for g in groups]
        histories=[];directions=[];proxy_info=[]
        start=time.time()
        for j,u in enumerate(updates):
            self.budget();name=list(self.model.adapters)[j//2]
            a=self.model.adapters[name].A.detach().cpu().double()
            h=a@self.s['history'].raw[name].double()@a.T
            ms=[self.s['client_states'][k]['m'][j] for k in range(kcount)]
            direction,info=history_damped_direction(ms,weights,h)
            histories.append(h);directions.append(direction);proxy_info.append(info)
        corrected,receipt=actual_step(updates,directions,ga,gc,histories,radius=.1,history_tolerance=1e-10)
        if receipt['accepted']:
            deployed,deploy_receipt=deployed_parameters(before,baseline,corrected,ga,gc,histories,
                                                        radius=.1,tolerance=1e-10)
            receipt={'accepted':deploy_receipt['accepted'],'reason':deploy_receipt['reason'],
                     'update_space':receipt,'physical_deployment':deploy_receipt}
        else:
            deployed=[v.clone() for v in baseline]
        with torch.no_grad():
            for parameter,value in zip(self.model.b_parameters(),deployed):parameter.copy_(value)
        if not receipt['accepted'] and any(not torch.equal(v.detach().cpu(),b) for v,b in zip(self.model.b_parameters(),baseline)):
            raise ValueError('fallback does not equal actual FedAvg')
        if receipt['accepted'] and any(not torch.isfinite(v).all() for v in self.model.b_parameters()):
            raise ValueError('nonfinite actual server candidate')
        spectral_start=time.time();spectra=[]
        for current,base in zip(self.model.b_parameters(),baseline):
            delta=current.detach().cpu().double()-base.double()
            singular=torch.linalg.svdvals(delta);energy=float((singular*singular).sum())
            spectra.append({'frobenius':math.sqrt(energy),
                            'stable_rank':energy/max(float(singular.max())**2,1e-30)})
        spectral_seconds=time.time()-spectral_start
        bytes_per_model=sum(v.numel()*v.element_size() for v in self.model.b_parameters())
        record={'round':round_no,'decision':receipt,'fit_probe_cost':cost,
                'proxy_nonzero_matrices':sum(float(torch.linalg.vector_norm(d))>0 for d in directions),
                'ssb_calls':11*len(proxy_info),
                'theoretical_client_momentum_bytes':kcount*bytes_per_model,
                'theoretical_fit_two_gradient_bytes':2*kcount*bytes_per_model,
                'measured_network_bytes':None,'operator_seconds':time.time()-start,
                'correction_spectra':spectra,'svd_calls':len(spectra),'svd_seconds':spectral_seconds}
        if not math.isfinite(record['operator_seconds']):raise ValueError('invalid timing')
        self.s.setdefault('mainline_server_records',[]).append(record)
