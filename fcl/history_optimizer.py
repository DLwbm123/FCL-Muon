"""Research updates: current direction state never contains historical correction."""
import math
import torch
from .spectral_qp import SolverConfig,solve_spectral_qp,solve_frobenius_qp,compact_polar,actual_update_certificate

METHODS=['mulora_fl_exact_polar','history_grad_muon','history_spectral','history_frobenius_qp']


def update_geometry(d):
    s=torch.linalg.svdvals(d);mass=s.sum();energy=s.square()
    if not float(mass):return {'fro':0.,'operator':0.,'stable_rank':0.,'effective_rank':0.}
    prob=s/mass
    return {'fro':float(d.norm()),'operator':float(s.max()),'stable_rank':float(energy.sum()/energy.max()),
            'effective_rank':float(torch.exp(-(prob*prob.clamp_min(1e-30).log()).sum())),
            'effective_rank_definition':'singular_value_amplitude_entropy'}


def propose(method,m,b,core,eta,lam,previous=None,solver=None,rho_factor=1.):
    solver=solver or SolverConfig()
    h=core['H'];f=core['F'];g=m+lam*(f+b@h);hh=lam*h
    if method=='history_spectral':d,diagnostic=solve_spectral_qp(g,hh,eta,previous,solver)
    elif method=='history_grad_muon':
        d=-eta*compact_polar(g)[0];diagnostic={'oracle':'exact_polar_corrected_gradient'}
    elif method=='mulora_fl_exact_polar':
        d=-eta*compact_polar(m)[0];diagnostic={'oracle':'exact_polar_current_direction'}
    elif method=='history_frobenius_qp':
        d,diagnostic=solve_frobenius_qp(g,hh,eta*rho_factor)
    else:raise ValueError(method)
    return d,diagnostic,g,hh


def paired_update_audit(m,b,core,eta,lam,a,config=None):
    result={};d0=-eta*compact_polar(m)[0]
    ds,_,_,_=propose('history_spectral',m,b,core,eta,lam,solver=config)
    g=m+lam*(core['F']+b@core['H'])
    matched=-ds.norm()*g/g.norm() if float(g.norm()) else torch.zeros_like(g)
    candidates={name:propose(name,m,b,core,eta,lam,solver=config,rho_factor=max(1,b.shape[-1]**.5))[0] for name in METHODS}
    candidates['matched_energy_gradient_direction']=matched
    candidates['matched_budget_frobenius_qp']=solve_frobenius_qp(g,lam*core['H'],float(ds.norm()))[0]
    for name,d in candidates.items():
        benefit=-float((m*d).sum());change=float(((core['F']+b@core['H'])*d).sum()+.5*((d@core['H'])*d).sum())
        factor=float((d*d0).sum()/d0.square().sum()) if float(d0.norm()) else 0.
        result[name]={**update_geometry(d),'predicted_current_gain':benefit,'historical_q_change':change,
                      'phi':-benefit+lam*change,'weight_space_fro':float((d@a).norm()),
                      'weight_space_operator':float(torch.linalg.matrix_norm(d@a,ord=2)),
                      'cosine_vs_muon':float((d*d0).sum()/(d.norm()*d0.norm()).clamp_min(1e-30)),
                      'best_uniform_scale_vs_muon':factor,'uniform_scale_residual':float((d-factor*d0).norm()/d.norm().clamp_min(1e-30))}
    return result


class HistoryOptimizer:
    def __init__(self,parameters,state,local_c,global_c,extension,client):
        self.parameters=list(parameters);self.state=state;self.local_c=local_c;self.global_c=global_c
        self.ext=extension;self.client=client;self.corrected=extension.provider=='fedmuon'
        self.diag=[];self.steps=0;self.audits=[]
    @torch.no_grad()
    def step(self,lr):
        updates=[]
        for i,b in enumerate(self.parameters):
            self.state[i].mul_(.9).add_(b.grad.float(),alpha=.1)
            m=self.state[i]-self.local_c[i]+self.global_c[i] if self.corrected else self.state[i]
            eta=lr*(max(b.shape)**.5 if self.corrected else 1.)
            core=self.ext.cores[i];lam=0. if self.ext.method=='mulora_fl_exact_polar' else self.ext.lam
            before=b.detach().clone();previous=self.ext.caches[self.client][i]
            d,diagnostic,g,hh=propose(self.ext.method,m,b,core,eta,lam,previous,self.ext.solver,self.ext.rho_factor)
            if self.ext.audit_enabled and self.client==0 and self.ext.task not in self.ext.audited_tasks and self.steps==0 and i==0:
                a=list(self.ext.model.adapters.values())[0].A
                self.audits.append(paired_update_audit(m,b,core,eta,lam,a,self.ext.solver))
                self.ext.audited_tasks.add(self.ext.task)
            b.add_(d);actual=b-before
            if self.ext.method=='history_spectral':
                proposal_diag=diagnostic['proposed_certificate'].copy()
                actual_diag=actual_update_certificate(g,hh,before,b,eta,self.ext.solver)
                diagnostic.update(actual_diag)
                diagnostic['proposed_certificate']=proposal_diag
                diagnostic['actual_certificate']=actual_diag.copy()
                diagnostic['application_roundoff_fro']=float((b.double()-before.double()-d.double()).norm())
                diagnostic['failure_stage']=('none' if actual_diag['converged'] else
                                             'parameter_addition' if proposal_diag['converged'] else 'proposal')
                diagnostic['failure_reason']=[f'{stage}_{check}' for stage,info in
                    [('proposed',proposal_diag),('actual',actual_diag)] for check in
                    ['gap_satisfied','feasible_within_tolerance'] if not info[check]]
                diagnostic['certificate_for']='actual_B_after_minus_before'
            self.ext.caches[self.client][i]=actual.clone()
            mu=-eta*compact_polar(m)[0];gradient=core['F']+before@core['H']
            gain=-float((m*mu).sum());curve=.5*float(((mu@core['H'])*mu).sum())
            q0=float(core['q0']+(core['F']*before).sum()+.5*((before@core['H'])*before).sum())
            change=float((gradient*actual).sum()+.5*((actual@core['H'])*actual).sum())
            self.diag.append({'layer':i,'step':self.steps,'eta':eta,'lambda':lam,**diagnostic,
                              'r_grad':lam*float(gradient.norm())/max(float(m.norm()),1e-30),
                              'r_curv':lam*curve/max(gain,1e-30),'history_linear_term':lam*float((gradient*mu).sum()),
                              'q_before':q0,'q_change':change,'actual_update':update_geometry(actual)})
            updates.append(actual)
        self.steps+=1
        return updates
    def flush(self,task,round_idx):
        ds=self.diag;n=len(ds)
        self.ext.event('history_solver_client',task=task,round=round_idx,client=self.client,updates=n,
            lambda_history=self.ext.lam,qp_certificate_applicable=self.ext.method=='history_spectral',mean_iterations=sum(d.get('iterations',0) for d in ds)/max(n,1),
            nonconverged=sum(not d.get('converged',True) for d in ds),zero_steps=sum(d.get('zero_step',d['actual_update']['fro']==0) for d in ds),
            max_relative_gap=max((d.get('relative_gap',0) for d in ds),default=0),
            max_feasibility_residual=max((d.get('feasibility_residual',0) for d in ds),default=0),
            max_objective=max((d.get('objective',0) for d in ds),default=0),
            solver_seconds=sum(d.get('elapsed_seconds',0) for d in ds),
            mean_r_grad=sum(d['r_grad'] for d in ds)/max(n,1),mean_r_curv=sum(d['r_curv'] for d in ds)/max(n,1),
            mean_history_q=sum(d['q_before'] for d in ds)/max(n,1),history_q_change_sum=sum(d['q_change'] for d in ds),
            first_step_per_layer=ds[:len(self.parameters)],paired_first_layer_audit=self.audits,
            # Preserve every failed step, not just the first update of each round.
            failed_updates=[d for d in ds if not d.get('converged',True)],
            failure_counts={key:sum(d.get('failure_stage')==key for d in ds) for key in ['proposal','parameter_addition']})
