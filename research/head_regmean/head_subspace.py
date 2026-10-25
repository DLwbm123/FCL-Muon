"""Projected new-head learning and RegMean restricted to the old-head complement."""
import math
import torch
from torch.nn import functional as F
from fcl.math import ClientMuon
from fcl.checkpoint import atomic_save
from .old_response import OldResponsePilot
from .solver import shrink_gram,sphere_solve,certify_sphere,objective


def head_bases(old_rows):
    old=old_rows.detach().cpu().double();n,d=old.shape
    if not 0<n<d:raise ValueError('No nonempty old-head complement')
    q,r=torch.linalg.qr(old.T,mode='complete')
    tol=max(n,d)*torch.finfo(torch.float64).eps*torch.linalg.norm(r,2)
    if bool((r[:n,:].diag().abs()<=tol).any()):raise ValueError('Old heads are not full row rank')
    pivots=q.abs().argmax(dim=0);sign=q[pivots,torch.arange(d)].sign()
    q=q*sign[None,:]
    return q[:,:n].contiguous(),q[:,n:].contiguous()


def project_rows(rows,old_basis,preserve_norm=False):
    x=rows.double();q=old_basis.to(device=x.device,dtype=x.dtype)
    projected=x-(x@q)@q.T
    if preserve_norm:
        norms=projected.norm(dim=1,keepdim=True)
        if bool((norms<=1e-12).any()):raise ValueError('New head has no usable complementary component')
        projected=projected*x.norm(dim=1,keepdim=True)/norms
    return projected.to(rows.dtype)


def cosine_leak(rows,old_basis):
    u=F.normalize(rows.float(),dim=1).double()
    return float((u@old_basis.to(u.device)).norm(dim=1).max())


def aggregate_subspace(heads,raw_grams,sizes,old_basis,new_basis):
    # Shrink in the original feature coordinates before restricting the feasible
    # set. Shrinking a rotated Gram would silently change the old objective.
    if not heads or len(heads)!=len(raw_grams) or len(heads)!=len(sizes) or min(sizes)<=0:raise ValueError('Invalid endpoint collection')
    if any(not torch.isfinite(w).all() or bool((w.norm(dim=1)<=1e-12).any()) for w in heads):raise ValueError('Invalid cosine head')
    gs=[shrink_gram(g,.5)/sum(sizes) for g in raw_grams]
    h=sum(gs);q=sum(F.normalize(w.float(),dim=1).double()@g for w,g in zip(heads,gs))
    basis=new_basis.double();hr=basis.T@h@basis;qr=q@basis
    u,lam,lower,info=sphere_solve(hr,qr)
    raw=torch.zeros_like(heads[0],dtype=torch.float32)
    for w,n in zip(heads,sizes):raw.add_(w.float(),alpha=n/sum(sizes))
    norms=raw.double().norm(dim=1).clamp_min(1e-12)
    deployed=((u@basis.T)*norms[:,None]).float()
    actual=F.normalize(deployed,dim=1).double();reduced=actual@basis
    baseline=F.normalize(project_rows(raw,old_basis),dim=1).double()@basis
    cert=certify_sphere(reduced,hr,qr,lam,lower)
    leak=cosine_leak(deployed,old_basis)
    info.update(deployed_fp32=cert,old_span_cosine_leak_max=leak,
        complement_dimension=basis.shape[1],old_span_rank=old_basis.shape[1],alpha=.5,
        raw_moment_samples=sum(sizes),baseline_objective=float(objective(baseline,hr,qr).sum()),
        regmean_objective=float(objective(reduced,hr,qr).sum()),
        objective_note='Original-coordinate alpha.5 shrunk response objective restricted to old-head orthogonal complement; certificate is for this new feasible set, not the previous unconstrained problem.')
    if not cert['passed'] or leak>2e-6:raise ArithmeticError(('Subspace deployment qualification failed',info))
    if info['regmean_objective']>info['baseline_objective']+1e-6*(1+abs(info['baseline_objective'])):
        raise ArithmeticError('Subspace solve worse than feasible projected FedAvg')
    return deployed,raw,info


class HeadSubspacePilot(OldResponsePilot):
    def setup_subspace(self):
        old=torch.cat(list(self.model.rows[:-1]))
        qo,qn=head_bases(old)
        self.s['head_subspace']={'old_basis':qo,'new_basis':qn,'rank':qo.shape[1],
            'dimension':qn.shape[1],'initialized_from_shared_source':True}
        self.old_basis=qo.to(self.device)
        with torch.no_grad():self.model.rows[-1].copy_(project_rows(self.model.rows[-1],self.old_basis,True))
        assert cosine_leak(self.model.rows[-1],self.old_basis)<=2e-6

    def backward_loss(self,x,y):
        expected=self.expected_first;self.expected_first=None
        loss=super().backward_loss(x,y)
        with torch.no_grad():
            grad=self.model.rows[-1].grad;grad.copy_(project_rows(grad,self.old_basis))
        if expected is not None:
            assert torch.equal(x.cpu(),expected['x']) and torch.equal(y.cpu(),expected['y'])
            assert all(torch.equal(v.grad.cpu(),g) for v,g in zip(self.model.b_parameters(),expected['B']))
            assert torch.equal(self.model.rows[-1].grad.cpu(),expected['head'])
            self.first_batch_verified=True
            self.event('actual_first_batch_verified',head_complement_projected=True,
                       initial_preservation_loss_zero=self.s['preservation_training']['mse_sum']==0.)
        return loss

    def train_client(self):
        s=self.s;k=s['next_client'];t=s['task'];r=s['round'];bs=self.model.b_parameters()
        global_b=[b.detach().clone() for b in bs];global_head=self.model.rows[-1].detach().clone()
        old_heads=[v.detach().clone() for v in self.model.rows[:-1]]
        ids=self.ids('train',t,k);n=len(ids)
        if n:
            dl=self.loader(self.raw[0],ids,self.cfg['batch'],True);count=self.cfg['epochs']*len(dl)
            momentum=[v.to(self.device) for v in s['client_states'][k]['m']]
            opt=ClientMuon(bs,momentum,[],[],False)
            headopt=torch.optim.AdamW([self.model.rows[-1]],lr=.001,weight_decay=0)
            steps=seen=0;loss_sum=0.;max_leak=0.
            for _ in range(self.cfg['epochs']):
                for x,y in dl:
                    self.budget();self.check_labels(y)
                    lr=self.cfg['lr']*.5*(1+math.cos(math.pi*(r*count+steps)/max(self.cfg['rounds']*count-1,1)))
                    self.model.zero_grad(set_to_none=True)
                    loss=self.backward_loss(self.train_tf(x.to(self.device)),y.to(self.device))
                    opt.step(lr);headopt.step()
                    with torch.no_grad():self.model.rows[-1].copy_(project_rows(self.model.rows[-1],self.old_basis))
                    if bool((self.model.rows[-1].norm(dim=1)<=1e-12).any()):raise ArithmeticError('Zero projected head row')
                    leak=cosine_leak(self.model.rows[-1],self.old_basis);max_leak=max(max_leak,leak)
                    if leak>2e-6:raise ArithmeticError('Local new head escaped old-head complement')
                    steps+=1;seen+=len(y);loss_sum+=float(loss.detach())*len(y)
            assert steps==count and seen==n*self.cfg['epochs']
            s['client_states'][k]={'m':[v.cpu() for v in momentum]};gram=self.local_gram(k)
            assert all(torch.equal(a,b) for a,b in zip(old_heads,self.model.rows[:-1]))
            with torch.no_grad():
                for dst,b in zip(s['agg_B'],bs):dst.add_(b.detach().cpu(),alpha=n/len(self.ids('train',t)))
            s['heads'].append(self.model.rows[-1].detach().cpu().clone());s['grams'].append(gram);s['sizes'].append(n)
            rec={'task':t,'round':r,'client':k,'N':n,'visited':seen,'steps':steps,'lr_last':lr,
                 'loss':loss_sum/seen,'head_old_span_cosine_leak_max':max_leak}
            s['training_records'].append(rec);self.event('client',**rec)
            self.charge('B_upload',sum(b.numel()*4 for b in bs));self.charge('current_head_upload',global_head.numel()*4)
        with torch.no_grad():
            for b,v in zip(bs,global_b):b.copy_(v)
            self.model.rows[-1].copy_(global_head)
        s['next_client']+=1;s['phase']='local_done' if s['next_client']==self.cfg['clients'] else 'clients'

    def fuse(self):
        s=self.s;basis=s['head_subspace']
        if any(not torch.isfinite(b).all() for b in s['agg_B']):raise ArithmeticError('Nonfinite B aggregate')
        head,raw,cert=aggregate_subspace(s['heads'],s['grams'],s['sizes'],basis['old_basis'],basis['new_basis'])
        with torch.no_grad():
            for b,v in zip(self.model.b_parameters(),s['agg_B']):b.copy_(v)
            self.model.rows[-1].copy_(head)
        s['certificates'].append({'task':s['task'],'round':s['round'],**cert})
        atomic_save({'task':s['task'],'round':s['round'],'B':s['agg_B'],'head':head,
                     'raw_FedAvg_head':raw,'certificate':cert},self.out/f"round_t{s['task']}_r{s['round']+1}.pt")
        self.event('aggregate',certificate=cert)
        s.update(round=s['round']+1,heads=[],grams=[],sizes=[],agg_B=[],next_client=0,phase='round_start')
