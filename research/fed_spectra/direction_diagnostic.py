"""A fixed-angle reflection and disjoint confirmation probes; no strength search."""
import torch


def confirmation_probes(groups,seed,selection):
    result=[]
    for client,(ids,used) in enumerate(zip(groups,selection)):
        pool=sorted(set(ids)-set(used))
        g=torch.Generator().manual_seed(61030+1009*seed+31*client)
        result.append([pool[i] for i in torch.randperm(len(pool),generator=g).tolist()[:32]])
    return result


def reflected_proposal(origin,baseline,positive):
    """Reflect each proposed update about its FedAvg axis, retaining norm/angle."""
    if not len(origin)==len(baseline)==len(positive) or not origin:raise ValueError('Mismatched endpoints')
    result=[];records=[]
    for o,b,p in zip(origin,baseline,positive):
        if not o.shape==b.shape==p.shape:raise ValueError('Mismatched endpoint shapes')
        o,b,p=o.double(),b.double(),p.double()
        if not all(torch.isfinite(v).all() for v in (o,b,p)):raise ArithmeticError('Nonfinite endpoint')
        u=b-o;v=p-o;n=float(u.norm())
        if n==0:
            if float(v.norm())!=0:raise ValueError('Nonzero proposal for zero baseline')
            negative=b.float();norm_error=0.;distance_error=0.
        else:
            reflected=2*float((u*v).sum()/u.square().sum())*u-v
            negative=(o+reflected).float();actual=negative.double()-o
            norm_error=abs(float(actual.norm())-n)/n
            distance_error=abs(float((negative.double()-b).norm())-float((p-b).norm()))/n
            if norm_error>5e-4 or distance_error>5e-4:raise ArithmeticError('Reflection FP32 mismatch')
        result.append(negative)
        records.append({'norm_relative_error':norm_error,'correction_length_relative_error':distance_error})
    return result,records


def choose_direction(grads,baseline,positive,negative):
    dots={key:sum(float((g.double()*(v.double()-b.double())).sum()) for g,b,v in zip(grads,baseline,values))
          for key,values in [('positive',positive),('negative',negative)]}
    if not all(torch.isfinite(torch.tensor(v)) for v in dots.values()):raise ArithmeticError('Nonfinite directional derivative')
    return ('negative' if dots['negative']<dots['positive']-1e-12 else 'positive'),dots


def deployed_record(proposal,origin,baseline,negative,direction):
    """Describe the selected FP32 endpoint, never copy positive spectral effects."""
    import copy
    record=copy.deepcopy(proposal)
    index=0
    for pair in record['layers'].values():
        for r in pair:
            o,b,n=origin[index].double(),baseline[index].double(),negative[index].double();index+=1
            old=b-o;actual=n-o;norm=float(old.norm());outnorm=float(actual.norm())
            error=abs(outnorm-norm)/max(norm,1e-30)
            correction=float((n-b).norm())/max(norm,1e-30)
            r.update(output_norm=outnorm,norm_relative_error=error,relative_correction=correction,
                actual_FP32_norm=outnorm,actual_FP32_norm_relative_error=error,actual_FP32_relative_correction=correction,
                cosine_with_fedavg=float((actual*old).sum())/max(norm*outnorm,1e-30) if norm else 1.,
                direction=direction,target_energy_role='positive_proposal_reference_only')
            if 'singular_values_before' in r and norm:
                before=torch.linalg.svdvals(old);after=torch.linalg.svdvals(actual)
                active=before>1e-12*before[0]
                energy=after[active].square();energy=energy/energy.sum()
                r.update(singular_values_after=after.tolist(),energy_after=energy.tolist(),
                    effective_energy_rank_after=float(torch.exp(-(energy*energy.clamp_min(1e-300).log()).sum())),
                    singular_scale_min=float((after[active]/before[active]).min()),
                    singular_scale_max=float((after[active]/before[active]).max()),
                    after_spectrum_measurement='descending SVD of deployed FP32 update')
                if direction=='matrix_tangent':
                    active_after=after>1e-12*after[0]
                    energy=after[active_after].square();energy=energy/energy.sum()
                    r.update(energy_after=energy.tolist(),retained_rank_before=int(active.sum()),
                        retained_rank_after=int(active_after.sum()),spectral_support_preserved=False,
                        effective_energy_rank_after=float(torch.exp(-(energy*energy.clamp_min(1e-300).log()).sum())))
            if error>5e-4:raise ArithmeticError('Reflected record norm mismatch')
    return record


def reflected_record(proposal,origin,baseline,negative):
    return deployed_record(proposal,origin,baseline,negative,'negative_reflection')
