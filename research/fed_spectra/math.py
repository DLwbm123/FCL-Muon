"""Small, coordinate-consistent spectral operators. All solves use FP64.

The historical matrix is a block-diagonal Fisher proxy on available training
inputs, not a bound on old-task classification risk. A Bures map targets each
client's regularized increment Gram, not the Gram of their signed average.
"""
import math
import torch


def sym(x):
    return (x + x.T) * .5


def psd_power(x, power, floor=0.):
    values, vectors = torch.linalg.eigh(sym(x.double()))
    values = values.clamp_min(floor)
    return sym((vectors * values.pow(power)) @ vectors.T)


def unit_trace(x):
    x = sym(x.double())
    trace = float(x.trace())
    return x / trace if trace > 1e-24 else torch.zeros_like(x)


def bures_distance(a, b):
    root = psd_power(a, .5)
    value = float(a.trace() + b.trace() - 2 * psd_power(root @ b @ root, .5).trace())
    return max(0., value)


def barycenter(matrices, weights, kind='bures', ridge=1e-5, iterations=60, tolerance=1e-7):
    """Positive definite Bures fixed point, with an explicit recorded fallback."""
    n = matrices[0].shape[0]
    eye = torch.eye(n, dtype=torch.float64, device=matrices[0].device) / n
    inputs = [(1-ridge)*unit_trace(a) + ridge*eye for a in matrices]
    mean = sym(sum(float(p)*a for p,a in zip(weights,inputs)))
    if float(mean.trace()) < ridge*.5 or not torch.isfinite(mean).all():
        raise ArithmeticError('Invalid spectral barycenter inputs')
    if kind == 'arithmetic':
        return unit_trace(mean), {'kind':kind, 'fallback':False, 'iterations':0, 'residual':0.}
    if kind != 'bures':
        raise ValueError('Unknown barycenter')
    x = mean.clone(); residual = float('inf')
    for step in range(iterations):
        root = psd_power(x, .5)
        inverse = psd_power(x, -.5, ridge*1e-6/n)
        middle = sum(float(p)*psd_power(root @ a @ root, .5) for p,a in zip(weights,inputs))
        new = sym(inverse @ middle @ middle @ inverse)
        residual = float((new-x).norm()/x.norm().clamp_min(1e-24))
        if not torch.isfinite(new).all():
            break
        x = new
        if residual <= tolerance:
            return unit_trace(x), {'kind':kind, 'fallback':False, 'iterations':step+1,
                                   'residual':residual, 'unscaled_trace':float(x.trace())}
    return unit_trace(mean), {'kind':'arithmetic', 'requested_kind':kind, 'fallback':True,
                             'iterations':step+1, 'residual':residual if math.isfinite(residual) else None}


def signed_disagreement(sketches, weights):
    mean = sum(float(p)*v.double() for p,v in zip(weights,sketches))
    return sym(sum(float(p)*(v.double()-mean).T@(v.double()-mean) for p,v in zip(weights,sketches)))


def gaussian_map(dimension, width, seed):
    # Separate generator: probes never consume the training RNG stream.
    return torch.randn(dimension,width,generator=torch.Generator().manual_seed(seed),dtype=torch.float64)/math.sqrt(width)


def orthogonal_pool(blocks, dimension, width, seed):
    """Ordered blocks; rank-revealing residual SVD, then seeded completion."""
    basis = torch.empty(dimension,0,dtype=torch.float64)
    for block in [*blocks, gaussian_map(dimension,width,seed)]:
        x = block.double().cpu()
        if not x.numel():
            continue
        for _ in range(2):
            x = x-basis@(basis.T@x)
        u,s,_ = torch.linalg.svd(x,full_matrices=False)
        take = min(width-basis.shape[1],int((s>1e-9*max(1.,float(s.max()))).sum()))
        basis = torch.cat([basis,u[:,:take]],dim=1)
        if basis.shape[1] == width:
            break
    if basis.shape[1] != width or float((basis.T@basis-torch.eye(width)).norm())>1e-8:
        raise ArithmeticError('Candidate pool lost orthogonality')
    return basis


def candidate_pool(previous, history, sketches, weights, width, current_rank, seed):
    d = previous.shape[0]
    factor = torch.cat([v.double()*math.sqrt(float(p)) for p,v in zip(weights,sketches)],dim=1)
    # Work in the smaller Gram when sketches are narrower than the input width.
    if factor.shape[1] < d:
        vals,vecs = torch.linalg.eigh(sym(factor.T@factor))
        current = factor@vecs[:,-current_rank:]
        current = current/vals[-current_rank:].clamp_min(1e-24).sqrt()
    else:
        _,vecs = torch.linalg.eigh(sym(factor@factor.T))
        current = vecs[:,-current_rank:]
    history_rank = max(0,width-previous.shape[1]-current_rank)
    if history_rank and float(history.trace())>1e-24:
        values,vectors = torch.linalg.eigh(sym(history.double()))
        historical = vectors[:,-history_rank:]*values[-history_rank:].clamp_min(0).sqrt()
    else:
        historical = torch.empty(d,0,dtype=torch.float64)
    # Include the previous space first; dynamic replacement then happens in M.
    return orthogonal_pool([previous,current,historical],d,width,seed)


def fisher_coefficients(logits, seed, temperature=2.):
    """Unbiased random contraction of the old-output softmax Fisher.

    E[aa^T]=diag(p)-pp^T. Independent signs per example avoid cross-example
    terms in expectation. Differentiating <a, logits/T>/sqrt(N) estimates the
    average parameter Fisher even when teacher and student logits coincide.
    """
    p = (logits.detach()/temperature).softmax(-1)
    signs = torch.randint(0,2,p.shape,generator=torch.Generator().manual_seed(seed),device='cpu').to(p)*2-1
    v = p.sqrt()*signs
    return v-p*v.sum(-1,keepdim=True)


def update_history(old, pool, new, decay):
    if float(new.trace()) <= 1e-24:
        return old.clone()
    lifted = sym(pool.double()@unit_trace(new)@pool.double().T)
    if float(old.trace()) <= 1e-24:
        return unit_trace(lifted)
    return unit_trace(decay*old.double()+(1-decay)*lifted)


def select_subspace(pool, current, history, disagreement, rank, lambda_old, lambda_dis):
    h = sym(pool.T@history@pool)
    objective = sym(current-lambda_old*h-lambda_dis*disagreement)
    values,vectors = torch.linalg.eigh(objective)
    z = vectors[:,-rank:]
    u = pool@z
    return u, {'selected_eigenvalues':values[-rank:].tolist(),
               'current_capture':float((z.T@current@z).trace()),
               'history_exposure':float((u.T@history@u).trace()),
               'signed_disagreement':float((z.T@disagreement@z).trace())}


@torch.no_grad()
def rebase_adapter(adapter, new_u):
    """Retain the exact off-subspace residual in the frozen Q/V weight.

    Active adaptation remains rank r; accumulated within-task updates can have
    higher rank. This is explicitly not lossless rank-r compression.
    """
    old_a = adapter.A.detach().double()
    new_a = new_u.T.to(adapter.A).contiguous()
    u = new_a.T.double()
    d = adapter.Bq.shape[0]
    reports = []
    for param,offset in [(adapter.Bq,0),(adapter.Bv,2*d)]:
        old = param.double()@old_a
        projected = (old@u).to(param)
        residual = old-projected.double()@new_a.double()
        before = adapter.base.weight[offset:offset+d].double()+old
        adapter.base.weight[offset:offset+d].add_(residual.to(adapter.base.weight))
        param.copy_(projected)
        after = adapter.base.weight[offset:offset+d].double()+param.double()@new_a.double()
        reports.append({'residual_fro':float(residual.norm()), 'old_update_fro':float(old.norm()),
                        'maximum_weight_roundoff':float((after-before).abs().max())})
    adapter.A.copy_(new_a)
    return reports


def spectrum(gram):
    values = torch.linalg.eigvalsh(sym(gram.double())).clamp_min(0)
    singular = values.sqrt()
    total = float(singular.sum())
    if total <= 1e-24:
        return {'singular_values':singular.tolist(),'effective_rank':0.,'condition':None,'fro':0.}
    probs = singular/total
    return {'singular_values':singular.tolist(),
            'effective_rank':float(torch.exp(-(probs*probs.clamp_min(1e-30).log()).sum())),
            'condition':float(singular[-1]/singular[0]) if float(singular[0])>1e-12*float(singular[-1]) else None,
            'fro':math.sqrt(float(values.sum()))}


def transport_increments(increments, weights, current, history, disagreement, settings):
    """Symmetric Gaussian OT map, clipped eigenvalues, and global norm cap.

    Sign flips or rotations of an eigendecomposition cannot change this map.
    Inputs and targets use the SAME active rank-r coordinate system. Amplitude
    comes from actual increments, never normalized gradient-energy units.
    """
    ds = [v.double() for v in increments]
    n = current.shape[0];eye = torch.eye(n,dtype=torch.float64)
    energy = sum(float(p)*float(v.square().sum()) for p,v in zip(weights,ds))
    baseline = sum(float(p)*v for p,v in zip(weights,ds))
    if energy <= 1e-24:
        return baseline, {'zero_update':True,'clients':[],'global_scale':1.,'fallback':False}
    penalty = eye+settings['lambda_hist']*history+settings['lambda_dis']*disagreement
    shrink = psd_power(penalty,-.5,1e-12)
    shape = unit_trace(shrink@current@shrink)
    if float(shape.trace()) <= 1e-24:
        shape = eye/n
    ridge = settings['transport_ridge']
    target = energy*((1-ridge)*shape+ridge*eye/n)
    transformed=[];records=[]
    for delta in ds:
        gram = sym(delta.T@delta);amount=float(gram.trace())
        if amount <= 1e-24:
            transformed.append(delta);records.append({'zero_update':True});continue
        regularized = gram+ridge*amount*eye/n
        root = psd_power(regularized,.5)
        inverse = psd_power(regularized,-.5,ridge*amount/n*1e-6)
        operator = sym(inverse@psd_power(root@target@root,.5)@inverse)
        vals,vectors=torch.linalg.eigh(operator)
        clipped=vals.clamp(settings['transport_clip_min'],settings['transport_clip_max'])
        operator=sym((vectors*clipped)@vectors.T)
        value=delta@operator;after=sym(value.T@value)
        if not torch.isfinite(value).all():raise ArithmeticError('Nonfinite transport')
        transformed.append(value)
        records.append({'zero_update':False,'before':spectrum(gram),'after':spectrum(after),
                        'bures_before':bures_distance(gram,target),'bures_after':bures_distance(after,target),
                        'scale_min':float(clipped.min()),'scale_max':float(clipped.max()),'scale_mean':float(clipped.mean()),
                        'clip_count':int((vals!=clipped).sum()),'modes':n})
    aggregated=sum(float(p)*v for p,v in zip(weights,transformed))
    max_norm=settings['global_norm_cap']*float(baseline.norm())
    scale=min(1.,max_norm/max(float(aggregated.norm()),1e-30))
    aggregated*=scale
    before=baseline.T@baseline;after=aggregated.T@aggregated
    return aggregated, {'zero_update':False,'clients':records,'global_scale':scale,'fallback':False,
                        'target_eigenvalues':torch.linalg.eigvalsh(target).tolist(),
                        'baseline':spectrum(before),'deployed':spectrum(after),
                        'aggregate_bures_before':bures_distance(before,target),'aggregate_bures_after':bures_distance(after,target),
                        'history_before':float((before@history).trace()),'history_after':float((after@history).trace()),
                        'signed_coherence':float(baseline.square().sum())/energy}
