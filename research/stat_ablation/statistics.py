"""Common-coordinate sufficient statistics and finite-response transport.

Only decoded FP32 packets enter the server. Selected's codec backward is an
identity straight-through surrogate; acceptance always uses the actual codec.
"""
import struct
import torch
from research.rrs_compression.codec import encode, decode


def codec_matrix(matrix, rank, track=False):
    packet=encode(matrix,rank=rank)
    decoded,record=decode(packet)
    decoded=decoded.to(matrix)
    if track:decoded=decoded+matrix-matrix.detach()
    return decoded,packet,record


def projection(d,q,seed=530911):
    g=torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(d,min(d,q),generator=g,dtype=torch.float64),mode='reduced').Q


def fp32(x):return x.float().double()


def local_packet(z0,z,labels,classes,R,rank=64,scale=1.,track=False):
    if z.shape!=z0.shape or len(z)!=len(labels) or not len(z):raise ValueError('unpaired statistics')
    if bool(((labels<0)|(labels>=classes)).any()):raise ValueError('noncurrent statistics label')
    z0=z0.double();z=z.double();R=R.to(z);h=z0@R;delta=z-z0
    hm=h.mean(0);dm=delta.mean(0);hc=h-hm;dc=delta-dm
    n=torch.tensor(len(z)*scale,device=z.device,dtype=torch.float64)
    counts=torch.stack([(labels==c).sum() for c in range(classes)]).double()*scale
    means=torch.stack([z[labels==c].mean(0) if bool(counts[c]) else torch.zeros_like(z[0]) for c in range(classes)])
    residual=z-means[labels]
    scatter,wire,record=codec_matrix(residual.T@residual*scale,rank,track)
    packet={'n':n,'hm':fp32(hm),'dm':fp32(dm),'hh':fp32(hc.T@hc*scale),
            'dh':fp32(dc.T@hc*scale),'counts':counts,'means':fp32(means),'scatter':scatter}
    # Actual formal packet: int64 counts, FP32 means/moments and codec bytes.
    if scale==1 and not track:
        head=struct.pack('<q',len(z))+counts.long().cpu().numpy().astype('<i8').tobytes()
        body=b''.join(packet[k].detach().float().cpu().numpy().astype('<f4').tobytes()
                      for k in ('hm','dm','hh','dh','means'))
        wire=head+body+wire
    else:wire=b''
    record.update(samples=len(z),scale=scale)
    return packet,wire,record


def merge(packets,old,R,ridge_u=.1,ridge_s=.1,floor=1e-6,mode="full",return_map=False):
    if mode not in ('identity','full','mean_only'):raise ValueError('unknown transport mode')
    if not packets:raise ValueError('no statistics')
    # Identity accepts ONLY class sufficient statistics; no untransmitted
    # response moments are read, including for certificates.
    if mode=='identity':
        template=packets[0]['means'];d=template.shape[1];q=R.shape[1]
        U=torch.zeros(d,q,device=template.device,dtype=template.dtype)
        T=torch.eye(d,device=template.device,dtype=template.dtype)
        b=torch.zeros(d,device=template.device,dtype=template.dtype);lu=torch.zeros((),device=template.device,dtype=template.dtype)
    else:
        n=sum(p['n'] for p in packets)
        hm=sum(p['n']*p['hm'] for p in packets)/n;dm=sum(p['n']*p['dm'] for p in packets)/n
        hh=sum(p['hh']+p['n']*torch.outer(p['hm']-hm,p['hm']-hm) for p in packets)
        dh=sum(p['dh']+p['n']*torch.outer(p['dm']-dm,p['hm']-hm) for p in packets)
        q=len(hm);d=len(dm);eye=torch.eye(q,device=hm.device,dtype=hm.dtype)
        lu=ridge_u*hh.trace()/(n*q)+floor
        U=torch.linalg.solve(hh+n*lu*eye,dh.T).T
        T=torch.eye(d,device=hm.device,dtype=hm.dtype)+U@R.to(hm).T;b=dm-U@hm
    counts=sum(p['counts'] for p in packets)
    if bool((counts<=0).any()):raise ValueError('support missing a current class')
    means=sum(p['counts'][:,None]*p['means'] for p in packets)/counts[:,None]
    scatter=sum(p['scatter']+sum(p['counts'][c]*torch.outer(p['means'][c]-means[c],p['means'][c]-means[c])
                               for c in range(len(counts))) for p in packets)
    if old is not None:
        oc=old['counts'].to(counts);om=old['means'].to(means);ov=old['scatter'].to(scatter)
        means=torch.cat((om@T.T+b,means));counts=torch.cat((oc,counts));scatter=(T@ov@T.T if mode=='full' else ov)+scatter
    scatter=(scatter+scatter.T)*.5
    cov=scatter/counts.sum();ls=ridge_s*cov.trace()/d+floor
    chol=torch.linalg.cholesky(cov+ls*torch.eye(d,device=means.device,dtype=means.dtype))
    W=torch.cholesky_solve(means.T,chol).T;beta=-.5*(W*means).sum(1)
    if not torch.isfinite(W).all() or not torch.isfinite(beta).all():raise ArithmeticError('invalid LDA')
    result=({'counts':counts,'means':means,'scatter':scatter},W,beta,{'ridge_u':float(lu.detach()),
        'ridge_s':float(ls.detach()),'U_norm':float(U.detach().norm()),'bias_norm':float(b.detach().norm()),
        'unique_count':float(counts.sum().detach()),'transport':mode})
    return result+((T,b),) if return_map else result


def class_wire(full_wire,d,q,c,mode):
    if mode!='identity':return full_wire
    moment_bytes=4*(q+d+q*q+d*q)
    return full_wire[8:8+8*c]+full_wire[8+8*c+moment_bytes:]


def dense_packet(z,labels,classes,R):
    packet,_,_=local_packet(z,z,labels,classes,R,rank=min(64,z.shape[1]))
    means=packet['means']
    # Match the encoded branch's FP32 class means; only scatter encoding differs.
    exact_means=torch.stack([z[labels==c].double().mean(0) if bool((labels==c).any()) else torch.zeros(z.shape[1],dtype=torch.float64) for c in range(classes)])
    residual=z.double()-exact_means[labels]
    packet['scatter']=residual.T@residual
    wire=packet['counts'].long().numpy().astype('<i8').tobytes()+means.float().numpy().astype('<f4').tobytes()+packet['scatter'].numpy().astype('<f8').tobytes()
    return packet,wire
