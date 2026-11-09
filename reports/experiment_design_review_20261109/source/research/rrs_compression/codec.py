"""Actual byte codec: FP32 low-rank PSD factor plus diagonal residual.

The reconstructed matrix is PSD, with approximately preserved diagonal/trace.
It is not a Loewner upper bound and does not certify the original dense risk.
Client-local dense statistics never enter server decisions except via bytes.
"""
import struct
import numpy as np
import torch

HEADER=struct.Struct('<III4d')  # version, dimension, rank, four client diagnostics


def encode(statistic,rank=64,mode='lowrank_diag'):
    s=statistic.detach().cpu().double()
    if s.ndim!=2 or s.shape[0]!=s.shape[1] or not s.shape[0] or not torch.isfinite(s).all():
        raise ValueError('invalid client Gram')
    s=(s+s.T)*.5;d=len(s);trace=float(s.trace())
    if mode=='dense64':
        return HEADER.pack(0,d,d,0.,0.,1.,trace)+s.numpy().astype('<f8',copy=False).tobytes(order='C')
    if mode!='lowrank_diag' or not isinstance(rank,int) or rank<=0:raise ValueError('invalid codec')
    k=min(rank,d);values,vectors=torch.linalg.eigh(s)
    scale=max(float(values.abs().max()),1e-30)
    if float(values.min()) < -1e-8*scale:raise ArithmeticError('client Gram not PSD')
    values=values.clamp_min(0)
    factor=(vectors[:,-k:]*values[-k:].sqrt()).float().contiguous()
    residual=(s.diag()-factor.double().square().sum(1)).clamp_min(0).float()
    decoded=factor.double()@factor.double().T+torch.diag(residual.double())
    error=float((s-decoded).norm()/s.norm().clamp_min(1e-30))
    diagonal_error=float((s.diag()-decoded.diag()).abs().max()/s.diag().abs().max().clamp_min(1e-30))
    fraction=float(values[-k:].sum()/values.sum()) if float(values.sum()) else 1.
    if not torch.isfinite(factor).all() or not torch.isfinite(residual).all():raise ArithmeticError('FP32 overflow')
    return (HEADER.pack(1,d,k,error,diagonal_error,fraction,trace)
            +factor.numpy().astype('<f4',copy=False).tobytes(order='C')
            +residual.numpy().astype('<f4',copy=False).tobytes(order='C'))


def decode(packet):
    if len(packet)<HEADER.size:raise ValueError('truncated header')
    version,d,k,error,diagonal_error,fraction,trace=HEADER.unpack_from(packet)
    if not 0<k<=d<=8192 or not all(np.isfinite(v) for v in (error,diagonal_error,fraction,trace)):
        raise ValueError('invalid packet metadata')
    expected=HEADER.size+(d*d*8 if version==0 else (d*k+d)*4)
    if len(packet)!=expected:raise ValueError('packet size mismatch')
    if version==0:
        if k!=d:raise ValueError('invalid dense rank')
        matrix=torch.from_numpy(np.frombuffer(packet,offset=HEADER.size,dtype='<f8').copy()).reshape(d,d)
    elif version==1:
        x=torch.from_numpy(np.frombuffer(packet,offset=HEADER.size,dtype='<f4').copy())
        factor=x[:d*k].reshape(d,k).double();diagonal=x[d*k:].double()
        if bool((diagonal<0).any()):raise ValueError('negative diagonal residual')
        matrix=factor@factor.T+torch.diag(diagonal)
    else:raise ValueError('unknown packet version')
    if not torch.isfinite(matrix).all():raise ArithmeticError('nonfinite decoded Gram')
    return matrix,{'version':version,'dimension':d,'rank':k,'wire_bytes':len(packet),
        'relative_fro_error':error,'relative_diagonal_error':diagonal_error,
        'top_rank_trace_fraction':fraction,'original_trace':trace,
        'dense_reference_tensor_bytes':d*d*8}
