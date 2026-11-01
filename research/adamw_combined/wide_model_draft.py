"""Synthetic-stage temporary expanded adapter model; never mutates the source."""
import copy
import torch
from torch import nn


def wide_model(source, pools, tolerance=1e-5):
    """Copy a zero-B task-start model and install each d×m expanded pool."""
    if set(pools) != set(source.adapters):
        raise ValueError('pool and adapter names differ')
    for name, adapter in source.adapters.items():
        pool = pools[name]
        rank, dim = adapter.A.shape
        if (pool.ndim != 2 or pool.shape[0] != dim or not rank < pool.shape[1] <= dim or
                pool.device != adapter.A.device or pool.dtype != adapter.A.dtype or
                not torch.isfinite(pool).all() or torch.count_nonzero(adapter.Bq) or
                torch.count_nonzero(adapter.Bv) or adapter.collect):
            raise ValueError('invalid pool or nonzero active adapter')
        p = pool.double()
        eye = torch.eye(pool.shape[1], dtype=torch.float64, device=pool.device)
        if float(torch.linalg.matrix_norm(p.T @ p-eye)) > tolerance:
            raise ValueError('pool is not orthonormal')
        if float(torch.linalg.matrix_norm(p @ (p.T @ adapter.A.double().T)-adapter.A.double().T)) > tolerance:
            raise ValueError('pool excludes original basis')
    wide = copy.deepcopy(source)
    for name, adapter in wide.adapters.items():
        pool = pools[name]
        adapter.A = pool.T.contiguous().clone()
        for attr in ('Bq', 'Bv'):
            prior = getattr(adapter, attr)
            setattr(adapter, attr, nn.Parameter(prior.new_zeros((prior.shape[0], pool.shape[1])),
                                               requires_grad=prior.requires_grad))
    return wide
