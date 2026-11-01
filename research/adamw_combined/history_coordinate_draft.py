"""Synthetic-stage history-aligned coordinates for a fixed RRS subspace."""
import torch


def history_coordinates(a, history, relative_gap=1e-7, tolerance=1e-5):
    """Return a rotated FP32 basis or an exact baseline when numerically ambiguous.

    This preserves the selected subspace; it does not by itself select a new one.
    """
    original = a.detach().clone()
    if (a.ndim != 2 or history.shape != (a.shape[1], a.shape[1]) or
            not torch.isfinite(a).all() or not torch.isfinite(history).all()):
        raise ValueError('invalid basis or history')
    ad = a.double()
    hd = history.double()
    eye = torch.eye(a.shape[0], device=a.device, dtype=torch.float64)
    if float(torch.linalg.matrix_norm(ad @ ad.T-eye)) > tolerance:
        return original, 'basis_not_orthonormal'
    hnorm = float(torch.linalg.matrix_norm(hd))
    if float(torch.linalg.matrix_norm(hd-hd.T)) > tolerance*max(hnorm, 1e-12):
        return original, 'history_not_symmetric'
    projected = ad @ ((hd+hd.T)*.5) @ ad.T
    values, vectors = torch.linalg.eigh(projected)
    scale = max(float(values.abs().max()), 1e-12)
    if float(values.min()) < -tolerance*scale:
        return original, 'history_not_psd'
    if values.numel() > 1 and float(values.diff().abs().min()) <= relative_gap*scale:
        return original, 'eigenvalue_tie'
    # Resolve eigenvector signs in the original RRS coordinates.
    pivots = vectors.abs().argmax(dim=0)
    signs = torch.sign(vectors[pivots, torch.arange(vectors.shape[1], device=a.device)])
    rotated = ((vectors*signs).T @ ad).to(a.dtype)
    if (not torch.isfinite(rotated).all() or
            float(torch.linalg.matrix_norm(rotated.double() @ rotated.double().T-eye)) > tolerance or
            float(torch.linalg.matrix_norm(rotated.double().T @ rotated.double()-ad.T @ ad)) > tolerance):
        return original, 'deployment_check_failed'
    return rotated, 'accepted'
