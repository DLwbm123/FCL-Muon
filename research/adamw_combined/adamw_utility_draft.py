"""Unregistered task-boundary AdamW one-step utility; not a local trajectory."""
import torch


def quantities(z, client_gradients, weights, history, epsilon=1e-8):
    """Return first-step descent, history proxies and actual dimensionless steps.

    Assumes a common zero B, reset AdamW moments, zero weight decay, identical
    scalar learning rates and a fixed basis. Overall learning rate is factored out.
    """
    if z.ndim != 2 or client_gradients.ndim != 4 or history.ndim != 2:
        raise ValueError('incorrect tensor dimensions')
    clients, _, _, width = client_gradients.shape
    if z.shape[0] != width or history.shape != (width, width) or epsilon <= 0:
        raise ValueError('incompatible dimensions or epsilon')
    w = torch.as_tensor(weights, device=client_gradients.device, dtype=client_gradients.dtype)
    if (w.shape != (clients,) or not torch.isfinite(w).all() or (w < 0).any()
            or float(w.sum()) <= 0 or not torch.isfinite(z).all()
            or not torch.isfinite(client_gradients).all() or not torch.isfinite(history).all()):
        raise ValueError('invalid weights or nonfinite input')
    w = w / w.sum()
    projected = client_gradients @ z
    local_descent = projected / (projected.abs() + epsilon)
    gbar = torch.einsum('k,kvdr->vdr', w, projected)
    dbar = torch.einsum('k,kvdr->vdr', w, local_descent)
    h = z.double().T @ history.double() @ z.double()
    return {'utility': (gbar * dbar).sum(), 'gamma': h.trace(),
            'risk': torch.einsum('vdr,rs,vds->', dbar.double(), h, dbar.double()),
            'direction': dbar, 'update_norm': dbar.norm()}
