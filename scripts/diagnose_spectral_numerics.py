"""Seeded synthetic arithmetic audit only; never loads data or model checkpoints."""
import json
import os
from pathlib import Path
import torch
from fcl.spectral_qp import SolverConfig, certificate, solve_spectral_qp, spectral_project

torch.set_num_threads(4)
device = os.environ.get('MATH_DEVICE', 'cpu')
cfg = SolverConfig()
rows = []

def reference(g, h, d, eta):
    g, h, d = g.double(), h.double(), d.double()
    k = g + d @ h
    scale = max(float(eta * torch.linalg.svdvals(g).sum()), 1e-30)
    gap = float((k * d).sum() + eta * torch.linalg.svdvals(k).sum())
    norm = float(torch.linalg.matrix_norm(d, ord=2))
    return {'gap': gap, 'relative_gap': gap / scale, 'norm': norm,
            'relative_radius_excess': (norm - eta) / eta,
            'gap_pass': gap <= cfg.gap_atol + cfg.gap_rtol * scale,
            'feasible_pass': norm <= eta * (1 + 64 * torch.finfo(torch.float32).eps)}

for seed in range(3):
    torch.manual_seed(seed)
    g = torch.randn(768, 32, device=device) * 1e-5
    h = torch.diag(torch.logspace(-3, 0, 32, device=device))
    for eta in [2e-4, 5e-5, 1e-8]:
        d, info = solve_spectral_qp(g, h, eta, config=cfg)
        for bscale in [0., 1e-3, 1.]:
            b = torch.randn_like(g) * bscale
            after = b + d
            actual32 = after - b
            actual64 = after.double() - b.double()
            old = certificate(g, h, actual32, eta, cfg)
            rows.append({'seed': seed, 'eta': eta, 'B_scale': bscale,
                         'solver_converged': info['converged'], 'iterations': info['iterations'],
                         'actual_certificate_converged': old['converged'],
                         'proposal_reference': reference(g, h, d, eta),
                         'actual_reference': reference(g, h, actual64, eta),
                         'proposal_to_actual_fro': float((actual64 - d.double()).norm()),
                         'subtraction_fp32_fro_error': float((actual32.double() - actual64).norm())})
    # Projection must be checked independently of the same FP32 SVD that produced it.
    projected = spectral_project(g, 5e-5)
    rows.append({'seed': seed, 'projection_reference': reference(g, torch.zeros_like(h), projected, 5e-5)})
result = {'device': device, 'torch': torch.__version__, 'data': 'synthetic seeded matrices only', 'rows': rows}
out = Path(os.environ['MATH_REPORT'])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'report': str(out), 'cases': len(rows),
                  'proposal_feasibility_failures': sum(not r['proposal_reference']['feasible_pass'] for r in rows if 'proposal_reference' in r),
                  'projection_feasibility_failures': sum(not r['projection_reference']['feasible_pass'] for r in rows if 'projection_reference' in r),
                  'actual_feasibility_failures': sum(not r['actual_reference']['feasible_pass'] for r in rows if 'actual_reference' in r)}))
