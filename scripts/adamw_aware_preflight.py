"""Synthetic-image GPU preflight for AdamW-aware utility mode."""
import json, os, runpy
from pathlib import Path
import torch
from research.aggregation_utility.selection import quantities

runpy.run_module('scripts.adamw_combined_preflight',run_name='__main__')
torch.manual_seed(123)
c=torch.randn(2,2,768,64,device='cuda',dtype=torch.float32)
z=torch.eye(64,32,device='cuda',dtype=torch.float32);weights=torch.tensor([.4,.6],device='cuda')
h=torch.eye(64,device='cuda',dtype=torch.float64)
q=quantities(z,c,weights,h,'adamw')
local=[]
for g in c@z:
    p=torch.nn.Parameter(torch.zeros_like(g))
    opt=torch.optim.AdamW([p],lr=.0002,betas=(.9,.999),eps=1e-8,weight_decay=0.)
    p.grad=g.detach().clone();opt.step();local.append(-p.detach()/.0002)
expected=sum(w*d for w,d in zip(weights,local))
error=float((expected-q['direction']).abs().max())
if not torch.isfinite(q['utility']) or error>2e-6:
    raise ValueError('GPU AdamW utility mismatch')
path=Path(os.environ['PROJECT_ROOT'])/'reports/GPU_PREFLIGHT.json'
data=json.loads(path.read_text());data.update(adamw_utility_clients=2,adamw_utility_max_abs_error=error,
                                              extra_dataset_images=0,synthetic_gradient_only=True)
path.write_text(json.dumps(data,indent=2))
