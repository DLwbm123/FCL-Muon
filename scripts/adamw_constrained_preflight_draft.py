"""Synthetic-only real-backbone and wide-pool preflight."""
import json, os
from pathlib import Path
import torch, timm
from safetensors.torch import load_file
from fcl.model import MuLoRAFL
from research.adamw_combined.wide_model_draft import wide_model
from research.adamw_combined.multistep_probe_draft import simulate_candidates
from research.adamw_combined.constrained_multistep_draft import propose_from_trials

source=Path(os.environ['SOURCE_ROOT']); root=Path(os.environ['PROJECT_ROOT'])
w=json.loads((source/'weights.json').read_text())
torch.manual_seed(0);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
backbone=timm.create_model('vit_base_patch16_224.augreg_in21k',pretrained=False,num_classes=0)
backbone.load_state_dict({k:v for k,v in load_file(w['path']).items() if not k.startswith('head.')},strict=True)
model=MuLoRAFL(backbone,100,32,16).cuda();model.add_head(10);model.eval()
for a in model.adapters.values():a.start(torch.eye(32,768,device='cuda'))
pools={k:torch.eye(768,64,device='cuda') for k in model.adapters}
histories={k:torch.eye(768,dtype=torch.float64,device='cuda') for k in model.adapters}
x=torch.zeros(2,3,224,224,device='cuda');y=torch.tensor([0,1],device='cuda')
with torch.no_grad(): before=model(x).clone()
wide=wide_model(model,pools)
with torch.no_grad(): after=wide(x).clone()
if not torch.equal(before,after): raise ValueError('wide model output mismatch')
trials=simulate_candidates(wide,[{k:v.T for k,v in pools.items()}],[[(x,y)]],[1],1,0,10,.0002,5,123,5)[0]
summary={}
for mode in ('mean','individual'):
    candidates,records=propose_from_trials(model,pools,histories,trials,mode)
    if len(candidates)!=12 or trials[0]['train_visits']!=2:raise ValueError('trial mismatch')
    if any(not torch.isfinite(v).all() for v in candidates.values()):raise ValueError('nonfinite candidates')
    summary[mode]=sum(r['accepted'] for r in records.values())
if any(torch.count_nonzero(b) for b in model.b_parameters()):raise ValueError('source B modified')
(root/'reports/GPU_PREFLIGHT.json').write_text(json.dumps({'passed':True,'synthetic_data_images':2,'dataset_images':0,
    'model_before_after_forward_images':4,'trial_forward_images':6,'trial_backward_images':2,
    'layers':len(candidates),'accepted_by_mode':summary,
    'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated()},indent=2))
