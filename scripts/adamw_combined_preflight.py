"""Real model, synthetic input only; independent process before formal training."""
import json, os
from pathlib import Path
import torch, timm
from safetensors.torch import load_file
from torch.nn import functional as F
from fcl.model import MuLoRAFL
from research.adamw_combined.pilot import AdamAdapter
from research.fed_spectra.actual_update import global_shape_update
from research.fed_spectra.matrix_tangent import tangent_candidate

source=Path(os.environ['SOURCE_ROOT']); root=Path(os.environ['PROJECT_ROOT'])
weights=json.loads((source/'weights.json').read_text())
torch.manual_seed(0);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
backbone=timm.create_model('vit_base_patch16_224.augreg_in21k',pretrained=False,num_classes=0)
backbone.load_state_dict({k:v for k,v in load_file(weights['path']).items() if not k.startswith('head.')},strict=True)
model=MuLoRAFL(backbone,100,32,16).cuda();model.add_head(10)
for adapter in model.adapters.values():adapter.start(torch.eye(32,768,device='cuda'))
params=model.b_parameters();opt=AdamAdapter(params)
x=torch.zeros(2,3,224,224,device='cuda');y=torch.tensor([0,1],device='cuda')
loss=F.cross_entropy(model(x),y);loss.backward()
if not torch.isfinite(loss) or not all(p.grad is not None and torch.isfinite(p.grad).all() for p in params):raise ValueError('gradient failure')
before=[p.detach().clone() for p in params];opt.step(.0002)
if not any(not torch.equal(a,b) for a,b in zip(before,params)):raise ValueError('AdamW step unchanged')
d=params[0].detach().cpu().double();g=params[0].grad.detach().cpu()
d2=torch.roll(d,1,0); base=(d+d2)/2
ref,_=global_shape_update([d,d2],[1,1],base,eta=.1)
origin=torch.full_like(d,0.01).float(); baseline=(origin.double()+base).float()
z,info=tangent_candidate(origin,baseline,origin.double()+ref,g)
if not torch.isfinite(z).all() or info['actual_FP32_norm_relative_error']>5e-4:raise ValueError('candidate preflight failure')
out={'passed':True,'synthetic_student_forward_images':2,'synthetic_student_backward_images':2,'teacher_forward_images':0,
     'dataset_images':0,'adam_parameters':len(params),'synthetic_loss':float(loss),'geometry_identity_reason':info['identity_reason']}
(root/'reports/GPU_PREFLIGHT.json').write_text(json.dumps(out,indent=2))
