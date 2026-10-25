"""Synthetic actual-architecture gradient repeatability probe, no dataset access."""
import contextlib,json,os,time
from pathlib import Path
import torch,timm
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
from fcl.model import MuLoRAFL
from fcl.math import mulora_ssb

root=Path(os.environ['PROJECT_ROOT']);out=root/'runs/probe';out.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(4);torch.manual_seed(31);torch.set_float32_matmul_precision('highest')
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
model=MuLoRAFL(timm.create_model('vit_base_patch16_224.augreg_in21k',pretrained=False,num_classes=0),100,32,16).cuda().eval()
for adapter in model.adapters.values():
    a=torch.linalg.qr(torch.randn(768,32),mode='reduced').Q.T.cuda();adapter.start(a)
model.add_head(10);model.eval()
gen=torch.Generator().manual_seed(47);x=torch.randn(2,3,224,224,generator=gen).cuda();y=torch.tensor([1,7],device='cuda')
parameters=model.b_parameters()+[model.rows[-1]]

def compare(a,b):
    differences=[(u.double()-v.double()) for u,v in zip(a,b)]
    return {'exact':all(torch.equal(u,v) for u,v in zip(a,b)),
            'max_abs':max(float(d.abs().max()) for d in differences),
            'relative_norm_max':max(float(d.norm()/u.double().norm().clamp_min(1e-30)) for d,u in zip(differences,a))}

result={'scope':'Synthetic random-input/model actual ViT-B/QV-r32; no real-data training or test evaluation',
        'torch':torch.__version__,'fused_attention':model.backbone.blocks[0].attn.fused_attn,
        'workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),'modes':{}}
for mode in ['default_sdpa','deterministic_flag','math_deterministic']:
    torch.use_deterministic_algorithms(mode!='default_sdpa')
    ctx=sdpa_kernel(SDPBackend.MATH) if mode=='math_deterministic' else contextlib.nullcontext()
    gradients=[];operators=[];losses=[]
    try:
        with ctx:
            for _ in range(3):
                model.zero_grad(set_to_none=True)
                loss=F.cross_entropy(model(x),y);loss.backward()
                gradients.append([p.grad.detach().cpu().clone() for p in parameters]);losses.append(float(loss))
                operators.append([mulora_ssb(p.grad).cpu() for p in model.b_parameters()])
        result['modes'][mode]={'losses':losses,'gradient_repeats':[compare(gradients[0],v) for v in gradients[1:]],
                              'SSB_repeats':[compare(operators[0],v) for v in operators[1:]]}
    except RuntimeError as e:
        result['modes'][mode]={'error':str(e)}
    print(json.dumps({mode:result['modes'][mode]}),flush=True)
(out/'RESULTS.json').write_text(json.dumps(result,indent=2))
(out/'COMPLETE.json').write_text(json.dumps({'completed':time.time(),'diagnostic_only':True}))
