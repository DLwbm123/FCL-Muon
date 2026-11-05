"""Synthetic real-backbone two-round response preflight; no dataset images."""
import json, os, time
from pathlib import Path
import torch
import timm
from safetensors.torch import load_file
from fcl.model import MuLoRAFL
from research.route_compare_20261102.response import federated_response


def main():
    start = time.monotonic()
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.75, 0)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)
    weights = json.loads((Path(os.environ['SOURCE_ROOT'])/'weights.json').read_text())
    backbone = timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=False, num_classes=0)
    backbone.load_state_dict({k:v for k,v in load_file(weights['path']).items() if not k.startswith('head.')}, strict=True)
    model = MuLoRAFL(backbone, 100, 32, 16).cuda().eval()
    model.add_head(10); model.add_head(10)
    for a in model.adapters.values(): a.start(torch.eye(32, 768, device='cuda'))
    bases = {k:a.A.detach().clone().requires_grad_(True) for k,a in model.adapters.items()}
    n = int(os.environ.get('PREFLIGHT_CLIENTS', '10'))
    rounds = int(os.environ.get('PREFLIGHT_ROUNDS', '2'))
    optimizer = os.environ.get('PREFLIGHT_OPTIMIZER', 'adamw')
    if not (1 <= n <= 10 and rounds in (1,2) and optimizer in ('adamw','muon')):
        raise ValueError('invalid preflight configuration')
    # Shared synthetic examples minimize preloaded input storage; computation
    # still covers the full client/round/step/batch shape.
    x = torch.randn(4,3,224,224,device='cuda')
    y = torch.tensor([10,11,12,13],device='cuda')
    support = [[[(x,y)]*4 for _ in range(n)] for _ in range(rounds)]
    gram = [[(x,y)] for _ in range(n)]
    query = [[(x,y)] for _ in range(n)]
    before = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    # Higher-order multi-client autograd holds a full response graph. Offload
    # saved intermediates while retaining the same forward and backward math.
    with torch.autograd.graph.save_on_cpu(pin_memory=False):
        response = federated_response(model,bases,support,gram,query,[32]*n,[2]*n,
                                      optimizer,.0002,5,5,1,10,use_checkpoint=True)
    gradients = torch.autograd.grad(response['loss'],list(bases.values()))
    if not all(torch.isfinite(g).all() for g in gradients): raise ArithmeticError('nonfinite gradient')
    if any(not torch.equal(v,model.state_dict()[k].cpu()) for k,v in before.items()):
        raise AssertionError('source model changed')
    print(json.dumps({'passed':True,'clients':n,'short_rounds':rounds,'optimizer':optimizer,
                      'batch_size':4,'updates_per_client_round':4,'saved_tensors':'cpu',
                      'counts':response['counts'],
                      'real_dataset_images':0,'seconds':time.monotonic()-start,
                      'peak_allocated_bytes':torch.cuda.max_memory_allocated()}),flush=True)

if __name__ == '__main__': main()
