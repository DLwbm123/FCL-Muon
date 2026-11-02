"""Synthetic-only device viability check; not a registered scientific experiment."""
import json
import os
import time
from pathlib import Path
import torch
import timm
from safetensors.torch import load_file
from fcl.model import MuLoRAFL
from research.adamw_combined.differentiable_response_draft import differentiable_response
from research.adamw_combined.bilevel_step_draft import propose_step


def main():
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.6, 0)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    source = Path(os.environ['SOURCE_ROOT'])
    weights = json.loads((source/'weights.json').read_text())
    backbone = timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=False, num_classes=0)
    backbone.load_state_dict({k: v for k, v in load_file(weights['path']).items() if not k.startswith('head.')}, strict=True)
    model = MuLoRAFL(backbone, 100, 32, 16).cuda()
    model.add_head(10); model.add_head(10); model.eval()
    for adapter in model.adapters.values():
        adapter.start(torch.eye(32, 768, device='cuda'))
    bases = {n: a.A.detach().clone() for n, a in model.adapters.items()}
    nclients = int(os.environ.get('PREFLIGHT_CLIENTS', '2'))
    steps = int(os.environ.get('PREFLIGHT_STEPS', '1'))
    if not (2 <= nclients <= 10 and 1 <= steps <= 2):
        raise ValueError('preflight size exceeds synthetic limit')
    client = [[(torch.randn(1, 3, 224, 224, device='cuda'), torch.tensor([10+k%10], device='cuda'))
               for _ in range(steps)] for k in range(nclients)]
    fit = [[(torch.randn(1, 3, 224, 224, device='cuda'), torch.tensor([10+k%10], device='cuda'))]
           for k in range(nclients)]
    if os.environ.get('PREFLIGHT_BATCHED') == '1':
        client = [[(torch.cat([x for x, _ in batches]), torch.cat([y for _, y in batches]))]
                  for batches in client]
        fit = [[(torch.randn(2, 3, 224, 224, device='cuda'),
                 torch.full((2,), 10+k%10, device='cuda'))] for k in range(nclients)]
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    calls = []
    def evaluate(b):
        result = differentiable_response(model, b, client, [2]*nclients, [steps]*nclients,
                                         fit, 1, 1, 10, .0002, 5, 5)
        calls.append({k: result[k] for k in ('trial_visits', 'gram_visits', 'evaluation_visits')})
        return result
    histories = {n: torch.zeros(768, 768, dtype=torch.float64, device='cuda') for n in bases}
    candidate, decision = propose_step(bases, histories, evaluate, .01)
    if decision['reason'] == 'zero_or_nonfinite_direction':
        raise ArithmeticError('zero or nonfinite surrogate direction')
    if any(not torch.isfinite(value).all() for value in candidate.values()):
        raise ArithmeticError('nonfinite candidate basis')
    if any(not torch.equal(before[k], v) for k, v in model.state_dict().items()):
        raise AssertionError('source model changed')
    confirmation = None
    if os.environ.get('PREFLIGHT_CONFIRMATION') == '1':
        # Exercise the full-holdout, no-outer-gradient evaluation path using
        # synthetic images only. Counts intentionally approximate a 500-image
        # confirmation, without using the private manifest or dataset.
        holdout = [[(torch.randn(10, 3, 224, 224, device='cuda'),
                     torch.full((10,), 10+k%10, device='cuda')) for _ in range(5)]
                   for k in range(nclients)]
        result = differentiable_response(model, bases, client, [2]*nclients, [steps]*nclients,
                    holdout, 1, 1, 10, .0002, 5, 5, track_response=False)
        if result['loss'].requires_grad or not torch.isfinite(result['loss']):
            raise ArithmeticError('invalid no-outer-gradient confirmation')
        confirmation = {k: result[k] for k in ('trial_visits', 'gram_visits', 'evaluation_visits')}
    print(json.dumps({'passed': True, 'synthetic_clients': nclients, 'trial_steps': steps,
                      'synthetic_distinct_trial_images': nclients*steps,
                      'synthetic_distinct_fit_images': sum(len(y) for groups in fit for _, y in groups),
                      'response_calls': calls, 'decision': decision, 'real_dataset_images': 0,
                      'confirmation': confirmation,
                      'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                      'elapsed_seconds': time.monotonic()-started}), flush=True)


if __name__ == '__main__':
    main()
