"""Actual backbone, synthetic inputs only; no source training or test images."""
import json
import os
from pathlib import Path
import torch
import timm
from safetensors.torch import load_file
from fcl.model import MuLoRAFL
from research.adamw_combined.aggregated_response_draft import paired_aggregates
from research.adamw_combined.aggregated_response_pilot_draft import AggregatedResponsePilot
from unittest.mock import patch

root = Path(os.environ['PROJECT_ROOT']); source = Path(os.environ['SOURCE_ROOT'])
weights = json.loads((source/'weights.json').read_text())
torch.manual_seed(0); torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True); torch.set_num_threads(4)
backbone = timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=False, num_classes=0)
backbone.load_state_dict({k:v for k,v in load_file(weights['path']).items() if not k.startswith('head.')}, strict=True)
model = MuLoRAFL(backbone, 100, 32, 16).cuda(); model.add_head(10); model.eval()
for adapter in model.adapters.values(): adapter.start(torch.eye(32, 768, device='cuda'))
bases = {name:a.A.detach().clone() for name,a in model.adapters.items()}
x = torch.zeros(2, 3, 224, 224, device='cuda'); y = torch.tensor([0, 1], device='cuda')
clients = [[(x, y)], [(x, y)]]
before = {k:v.detach().clone() for k,v in model.state_dict().items()}
def measure(m):
    with torch.no_grad(): return {'ce':float(torch.nn.functional.cross_entropy(m(x), y))}
results = paired_aggregates(model, [bases, bases], clients, [1, 1], [2, 2], 1, 0, 10,
                            .0002, 5, 123, measure, schedule_epochs=5)
if (results[0]['measurement'] != results[1]['measurement'] or
        any(not torch.equal(before[k], v) for k,v in model.state_dict().items()) or
        any(r['gram_visits'] != 4 or r['train_visits'] != 4 for r in results)):
    raise ValueError('paired trial or state restore mismatch')
model.add_head(10)
pilot = object.__new__(AggregatedResponsePilot); pilot.model = model; pilot.device = torch.device('cuda')
pilot.cfg = {'clients':2, 'batch':2, 'epochs':5, 'cpt':10, 'lr':.0002, 'rounds':5,
             'multistep':{'split_seed':3, 'client_cap':2, 'pool_width':64, 'trial_epochs':1,
                          'trial_seed':12, 'response_mode':'selected'}}
history = torch.eye(768,dtype=torch.float64,device='cuda')
pilot.s = {'task':1, 'pending':{k:history for k in model.adapters},
           'history':type('H',(),{'raw':{k:history for k in model.adapters}})(), 'communication':{}}
pilot.raw = [None]; pilot.ids = lambda _, __, k=None: list(range(8*k,8*k+8)) if k is not None else list(range(16))
pilot.manifest = {'partitions':{'train':[[],[list(range(8)),list(range(8,16))]]}}
def loader(_, ids, batch):
    for i in range(0,len(ids),batch):
        n = len(ids[i:i+batch]); yield torch.zeros(n,3,224,224), torch.full((n,),10,dtype=torch.long)
pilot.loader = loader; pilot.test_tf = lambda x:x; pilot.check_labels = lambda _:None
pilot.budget = lambda:None; pilot.event = lambda *args, **kwargs:None
pool = torch.eye(768,64,device='cuda')
def fixed_pool(*_): return pool, None, None
before = {k:v.detach().clone() for k,v in model.state_dict().items()}
with patch('research.adamw_combined.aggregated_response_pilot_draft.candidate_pool', fixed_pool):
    pilot._propose_multistep()
if (len(pilot.s.get('selection_records',[])) != 1 or
        any(torch.count_nonzero(b) for b in model.b_parameters()) or
        any(not torch.equal(before[k], v) for k,v in model.state_dict().items()
            if not k.endswith('.A'))):
    raise ValueError('complete task-boundary synthetic lifecycle failed')
(root/'reports/GPU_RESPONSE_PREFLIGHT.json').write_text(json.dumps({
    'passed':True, 'synthetic_inputs':2, 'dataset_images':0, 'candidates':2,
    'trial_backward_images':8, 'gram_forward_images':8,
    'full_boundary_synthetic_clients':2, 'full_boundary_synthetic_visits':pilot.s['selection_records'][0]['trial_train_visits'],
    'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated()},indent=2))
