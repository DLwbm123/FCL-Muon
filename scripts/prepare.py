import hashlib
import json
import os
from pathlib import Path
import torch
import timm
from torchvision.datasets import CIFAR100
from huggingface_hub import HfApi,hf_hub_download

root=Path(os.environ['PROJECT_ROOT'])
print('preparing weights',flush=True)
repo='timm/vit_base_patch16_224.augreg_in21k'
revision=HfApi().model_info(repo).sha
weight=hf_hub_download(repo,'model.safetensors',revision=revision,cache_dir=root/'cache')
h=hashlib.sha256()
with open(weight,'rb') as f:
    for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
(root/'weights.json').write_text(json.dumps({'repo':repo,'revision':revision,'path':weight,'sha256':h.hexdigest(),'timm':timm.__version__},indent=2))
for train in (True,False):CIFAR100(root/'data',train=train,download=True)
print('PREPARATION_COMPLETE',flush=True)
