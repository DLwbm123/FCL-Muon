"""Build the original AdamW selected-route method without any proximal class."""
import hashlib
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
from research.stat_ablation.pilot import AblationPilot as HeadPilot


def setup():
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True);torch.use_deterministic_algorithms(True)


def validate():
    import timm,kornia
    root=Path(os.environ['STUDY_ROOT']);f=json.loads((root/'src/reports/FROZEN_PROTOCOL.json').read_text())
    for name,sha in f['source_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=sha:raise ValueError('frozen source changed: '+name)
    for path,stat in f['asset_stat'].items():
        s=Path(path).stat()
        if [s.st_size,s.st_mtime_ns,s.st_ino]!=stat:raise ValueError('shared asset changed')
    if f['environment']!={'torch':str(torch.__version__),'timm':str(timm.__version__),'kornia':str(kornia.__version__),'cuda':torch.version.cuda}:
        raise ValueError('environment changed')
    manifest=root/'src/manifests/seed0.json'
    if hashlib.sha256(manifest.read_bytes()).hexdigest()!=f['manifest_sha256']:raise ValueError('manifest changed')
    return root,f


def build(clock, synthetic=None):
    import timm
    from safetensors.torch import load_file
    from fcl.model import MuLoRAFL
    from research.stat_head.model import StatisticsViT
    from fcl.data import transforms,loader
    root,f=validate();import copy
    cfg=copy.deepcopy(f['config']);cfg['head_protocol']='unified';cfg['arm']=os.environ.get('ARM','identity_selected')
    cfg['branch_modes']=['identity'] if cfg['arm']=='identity_selected' else ['identity','full','mean_only']
    cfg['route']['mode']='selected' if cfg['arm']=='identity_selected' else 'rrs'
    cfg['primary_mode']='identity' if cfg['arm']=='identity_selected' else 'full'
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed'])
    backbone=timm.create_model(cfg['backbone'],pretrained=False,num_classes=0)
    backbone.load_state_dict({k:v for k,v in load_file(f['weights']['path']).items() if not k.startswith('head.')},strict=True)
    model=(StatisticsViT if cfg['head_protocol']=='unified' else MuLoRAFL)(backbone,100,cfg['rank'],cfg['cosine_scale']).cuda().eval()
    train_tf,test_tf=transforms()
    accesses={'train_loaded':0,'test_loaded':0}
    if synthetic is None:
        from torchvision.datasets import CIFAR100
        raw=[CIFAR100(root/'src/data',train=t,download=False) for t in (True,False)]
        manifest=json.loads((root/'src/manifests/seed0.json').read_text())
        if any(list(r.targets)!=manifest['labels'][s] for r,s in zip(raw,('train','test'))):raise ValueError('labels mismatch')
        def used_loader(dataset,ids,batch,shuffle=False):
            class Counted:
                def __len__(self):return (len(ids)+batch-1)//batch
                def __iter__(self):
                    for x,y in loader(dataset,ids,batch,shuffle):
                        accesses['train_loaded' if dataset is raw[0] else 'test_loaded']+=len(y)
                        yield x,y
            return Counted()
    else:raw,manifest,used_loader=synthetic
    pilot_class=HeadPilot
    p=pilot_class(model,raw,manifest,used_loader,train_tf.cuda(),test_tf.cuda(),cfg,
                   root/('src/runs/'+cfg['arm']),{'source_sha256':f['code_identity'],'manifest_sha256':f['manifest_sha256'],
                   'weights_sha256':f['weights']['sha256'],'seed':0,
                   'method':'adamw_selected_rrs_compressed' if 'rrs_compression' in cfg else 'original_adamw_selected_no_prox'},clock)
    p.accesses=accesses
    return p
