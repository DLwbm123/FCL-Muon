import json,os
from pathlib import Path
import torch
from safetensors.torch import load_file
from lightning_fabric import Fabric
from fcl.lorm_upstream import load_lorm
root=Path(os.environ['PROJECT_ROOT'])
torch.set_num_threads(4);torch.manual_seed(0)
LoRM,ViT=load_lorm()
fabric=Fabric(accelerator='cuda',devices=1,precision='16-mixed');fabric.launch()
net=ViT(pretrained=False,num_classes=100)
w=json.loads((root/'weights.json').read_text());sd={k:v for k,v in load_file(w['path']).items() if not k.startswith('head.')}
missing,extra=net.model.load_state_dict(sd,strict=False);assert set(missing)=={'head.weight','head.bias'} and not extra
model=LoRM(fabric,net,'cuda:0',lr=.0005,r=16,wd_reg=0,lora_head=False,regmean_all=True,alpha_regmean_head=.5,alpha_regmean_backbone=0,gram_dtype='32',reg_dtype_64=True,train_matrix='alt',lr_B=-1,lr_A=-1)
print('LORM_SOURCE_READY',len(model.lora_keys),len(model.gram_modules),model.lora_keys[:4],flush=True)
model.begin_task(10);model.begin_round_server()
print('LORM_SERVER_INFO',list(model.get_server_info()),flush=True)
