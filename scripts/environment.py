import importlib.metadata as md,json,os,platform,sys
from pathlib import Path
import torch
names=['torch','torchvision','timm','kornia','kornia_rs','numpy','huggingface-hub','safetensors','PyYAML','pytest','Pillow','pluggy','iniconfig']
info={'python':sys.version,'platform':platform.platform(),'versions':{n:md.version(n) for n in names},
      'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'free_total_bytes':torch.cuda.mem_get_info()}
root=Path(os.environ['PROJECT_ROOT']);(root/'reports/environment.json').write_text(json.dumps(info,indent=2));print(json.dumps(info))
