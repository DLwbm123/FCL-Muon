"""Verify the separately saved H1 repair checkpoint on the fixed T1 test set."""
import json
import os
from pathlib import Path
import torch
import timm
from torchvision.datasets import CIFAR100
from fcl.model import MuLoRAFL
from fcl.data import loader,transforms

root=Path(os.environ['PROJECT_ROOT'])
out=root/'runs/head_diagnostics'
report=json.loads((out/'results.json').read_text())
s=torch.load(out/'t1_repaired_H1.pt',map_location='cpu',weights_only=False)
assert s['task']==1 and not s['initialized'] and s['repair']['test_selected'] is False
torch.set_num_threads(4)
torch.set_float32_matmul_precision('highest')
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
model=MuLoRAFL(timm.create_model('vit_base_patch16_224.augreg_in21k',pretrained=False,num_classes=0),100,32,16)
model.add_head(10);model.load_state_dict(s['model'],strict=True)
model=model.eval().requires_grad_(False).cuda()
assert all(torch.count_nonzero(b)==0 for b in model.b_parameters())
m=json.loads((root/'manifests/seq-cifar100_n10_a0.05_s0.json').read_text())
ids=[i for client in m['partitions']['test'][0] for i in client]
raw=CIFAR100(root/'data',train=False,download=False)
_,tf=transforms();tf=tf.cuda();confusion=torch.zeros(10,10,dtype=torch.int64)
with torch.inference_mode():
    for x,y in loader(raw,ids,16):
        pred=model(tf(x.cuda())).argmax(1).cpu()
        confusion+=torch.bincount(y*10+pred,minlength=100).reshape(10,10)
expected=report['encoders']['adapted']['readouts']['H1_checkpoint_fp32']
assert confusion.tolist()==expected['confusion']
assert len(list((root/'runs/t1_original').glob('client_t0_*.pt')))==50
assert len(list((root/'runs/t1_original').glob('boundary_*.pt')))==7
result={'qualified':True,'checkpoint':'runs/head_diagnostics/t1_repaired_H1.pt','accuracy':int(confusion.diag().sum())/int(confusion.sum()),'correct':int(confusion.diag().sum()),'n':int(confusion.sum()),'full_confusion_matches_head_diagnostic':True,'local_snapshots':50,'global_boundary_snapshots':7,'merged_B_zero':True,'test_selected':False}
(out/'REPAIR_QUALIFICATION.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
