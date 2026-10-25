"""Evaluate a merged completed checkpoint on all seen labels; no task ID input."""
import argparse,json,os
from pathlib import Path
import torch,timm
from fcl.data import load_raw,loader,transforms
from fcl.model import MODELS
p=argparse.ArgumentParser();p.add_argument('--checkpoint',default=os.environ.get('EVAL_CHECKPOINT'));a=p.parse_args()
root=Path(os.environ.get('PROJECT_ROOT','.'));s=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
if s['initialized']:raise ValueError('Use a merged task-boundary checkpoint for standalone evaluation')
cfg=s['config'];raw,cpt,classes=load_raw(cfg['dataset'],root/'data')
net=timm.create_model('vit_base_patch16_224.augreg_in21k',pretrained=False,num_classes=0)
model=MODELS[cfg['method']](net,classes,cfg['rank'],cfg.get('cosine_scale',16))
for _ in range(s['head_count']):model.add_head(cpt)
model.load_state_dict(s['model']);model.cuda().eval();_,transform=transforms();transform.cuda()
correct=total=0;tasks=[]
with torch.no_grad():
 for t in range(s['task']):
  ids=[i for i,y in enumerate(raw[1].targets) if t*cpt<=y<(t+1)*cpt]
  c=n=0
  for x,y in loader(raw[1],ids,cfg['batch']):
   pred=model(transform(x.cuda())).argmax(-1).cpu();c+=int((pred==y).sum());n+=len(y)
  tasks.append(c/n);correct+=c;total+=n
print(json.dumps({'micro':correct/total,'task_macro':sum(tasks)/len(tasks),'tasks':tasks,'all_seen_classes':s['task']*cpt}))
