"""Fresh-seed paired Head-RegMean/FedAvg learner; paths are environment-only."""
import json,os,random,time
from pathlib import Path
import numpy as np
import torch,timm
from safetensors.torch import load_file
from torchvision.datasets import CIFAR100
from fcl.data import loader,transforms
from fcl.model import MuLoRAFL
from research.head_regmean.confirmation import ConfirmationPilot


def main():
    root=Path(os.environ['PROJECT_ROOT']);cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
    out=root/cfg['output'];out.mkdir(parents=True,exist_ok=False)
    clock=json.loads((root/'reports/job.json').read_text());clock={k:clock[k] for k in ['started','deadline']}
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed'])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if cfg.get('math_sdpa',False):
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    mf=root/cfg['manifest'];m=json.loads(mf.read_text())
    assert m['identity']=={k:cfg[k] for k in ['dataset','seed','alpha','clients']}
    data=[CIFAR100(root/'data',train=t,download=False) for t in [True,False]]
    assert [list(v.targets) for v in data]==[m['labels']['train'],m['labels']['test']]
    wm=json.loads((root/'weights.json').read_text())
    backbone=timm.create_model(cfg['backbone'],pretrained=False,num_classes=0)
    backbone.load_state_dict({k:v for k,v in load_file(wm['path']).items() if not k.startswith('head.')},strict=True)
    model=MuLoRAFL(backbone,100,cfg['rank'],cfg['cosine_scale']).cuda()
    train_tf,test_tf=transforms();train_tf=train_tf.cuda();test_tf=test_tf.cuda()
    identity={'manifest':cfg['manifest'],'weight_identity':wm['sha256'],'source_revision':cfg['source_revision'],
              'purpose':'paired head aggregation with fixed raw/centered reporting'}
    p=ConfirmationPilot(model,data,m,loader,train_tf,test_tf,cfg,out,identity,clock);p.run()
    records=p.s['training_records'];expected=cfg['tasks']*cfg['rounds']*5000*cfg['epochs']
    assert sum(v['visited'] for v in records)==expected
    assert len({(v['task'],v['round'],v['client']) for v in records})==len(records)
    assert all(v['visited']==v['N']*cfg['epochs'] for v in records)
    assert p.s['merges']==list(range(cfg['tasks'])) and p.s['history'].committed==set(range(cfg['tasks']))
    assert all(not torch.count_nonzero(b) for b in model.b_parameters())
    if cfg['aggregation']=='regmean':
        assert len(p.s['certificates'])==cfg['tasks']*cfg['rounds']
        assert all(v['deployed_fp32']['passed'] for v in p.s['certificates'])
    # Actual saved final checkpoint is reloaded and its full confusion reproduced.
    p.restore();p.s['task']=cfg['tasks']-1
    final=p.evaluate();assert final['confusion']==p.s['stages'][-1]['confusion']
    assert final['centered']==p.s['stages'][-1]['centered']
    centered_matrix=[[v['all_seen_accuracy'] for v in s['centered']['per_task']] for s in p.s['stages']]
    last=centered_matrix[-1];tasks=cfg['tasks']
    qualification={'qualified':True,'training_visits':sum(v['visited'] for v in records),
        'optimizer_steps':sum(v['steps'] for v in records),'client_rounds':len(records),
        'stats_visits':sum(v['samples'] for v in p.s['stats_records']),
        'logical_communication_bytes':p.s['communication'],'checkpoint_reload_confusion_equal':True,
        'centered_matrix':centered_matrix,'centered_final':sum(last)/tasks,
        'centered_forgetting':sum(max(centered_matrix[j][i] for j in range(i,tasks))-last[i] for i in range(tasks-1))/(tasks-1),
        'clock':clock}
    (out/'QUALIFICATION.json').write_text(json.dumps(qualification,indent=2))
    (out/'COMPLETE.json').write_text(json.dumps({'completed':time.time(),'qualified':True}))
    print(json.dumps(qualification),flush=True)


if __name__=='__main__':main()
