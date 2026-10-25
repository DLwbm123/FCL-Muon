"""Verify the actual final LoRM checkpoint using the previously audited restore."""
import json
import numpy as np
import torch
from torchvision.datasets import CIFAR100
from fcl.data import loader,transforms
from scripts.diagnose_checkpoint import restore
from research.basic_repair.fcl_basic_repair_tools import center_task_logits


def verify(root,out,cfg,snapshot):
    torch.set_num_threads(4);torch.set_float32_matmul_precision('medium')
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    model=restore(snapshot);_,tf=transforms();tf=tf.cuda()
    raw=CIFAR100(root/'data',train=False,download=False)
    m=json.loads((root/cfg['manifest']).read_text())
    ids=[i for task in m['partitions']['test'][:3] for group in task for i in group]
    rows=[];ys=[]
    with torch.inference_mode():
        for x,y in loader(raw,ids,cfg['batch']):
            x=tf(x.cuda())
            with torch.autocast('cuda',dtype=torch.float16):scores=model(x)[:,:30]
            rows.append(scores.float().cpu().numpy());ys.append(y.numpy())
    a=np.concatenate(rows).astype(np.float64);y=np.concatenate(ys)
    with np.load(out/'task3_private_logits.npz',allow_pickle=False) as saved:
        assert np.array_equal(y,saved['labels']) and np.array_equal(ids,saved['indices'])
        raw_equal=np.array_equal(a.argmax(1),saved['logits'].argmax(1))
        centered_equal=np.array_equal(center_task_logits(a).argmax(1),center_task_logits(saved['logits']).argmax(1))
    assert raw_equal and centered_equal,'Final LoRM checkpoint predictions do not reproduce task-end predictions'
    return {'final_checkpoint_raw_predictions_equal':bool(raw_equal),'final_checkpoint_centered_predictions_equal':bool(centered_equal),'audit_images':len(y)}
