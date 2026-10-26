"""Complete-trajectory reporting and final checkpoint qualification; no fitting."""
import json
import numpy as np
import torch


def trajectory(matrix):
    assert len(matrix)==10 and all(len(row)==t+1 for t,row in enumerate(matrix))
    diagonal=[row[t] for t,row in enumerate(matrix)]
    last=matrix[-1]
    return {'matrix':matrix,'a_tt':diagonal,'mean_new_T2_T10':sum(diagonal[1:])/9,
            'final_all_seen':sum(last)/10,'final_old':sum(last[:-1])/9,'final_new':last[-1],
            'average_incremental_accuracy':sum(sum(r)/len(r) for r in matrix)/10,
            'forgetting':sum(max(matrix[j][i] for j in range(i,10))-last[i] for i in range(9))/9}


def report(out,method):
    if method=='regmean':
        stages=json.loads((out/'metrics.json').read_text())['stages']
        raw=[s['per_task'] for s in stages]
        centered=[[v['all_seen_accuracy'] for v in s['centered']['per_task']] for s in stages]
        competition=[{'task':i+1,'raw_old_to_current':s['old_to_new_error'],
                      'raw_current_to_old':s['new_to_old_error'],'centered':s['centered']} for i,s in enumerate(stages)]
    else:
        stages=[json.loads((out/f'task{t}_fixed_rules.json').read_text()) for t in range(1,11)]
        raw=[[v['all_seen_accuracy'] for v in s['raw']['per_task']] for s in stages]
        centered=[[v['all_seen_accuracy'] for v in s['centered']['per_task']] for s in stages]
        competition=[{'task':i+1,**s} for i,s in enumerate(stages)]
    result={'raw':trajectory(raw),'centered':trajectory(centered),'competition':competition,
            'primary_rule':'fixed centered logits for Head and equally centered LoRM; raw reported for both',
            'selection':'All ten stages and all three seeds; no performance-based exclusion'}
    (out/'TRAJECTORY.json').write_text(json.dumps(result,indent=2))
    return result


def verify_lorm(root,out,cfg):
    import timm
    from torchvision.datasets import CIFAR100
    from fcl.data import loader,transforms
    from research.basic_repair.fcl_basic_repair_tools import center_task_logits
    records=[json.loads(v) for v in (out/'events.jsonl').read_text().splitlines()]
    clients=[v for v in records if v['kind']=='client']
    assert len(clients)==500 and sum(v['visited'] for v in clients)==1250000
    assert len({(v['task'],v['round'],v['client']) for v in clients})==500
    assert all(v['visited']==v['N']*5 and v['train_matrix']==('B' if v['round']%2==0 else 'A') for v in clients)
    cp=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
    assert cp['task']==10 and cp['round']==0 and not cp['initialized'] and cp['config']==cfg
    metrics=json.loads((out/'metrics.json').read_text());assert cp['matrix']==metrics['matrix'] and len(cp['matrix'])==10
    torch.set_num_threads(4);torch.set_float32_matmul_precision('medium')
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    weights=cp['server']['attributes']['optimization_dict']
    assert all(k.startswith('model.') for k in weights)
    model=timm.create_model('vit_base_patch16_224.augreg_in21k',pretrained=False,num_classes=100)
    model.load_state_dict({k.removeprefix('model.'):v for k,v in weights.items()},strict=True)
    model=model.eval().requires_grad_(False).cuda();_,tf=transforms();tf=tf.cuda()
    raw=CIFAR100(root/'data',train=False,download=False);m=json.loads((root/cfg['manifest']).read_text())
    assert list(raw.targets)==m['labels']['test']
    ids=[i for task in m['partitions']['test'] for group in task for i in group]
    assert len(ids)==len(set(ids))==10000
    rows=[];ys=[]
    with torch.inference_mode():
        for x,y in loader(raw,ids,cfg['batch']):
            x=tf(x.cuda())
            with torch.autocast('cuda',dtype=torch.float16):scores=model(x)
            rows.append(scores.float().cpu().numpy());ys.append(y.numpy())
    a=np.concatenate(rows).astype(np.float64);y=np.concatenate(ys)
    with np.load(out/'task10_private_logits.npz',allow_pickle=False) as saved:
        assert np.array_equal(y,saved['labels']) and np.array_equal(ids,saved['indices'])
        raw_equal=np.array_equal(a.argmax(1),saved['logits'].argmax(1))
        centered_equal=np.array_equal(center_task_logits(a).argmax(1),center_task_logits(saved['logits']).argmax(1))
        max_logit_difference=float(np.abs(a-saved['logits']).max())
    assert raw_equal and centered_equal,'Final LoRM checkpoint must reproduce both task-end prediction rules'
    result={'qualified':True,'client_rounds':500,'training_visits':1250000,
            'optimizer_steps':sum(v['steps'] for v in clients),'checkpoint_boundary_complete':True,
            'final_checkpoint_raw_predictions_equal':bool(raw_equal),'final_checkpoint_centered_predictions_equal':bool(centered_equal),
            'max_reload_logit_difference':max_logit_difference,'audit_images':10000,
            'statistics_image_forwards':300000,'round_evaluation_images':275000,'task_evaluation_images':55000,
            'note':'Pinned official learner; end-round and end-task statistics retained, no performance gate'}
    (out/'QUALIFICATION.json').write_text(json.dumps(result,indent=2));return result
