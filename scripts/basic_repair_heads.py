"""Fixed-encoder T1 head diagnostics. Hyperparameters are locked before test use."""
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def fit_global_cosine(parts, initial, steps, lr):
    """One server head/state; sum client gradients of the pooled mean objective."""
    w = torch.nn.Parameter(initial.detach().clone())
    optimizer = torch.optim.AdamW([w], lr=lr, weight_decay=0)
    n = sum(len(y) for z,y in parts)
    losses = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        total = 0.
        for z,y in parts:
            loss = F.cross_entropy(16*z@F.normalize(w, dim=1).T, y, reduction='sum')/n
            loss.backward()
            total += float(loss.detach())
        optimizer.step()
        if step in (0,249,499,749,999):
            losses.append({'step': step+1, 'pooled_train_ce_before_step': total})
    return w.detach(), losses


def metrics(scores, labels):
    scores = np.asarray(scores)
    pred = scores.argmax(1)
    confusion = np.bincount(labels*10+pred, minlength=100).reshape(10,10)
    return {'accuracy': float((pred==labels).mean()), 'n': len(labels),
            'per_class_accuracy': (confusion.diagonal()/confusion.sum(1)).tolist(),
            'confusion': confusion.tolist(), 'class6_to7_count': int(confusion[6,7])}


def main():
    import timm
    from safetensors.torch import load_file
    from torchvision.datasets import CIFAR100
    from fcl.model import MuLoRAFL
    from fcl.data import loader, transforms
    from fcl.checkpoint import atomic_save
    from research.basic_repair.fcl_basic_repair_tools import fixed_feature_statistics, merge_statistics, global_prototype_rows, global_ridge
    root = Path(os.environ['PROJECT_ROOT'])
    out = root/'runs/head_diagnostics'
    out.mkdir(exist_ok=False)
    protocol = json.loads((root/'research/basic_repair/LOCKED_PROTOCOL.json').read_text())
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    manifest = json.loads((root/'manifests'/protocol['manifest']).read_text())
    raw_train = CIFAR100(root/'data', train=True, download=False)
    raw_test = CIFAR100(root/'data', train=False, download=False)
    assert list(raw_train.targets)==manifest['labels']['train'] and list(raw_test.targets)==manifest['labels']['test']
    train_parts = manifest['partitions']['train'][0]
    test_ids = [i for group in manifest['partitions']['test'][0] for i in group]
    assert sum(map(len,train_parts))==5000 and len(test_ids)==1000
    _, tf = transforms()
    tf = tf.cuda()
    checkpoint = torch.load(root/'runs/t1_original/latest.pt', map_location='cpu', weights_only=False)
    assert checkpoint['task']==1 and checkpoint['round']==0 and not checkpoint['initialized']
    model = MuLoRAFL(timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=False, num_classes=0),100,32,16)
    model.add_head(10)
    model.load_state_dict(checkpoint['model'], strict=True)
    model = model.eval().requires_grad_(False).cuda()
    original_head = model.rows[0].detach().clone()
    assert all(torch.count_nonzero(b)==0 for b in model.b_parameters())
    # Check merge equivalence on one fixed test batch; it never selects a model.
    before = torch.load(root/'runs/t1_original/boundary_t0_r5.pt', map_location='cpu', weights_only=False)
    first_x,_ = next(iter(loader(raw_test, test_ids[:16],16)))
    with torch.inference_mode():
        x = tf(first_x.cuda())
        post_scores = model(x)
        model.load_state_dict(before['model'], strict=True)
        pre_scores = model(x)
        merge_error = float((post_scores-pre_scores).abs().max())
        torch.testing.assert_close(pre_scores,post_scores,atol=5e-5,rtol=5e-5)
        model.load_state_dict(checkpoint['model'], strict=True)
    del before
    results = {'protocol':protocol, 'merge_max_abs_logit_error':merge_error,
               'training_identity':'Original full 5000 T1 training images; no held-out validation is claimed',
               'test_identity':'Original 1000 T1 test images; fixed head diagnostics only, no selection',
               'lorm_reference':{'T1_final_accuracy':.977,'rerun':False,'same_encoder_probes_available':False},
               'encoders':{}}
    def extract(raw, ids):
        zs,ys = [],[]
        with torch.inference_mode():
            for x,y in loader(raw,ids,16):
                z = model.backbone.forward_features(tf(x.cuda()))[:,0]
                zs.append(F.normalize(z,dim=1).cpu());ys.append(y)
        return torch.cat(zs),torch.cat(ys)
    for encoder in ('adapted','pretrained'):
        if encoder=='pretrained':
            wm = json.loads((root/'weights.json').read_text())
            sd = {k:v for k,v in load_file(wm['path']).items() if not k.startswith('head.')}
            # QKV adapter wrappers require the base keys; all B are kept zero.
            sd = {k.replace('.attn.qkv.', '.attn.qkv.base.'):v for k,v in sd.items()}
            missing,extra = model.backbone.load_state_dict(sd,strict=False)
            assert not extra and all(k.endswith(('.A','.Bq','.Bv')) for k in missing)
            assert all(torch.count_nonzero(b)==0 for b in model.b_parameters())
        parts, stats = [], []
        began = time.time()
        for k,ids in enumerate(train_parts):
            z,y = extract(raw_train,ids)
            # Client-labelled transient T1 cache only, never an old-task replay source.
            atomic_save({'z':z,'y':y,'indices':ids,'encoder':encoder,'scope':'current-task diagnostic only'},out/f'{encoder}_client{k}_features.pt')
            parts.append((z.cuda(),y.cuda()))
            stats.append(fixed_feature_statistics(z.numpy(),y.numpy(),10))
        combined = merge_statistics(stats)
        h1 = global_prototype_rows(combined)
        h3 = global_ridge(combined,protocol['H3']['regularization'])
        mat = combined['gram']/combined['n']
        penalty = np.eye(len(mat))*.01;penalty[-1,-1]=0
        residual = np.linalg.norm((mat+penalty)@h3-combined['cross']/combined['n'])
        # All heads are finalized BEFORE test features/scores for this encoder.
        head_state = {'H1':torch.from_numpy(h1), 'H3':torch.from_numpy(h3)}
        if encoder=='adapted':
            hp = protocol['H2']
            h2,losses = fit_global_cosine(parts,original_head,hp['full_batch_server_steps'],hp['lr'])
            head_state.update(H0=original_head.cpu(),H2=h2.cpu())
            results['H2_training_trace'] = losses
            # Save the predeclared H1 candidate unconditionally, never select by test.
            repaired = copy.deepcopy(checkpoint)
            repaired['model']['rows.0'] = torch.from_numpy(h1).float()
            repaired['repair']={'name':'mulora_t1_global_prototype_head','base_training_config':checkpoint['config'], 'source':'global same-encoder current-train class sums', 'test_selected':False, 'faithful_baseline':False}
            atomic_save(repaired,out/'t1_repaired_H1.pt')
            del repaired
        atomic_save(head_state,out/f'{encoder}_fixed_heads.pt')
        zt,yt = extract(raw_test,test_ids)
        z = zt.numpy().astype(np.float64);y = yt.numpy()
        z = z / np.linalg.norm(z,axis=1,keepdims=True)  # same FP64 normalization as the statistic helpers
        values = {'H1':metrics(16*z@h1.T,y), 'H3':metrics(np.c_[z,np.ones(len(z))]@h3,y)}
        if encoder=='adapted':
            values['H0']=metrics((16*zt@F.normalize(original_head.cpu(),dim=1).T).numpy(),y)
            values['H2']=metrics((16*zt@F.normalize(h2.cpu(),dim=1).T).numpy(),y)
            # Verify the actual saved FP32 H1 checkpoint function, not just FP64 formula.
            values['H1_checkpoint_fp32']=metrics((16*zt@F.normalize(torch.from_numpy(h1).float(),dim=1).T).numpy(),y)
            assert values['H0']['accuracy']==checkpoint['matrix'][0][0]
            local_heads=[]
            local_readouts=[]
            for k,ids in enumerate(train_parts):
                local=torch.load(root/f'runs/t1_original/client_t0_r4_k{k}.pt',map_location='cpu',weights_only=False)
                w=local['head']
                local_heads.append(w)
                local_readouts.append({'client':k,'n_train':len(ids),
                    'on_common_final_encoder':metrics((16*zt@F.normalize(w,dim=1).T).numpy(),y),
                    'row_norms':w.norm(dim=1).tolist()})
            average=sum(len(ids)/5000*w for ids,w in zip(train_parts,local_heads))
            torch.testing.assert_close(average,original_head.cpu(),atol=1e-6,rtol=1e-5)
            results['final_client_heads_on_common_global_encoder']=local_readouts
            results['head_aggregation_max_abs_error']=float((average-original_head.cpu()).abs().max())
        else:
            values.update(H0={'status':'not_applicable: no trained CIFAR head'},H2={'status':'not_in_prespecified_scope'})
        results['encoders'][encoder]={'readouts':values,'ridge_normal_equation_residual_fro':float(residual),'class_counts':combined['counts'].tolist(),'seconds':time.time()-began}
        (out/'results.json').write_text(json.dumps(results,indent=2))
        print(encoder,{k:v.get('accuracy') for k,v in values.items()},flush=True)
        del parts,stats
    (out/'COMPLETE.json').write_text(json.dumps({'completed':time.time(),'readouts':6,'history_executed':False}))


if __name__=='__main__':
    main()
