"""Read-only final-pilot evaluation; no optimizer, training data or checkpoint writes.

Run via the neutral entry with EXPERIMENT_ROOT pointing to the frozen baseline
checkout. Its fcl modules, dependencies, manifest and checkpoints are reused.
Current-task-restricted accuracy is diagnostic and uses task identity; it is NOT
the primary class-incremental evaluation protocol.
"""
import contextlib
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch


def summarize(logits, labels, cpt=10):
    logits = logits.double()
    labels = labels.long()
    assert logits.ndim == 2 and len(labels) == len(logits)
    assert torch.isfinite(logits).all() and logits.shape[1] % cpt == 0
    seen = logits.shape[1]
    low = seen - cpt
    assert low > 0 and ((labels >= 0) & (labels < seen)).all()
    pred = logits.argmax(1)
    new = labels >= low
    old = ~new
    assert new.any() and old.any()
    restricted = logits[new, low:].argmax(1) + low
    good = restricted == labels[new]
    all_good = pred[new] == labels[new]
    stolen = good & (pred[new] < low)
    # Old indices precede new ones, including ties, in torch.argmax.
    assert int(stolen.sum()) == int(good.sum() - all_good.sum())
    confusion = torch.bincount(labels * seen + pred, minlength=seen**2).reshape(seen, seen)
    margin = logits[new, low:].max(1).values - logits[new, :low].max(1).values
    def rate(x):
        return float(x.double().mean())
    return {
        'n_all': len(labels), 'n_current': int(new.sum()), 'n_old': int(old.sum()),
        'all_seen_accuracy': rate(pred == labels),
        'current_restricted_accuracy': rate(good),
        'current_all_seen_accuracy': rate(all_good),
        'current_competition_loss': rate(stolen),
        'current_restricted_error': rate(~good),
        'current_to_old_rate': rate(pred[new] < low),
        'current_to_wrong_current_rate': rate((pred[new] >= low) & ~all_good),
        'current_correct_but_old_wins_count': int(stolen.sum()),
        'old_all_seen_accuracy': rate(pred[old] == labels[old]),
        'old_to_current_rate': rate(pred[old] >= low),
        'old_to_wrong_old_rate': rate((pred[old] < low) & (pred[old] != labels[old])),
        'per_task_all_seen_accuracy': [rate(pred[labels // cpt == t] == labels[labels // cpt == t]) for t in range(seen // cpt)],
        'per_task_restricted_accuracy': [rate(logits[labels // cpt == t, t*cpt:(t+1)*cpt].argmax(1) + t*cpt == labels[labels // cpt == t]) for t in range(seen // cpt)],
        'current_restricted_ce': float(torch.nn.functional.cross_entropy(logits[new, low:], labels[new] - low)),
        'current_all_seen_ce': float(torch.nn.functional.cross_entropy(logits[new], labels[new])),
        'current_max_new_minus_max_old_quantiles': {
            name: torch.quantile(margin[mask], torch.tensor([0., .25, .5, .75, 1.], dtype=torch.float64)).tolist() if mask.any() else None
            for name, mask in [('all', torch.ones_like(good)), ('restricted_correct', good), ('restricted_wrong', ~good)]},
        'confusion_true_rows_pred_columns': confusion.tolist(),
        'per_class_support': confusion.sum(1).tolist(),
    }


def restore(snapshot):
    import timm
    from fcl.model import MODELS
    cfg = snapshot['config']
    assert snapshot['task'] == 3 and snapshot['round'] == 0 and not snapshot['initialized']
    if cfg['method'] == 'lorm_official':
        # The official forward functional-calls these already temporally merged
        # weights. The upstream wrapper forwards 224x224 input directly to timm.
        weights = snapshot['server']['attributes']['optimization_dict']
        assert all(k.startswith('model.') for k in weights)
        model = timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=False, num_classes=100)
        model.load_state_dict({k.removeprefix('model.'): v for k, v in weights.items()}, strict=True)
    else:
        backbone = timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=False, num_classes=0)
        model = MODELS[cfg['method']](backbone, 100, cfg['rank'], cfg['cosine_scale'])
        assert snapshot['head_count'] == 3
        for _ in range(snapshot['head_count']):
            model.add_head(10)
        model.load_state_dict(snapshot['model'], strict=True)
        assert all(torch.count_nonzero(p) == 0 for p in model.b_parameters())
    return model.eval().requires_grad_(False).cuda()


def signature(path):
    s = path.stat()
    return {'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns}


def main():
    from torchvision.datasets import CIFAR100
    from fcl.data import loader, transforms
    root = Path(os.environ['EXPERIMENT_ROOT'])
    out = root / 'runs/diagnostic_pilot_checkpoint_20261025'
    out.mkdir(parents=True, exist_ok=False)  # never silently repeat a run
    started = time.time()
    (out / 'clock.json').write_text(json.dumps({'started': started, 'deadline': started+1800, 'maximum_output_bytes': 10_000_000}))
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    manifest = json.loads((root / 'manifests/seq-cifar100_n10_a0.05_s0.json').read_text())
    raw = CIFAR100(root / 'data', train=False, download=False)
    assert list(raw.targets) == manifest['labels']['test']
    assert manifest['identity'] == {'dataset': 'seq-cifar100', 'seed': 0, 'alpha': .05, 'clients': 10}
    ids = [i for task in manifest['partitions']['test'][:3] for client in task for i in client]
    assert len(ids) == len(set(ids)) == 3000
    assert set(raw.targets[i] for i in ids) == set(range(30))
    _, test_tf = transforms()
    test_tf = test_tf.cuda()
    results = {'scope': 'Final T3 checkpoint, same 3000 CIFAR100 test examples; current classes20..29, seen classes0..29',
               'restricted_accuracy_warning': 'Uses task identity to mask logits: diagnostic only, not an FCIL result',
               'training_or_tuning': False, 'methods': {}}
    identities = []
    for method in ['mulora_fl', 'mulora_fl_fedmuon', 'lorm_official']:
        checkpoint = root / 'runs' / ('pilot_' + method) / 'latest.pt'
        before = signature(checkpoint)
        for receipt in ['COMPLETE.json', 'QUALIFICATION.json']:
            assert (checkpoint.parent / receipt).is_file()
        snapshot = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert snapshot['config']['method'] == method
        identities.append((snapshot['manifest_hash'], snapshot['weight_hash']))
        assert len(set(identities)) == 1
        saved = json.loads((checkpoint.parent / 'metrics.json').read_text())
        assert saved['matrix'] == snapshot['matrix']
        mixed = method == 'lorm_official'
        torch.set_float32_matmul_precision('medium' if mixed else 'highest')
        torch.backends.cuda.matmul.allow_tf32 = mixed
        torch.backends.cudnn.allow_tf32 = mixed
        model = restore(snapshot)
        config = snapshot['config']
        del snapshot
        gc.collect()
        logits, labels = [], []
        began = time.time()
        with torch.inference_mode():
            for x, y in loader(raw, ids, config['batch']):
                x = test_tf(x.cuda())  # original eval transforms run outside AMP
                with torch.autocast('cuda', dtype=torch.float16) if mixed else contextlib.nullcontext():
                    scores = model(x)[:, :30]
                logits.append(scores.float().cpu())
                labels.append(y)
        logits, labels = torch.cat(logits), torch.cat(labels)
        result = summarize(logits, labels)
        result.update({'checkpoint': str(checkpoint.relative_to(root)), 'config': config,
                       'checkpoint_signature_before': before, 'checkpoint_signature_after': signature(checkpoint),
                       'manifest_hash': identities[-1][0], 'weight_hash': identities[-1][1],
                       'saved_task_accuracy': saved['matrix'][-1],
                       'saved_accuracy_max_abs_difference': max(abs(a-b) for a,b in zip(result['per_task_all_seen_accuracy'], saved['matrix'][-1])),
                       'precision': '16-mixed' if mixed else 'FP32, TF32 disabled', 'seconds': time.time()-began})
        assert result['checkpoint_signature_before'] == result['checkpoint_signature_after']
        # MuLoRA's saved metrics were pre-merge, its final checkpoint is post-merge;
        # report any discrepancy explicitly rather than silently redefining results.
        result['saved_accuracy_exactly_reproduced'] = result['saved_accuracy_max_abs_difference'] == 0
        np.savez_compressed(out / (method + '_private_logits.npz'), logits=logits.numpy(), labels=labels.numpy(), indices=np.array(ids))
        results['methods'][method] = result
        (out / 'results.json').write_text(json.dumps(results, indent=2))
        print(method, json.dumps({k:v for k,v in result.items() if isinstance(v, (int,float,bool))}), flush=True)
        del model, logits, labels
        gc.collect()
        torch.cuda.empty_cache()
    assert sum(p.stat().st_size for p in out.iterdir() if p.is_file()) < 10_000_000
    (out / 'COMPLETE.json').write_text(json.dumps({'ended': time.time(), 'elapsed': time.time()-started, 'methods': list(results['methods']), 'all_saved_metrics_reproduced': all(r['saved_accuracy_exactly_reproduced'] for r in results['methods'].values())}, indent=2))


if __name__ == '__main__':
    main()
