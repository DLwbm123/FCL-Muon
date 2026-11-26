"""Same-manifest run of STSA-E and the fully supervised FedRAN v2 ViT recipe."""

import importlib.util
import json
import math
import os
import time
from types import SimpleNamespace

import numpy as np
import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import CIFAR100

from fcl.data import loader, transforms
from research.adamw_t10.pilot import centered_logits, metrics
from research.adamw_t10.storage import save, write
from research.stat_ablation.runtime import setup, validate


def build_adapter(root, weights, width):
    source = root / 'reference/vision_transformer_adapter.py'
    spec = importlib.util.spec_from_file_location('upstream_adapter', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tuning = SimpleNamespace(
        ffn_adapt=True, ffn_option='parallel', ffn_adapter_layernorm_option='none',
        ffn_adapter_init_option='lora', ffn_adapter_scalar='0.1', ffn_num=width,
        d_model=768, vpt_on=False, vpt_num=0,
    )
    model = module.VisionTransformer(num_classes=0, tuning_config=tuning)
    state = {}
    for name, tensor in load_file(weights).items():
        if name.startswith('head.'):
            continue
        if '.attn.qkv.' in name:
            for projection, chunk in zip(('q', 'k', 'v'), tensor.chunk(3)):
                state[name.replace('.qkv.', f'.{projection}_proj.')] = chunk
        else:
            state[name.replace('.mlp.fc', '.fc')] = tensor
    missing, unexpected = model.load_state_dict(state, strict=False)
    adapters = {name for name, _ in model.named_parameters() if '.adaptmlp.' in name}
    if set(missing) != adapters or unexpected:
        raise ValueError(f'Unexpected pretrained mapping: missing={missing}, unexpected={unexpected}')
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in adapters)
    return model.cuda()


class CosineHead(nn.Module):
    def __init__(self, dim, classes):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(classes, dim).uniform_(-dim**-0.5, dim**-0.5))
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, features):
        return self.scale * cosine(features, self.weight)


def cosine(features, weight):
    return F.linear(F.normalize(features, dim=-1), F.normalize(weight, dim=-1))


def t1_adapt(backbone, data, groups, transform, spec, check_budget, progress):
    model = nn.Sequential(backbone, CosineHead(backbone.num_features, 10).cuda()).train()
    params = [p for p in model.parameters() if p.requires_grad]
    round_payload = sum(p.numel() * p.element_size() for p in params)
    rng = np.random.RandomState(spec['seed'])
    visits, steps = 0, 0
    for round_index in range(spec['rounds']):
        starts = [p.detach().clone() for p in params]
        totals = [torch.zeros_like(p) for p in params]
        lr = spec['learning_rate'] * (1 + math.cos(math.pi * round_index / spec['rounds'])) / 2
        for client in rng.permutation(len(groups)):
            with torch.no_grad():
                for param, start in zip(params, starts):
                    param.copy_(start)
            opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=spec['weight_decay'])
            batches = loader(data, groups[client], spec['batch'], shuffle=True)
            for _ in range(spec['epochs']):
                for x, y in batches:
                    check_budget()
                    opt.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(model(transform(x.cuda())), y.cuda())
                    if not torch.isfinite(loss):
                        raise ArithmeticError('Nonfinite first-task loss')
                    loss.backward()
                    opt.step()
                    visits += len(y)
                    steps += 1
            with torch.no_grad():
                for total, param in zip(totals, params):
                    total.add_(param)
        with torch.no_grad():
            for param, total in zip(params, totals):
                param.copy_(total / len(groups))
        progress(round_index + 1, visits, steps)
    if visits != sum(map(len, groups)) * spec['epochs'] * spec['rounds']:
        raise ValueError('First-task training exposure does not match protocol')
    return dict(training_visits=visits, optimizer_updates=steps,
                adapter_and_head_exchange_bytes=2 * len(groups) * spec['rounds'] * round_payload,
                final_adapted_encoder_download_bytes=len(groups) * sum(
                    p.numel() * p.element_size() for p in backbone.parameters() if p.requires_grad))


def proposal(dim, width, seed):
    rng = torch.Generator().manual_seed(seed)
    return torch.randn(dim, width, generator=rng).cuda()


def class_sums(features, labels, classes):
    indicator = F.one_hot(labels, classes).to(features)
    return indicator.T @ features, indicator.sum(dim=0)


def stsa_packets(features, labels, groups, classes, generator):
    indices = torch.randperm(len(labels), generator=generator, device=labels.device)
    size = len(labels) // groups
    return [class_sums(features[part], labels[part], classes)
            for part in indices.split([size] * (groups - 1) + [len(labels) - size * (groups - 1)])]


def stsa_estimate(packets):
    totals = torch.stack([p[0] for p in packets])
    sizes = torch.stack([p[1] for p in packets])
    class_counts = sizes.sum(0)
    if bool((class_counts <= 0).any()) or len(packets) <= 1:
        raise ValueError('Cannot estimate task Gram without all classes and multiple shards')
    sums = totals.sum(0)
    k = len(packets)
    # The released STSA-E recipe clamps both coefficients below by zero.
    a = ((class_counts - 1) / (k - 1)).clamp_min(0)
    b = ((class_counts - k) / ((k - 1) * class_counts)).clamp_min(0)
    scaled = (totals * (a / sizes.clamp_min(1)).sqrt().unsqueeze(-1)).flatten(0, 1)
    return sums.T, scaled.T @ scaled - sums.T @ (b[:, None] * sums)


def fedran_packet(features, labels, classes, rank):
    _, singular, vectors = torch.linalg.svd(features, full_matrices=False, driver='gesvd' if features.is_cuda else None)
    width = min(rank, len(singular))
    return (vectors[:width].T.contiguous(), singular[:width]), class_sums(features, labels, classes)[0].T


def spectral_merge(left, right, rank):
    if left is None:
        return right
    A = torch.cat([v * s for v, s in (left, right)], dim=1)
    Q, R = torch.linalg.qr(A, mode='reduced')
    u, s, _ = torch.linalg.svd(R, full_matrices=False, driver='gesvd' if R.is_cuda else None)
    return Q @ u[:, :rank], s[:rank]


def spectral_ridge(summary, label_sums, ridge):
    V, S = summary
    return V @ ((V.T @ label_sums) / (S.square() + ridge)[:, None])


def get_features(backbone, data, indices, batch, transform, check_budget):
    features, labels = [], []
    with torch.no_grad():
        for x, y in loader(data, indices, batch):
            check_budget()
            features.append(backbone(transform(x.cuda())))
            labels.append(y.cuda())
    return torch.cat(features), torch.cat(labels)


def score(features, labels, projection, classifier, stages, task_classes, mode):
    expanded = (features @ projection).relu()
    logits = cosine(expanded, classifier.T) if mode == 'stsa_e' else expanded @ classifier
    classes = logits.shape[1]
    if not torch.isfinite(logits).all():
        raise ValueError('Nonfinite test logits')
    results = []
    for values, key in ((logits, None), (centered_logits(logits, task_classes), 'centered')):
        cm = torch.bincount(labels * classes + values.argmax(1), minlength=classes**2)
        previous = [stage['per_task'] if key is None else stage[key]['per_task'] for stage in stages]
        results.append(metrics(cm.reshape(classes, classes).cpu(), task_classes, previous))
    result, centered = results
    result['centered'] = centered
    return result


def mathematical_checks():
    rng = torch.Generator().manual_seed(30)
    features = torch.randn(15, 6, generator=rng, dtype=torch.float64)
    labels = torch.arange(15) % 3
    parts = list(zip(features.split([5, 10]), labels.split([5, 10])))
    merge, totals = None, torch.zeros(6, 3, dtype=features.dtype)
    for feats, y in parts:
        summary, sums = fedran_packet(feats, y, 3, 6)
        merge = spectral_merge(merge, summary, 6)
        totals += sums
    direct = torch.linalg.solve(features.T @ features + 2 * torch.eye(6), totals)
    torch.testing.assert_close(spectral_ridge(merge, totals, 2), direct)
    packets = [class_sums(feats, y, 3) for feats, y in parts]
    sums, gram = stsa_estimate(packets)
    torch.testing.assert_close(sums, totals)
    expected = torch.zeros_like(gram)
    for class_index in range(3):
        for class_sums_, counts in packets:
            expected += 4 * torch.outer(class_sums_[class_index], class_sums_[class_index]) / counts[class_index]
        expected -= (3 / 5) * torch.outer(sums[:, class_index], sums[:, class_index])
    torch.testing.assert_close(gram, expected)


def main():
    setup()
    root, freeze = validate()
    spec = freeze['analytic_baselines']
    reports = root / 'src/reports/analytic'
    output = root / 'src/runs/analytic'
    reports.mkdir(exist_ok=False)
    output.mkdir(exist_ok=False)
    began = time.monotonic()

    def check_budget():
        if time.time() >= freeze['deadline'] - 120:
            raise TimeoutError('Original clock reached checkpoint reserve')

    def progress(round_index, visits, steps):
        record = dict(round=round_index, training_visits=visits, updates=steps)
        write(root, reports / 'T1_ADAPTATION.json', record)
        print(json.dumps(record), flush=True)

    torch.manual_seed(spec['seed'])
    mathematical_checks()
    manifest = json.loads((root / 'src/manifests/seed0.json').read_text())
    raw = [CIFAR100(root / 'src/data', train=train, download=False) for train in (True, False)]
    if any(dataset.targets != manifest['labels'][key] for dataset, key in zip(raw, ('train', 'test'))):
        raise ValueError('CIFAR-100 labels differ from frozen split')
    parts = manifest['partitions']
    train_tf, test_tf = [tf.cuda() for tf in transforms()]
    backbone = build_adapter(root, freeze['weights']['path'], spec['adapter_width'])
    cost = t1_adapt(backbone, raw[0], parts['train'][0], train_tf, spec, check_budget, progress)
    backbone.eval().requires_grad_(False)
    projections = {mode: proposal(backbone.num_features, spec[mode]['width'], spec['seed']) for mode in ('fedran', 'stsa_e')}
    stsa_generator = torch.Generator(device='cuda').manual_seed(spec['seed'])
    data_types = {mode: projections[mode].dtype for mode in projections}
    histories = {mode: [] for mode in projections}
    charges = {mode: {'statistics_upload_bytes': 0, 'head_download_bytes': 0} for mode in projections}
    stsa_gram, fedran_summary = None, None
    cumulative = {mode: None for mode in projections}
    for t, clients in enumerate(parts['train']):
        cpt = freeze['config']['cpt']
        task_spectral, packets = None, []
        fedran_sums = torch.zeros(spec['fedran']['width'], cpt, device='cuda', dtype=torch.float64)
        for indices in clients:
            features, labels = get_features(backbone, raw[0], indices, spec['feature_batch'], test_tf, check_budget)
            current = labels - t * cpt
            fedran_features = (features.double() @ projections['fedran'].double()).relu()
            summary, sums = fedran_packet(fedran_features, current, cpt, spec['fedran']['rank'])
            # FP32 is the declared communication encoding; merge only decoded summaries.
            summary = tuple(tensor.float().double() for tensor in summary)
            task_spectral = spectral_merge(task_spectral, summary, spec['fedran']['rank'])
            fedran_sums += sums.float().double()
            charges['fedran']['statistics_upload_bytes'] += 4 * (
                sum(x.numel() for x in summary) + sums.shape[0] * (t + 1) * cpt
            )
            stsa_features = (features.double() @ projections['stsa_e'].double()).relu()
            client_packets = stsa_packets(stsa_features, current, spec['stsa_e']['dummy'], cpt, stsa_generator)
            packets.extend(client_packets)
            charges['stsa_e']['statistics_upload_bytes'] += sum(a.numel() * 8 + b.numel() * 8 for a, b in client_packets)
        num_train = sum(map(len, clients))
        if num_train != 5000 or sum(int(counts.sum()) for _, counts in packets) != num_train:
            raise ValueError('Task statistics do not include each current training sample once')
        fedran_summary = spectral_merge(fedran_summary, task_spectral, spec['fedran']['rank'])
        stsa_sums, gram = stsa_estimate(packets)
        stsa_gram = gram if stsa_gram is None else stsa_gram + gram
        for mode, new_sums in zip(('fedran', 'stsa_e'), (fedran_sums, stsa_sums)):
            cumulative[mode] = new_sums if t == 0 else torch.cat((cumulative[mode], new_sums), dim=1)
        classifiers = {
            'fedran': spectral_ridge(fedran_summary, cumulative['fedran'], spec['fedran']['ridge']),
            'stsa_e': torch.linalg.solve(
                stsa_gram + spec['stsa_e']['ridge'] * torch.eye(len(stsa_gram), device='cuda', dtype=stsa_gram.dtype),
                cumulative['stsa_e'],
            ),
        }
        indices = [i for groups in parts['test'][:t + 1] for group in groups for i in group]
        features, labels = get_features(backbone, raw[1], indices, spec['feature_batch'], test_tf, check_budget)
        if len(labels) != 1000 * (t + 1):
            raise ValueError('Unexpected all-seen evaluation size')
        for mode, classifier in classifiers.items():
            classifiers[mode] = classifier.to(data_types[mode])
            charges[mode]['head_download_bytes'] += len(clients) * classifier.numel() * 4
            histories[mode].append(score(features, labels, projections[mode], classifiers[mode], histories[mode], cpt, mode))
        record = {mode: dict(
            raw_AIA=float(np.mean([s['all_seen'] for s in stages])),
            raw_final=stages[-1]['all_seen'],
            centered_AIA=float(np.mean([s['centered']['all_seen'] for s in stages])),
            stages=stages, logical_bytes=cost | charges[mode],
        ) for mode, stages in histories.items()}
        write(root, reports / f'T{t + 1}.json', record)
        print(json.dumps({'task': t + 1, **{mode: row['raw_AIA'] for mode, row in record.items()}}), flush=True)

    save(root, output / 'latest.pt', dict(model=backbone.state_dict(), classifiers=classifiers, projections=projections))
    recovered = torch.load(output / 'latest.pt', map_location='cuda', weights_only=True)
    backbone.load_state_dict(recovered['model'], strict=True)
    features, labels = get_features(backbone, raw[1], indices, spec['feature_batch'], test_tf, check_budget)
    for mode in record:
        expected = histories[mode][-1]
        if score(features, labels, recovered['projections'][mode], recovered['classifiers'][mode],
                 histories[mode][:-1], cpt, mode) != expected:
            raise RuntimeError(f'Final reload changed {mode} metrics')
    write(root, reports / 'COMPLETE.json', dict(status='completed', tasks=len(parts['train']), branches=record,
          final_checkpoint_reload_equal=True, training_shared_t1_once=True, real_analytic_train_samples=50000,
          wall_seconds=time.monotonic() - began, peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved()))


if __name__ == '__main__':
    main()
