"""Create seed-specific partitions from existing label metadata; no pixel reads."""
import json, os
from pathlib import Path
import fcl.data as data


def main():
    root,source=Path(os.environ['PROJECT_ROOT']),Path(os.environ['SOURCE_ROOT'])
    source_manifest=source/'manifests/seq-cifar100_n10_a0.05_s0.json'
    base=json.loads(source_manifest.read_text())
    if base['identity'] != {'dataset':'seq-cifar100','seed':0,'alpha':.05,'clients':10}:
        raise ValueError('seed-zero metadata identity mismatch')
    labels=base['labels']
    if len(labels['train'])!=50000 or len(labels['test'])!=10000:
        raise ValueError('invalid label metadata lengths')
    data.UPSTREAM=source/'sources/fed-mammoth'
    dest=root/'manifests';dest.mkdir(exist_ok=True)
    for seed in (1,2):
        partitions=data.upstream_split(labels['train'],labels['test'],10,10,.05,seed)
        result={'identity':{'dataset':'seq-cifar100','seed':seed,'alpha':.05,'clients':10},
                'class_order':list(range(100)),'cpt':10,'partitions':partitions,'labels':labels}
        for split in ('train','test'):
            ids=[x for task in partitions[split] for client in task for x in client]
            if sorted(ids)!=list(range(len(labels[split]))):raise ValueError('split not a partition')
            for t, groups in enumerate(partitions[split]):
                if len(groups)!=10 or any(not 10*t<=labels[split][i]<10*(t+1) for g in groups for i in g):
                    raise ValueError('task/client partition mismatch')
        path=dest/f'seed{seed}.json'
        with path.open('x') as f:json.dump(result,f)
    print(json.dumps({'status':'completed','seeds':[1,2],
                      'source':'existing label metadata','pixel_reads':0,'test_scores':0}))

if __name__=='__main__':main()
