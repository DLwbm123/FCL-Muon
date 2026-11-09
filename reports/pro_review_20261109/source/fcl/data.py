"""Pinned upstream split executed on indices instead of duplicating image arrays."""
import ast
import json
import random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader

UPSTREAM=Path(__file__).resolve().parents[1]/'sources/fed-mammoth'


def upstream_split(train_labels,test_labels,cpt,clients,alpha,seed):
    # Reuse the actual upstream function without importing its unrelated NLP deps.
    source=(UPSTREAM/'_datasets/_utils.py').read_text()
    tree=ast.parse(source)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BaseDataset')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_split_fcil')
    module=ast.Module(body=[fn],type_ignores=[])
    ns={'np':np,'torch':torch}
    exec(compile(ast.fix_missing_locations(module),str(UPSTREAM/'_datasets/_utils.py'),'exec'),ns)
    obj=SimpleNamespace(N_CLASSES_PER_TASK=cpt,N_TASKS=10,train_data=[],train_targets=[],test_data=[],test_targets=[])
    for split,labels in [('train',train_labels),('test',test_labels)]:
        setattr(obj,split+'_dataset',SimpleNamespace(data=np.arange(len(labels)),targets=np.array(labels)))
    np_state,torch_state=np.random.get_state(),torch.get_rng_state()
    try:
        np.random.seed(seed);torch.manual_seed(seed)
        ns['_split_fcil'](obj,clients,'distribution',alpha)
    finally:
        np.random.set_state(np_state);torch.set_rng_state(torch_state)
    return {split:[[x.tolist() for x in task] for task in getattr(obj,split+'_data')] for split in ['train','test']}


def load_raw(dataset,root):
    from torchvision.datasets import CIFAR100
    from PIL import Image
    import yaml
    if dataset=='seq-cifar100':
        return [CIFAR100(root,train=t,download=True) for t in (True,False)],10,100
    class Images(Dataset):
        def __init__(self,split):
            cfg=yaml.safe_load((UPSTREAM/f'imagenet_r_utils/imagenet-r_{split}.yaml').read_text())
            self.paths=[Path(root)/p.removeprefix('data/') for p in cfg['data']]
            self.targets=cfg['targets']
            assert len(self.paths)==(24000 if split=='train' else 6000)
            missing=next((p for p in self.paths if not p.is_file()),None)
            if missing: raise FileNotFoundError(missing)
        def __len__(self):return len(self.targets)
        def __getitem__(self,i):
            with Image.open(self.paths[i]) as im:return im.convert('RGB'),self.targets[i]
    return [Images('train'),Images('test')],20,200


class Indexed(Dataset):
    def __init__(self,raw,indices):
        from torchvision import transforms as T
        self.raw,self.indices=raw,indices
        self.base=T.Compose([T.Resize((224,224),interpolation=T.InterpolationMode.BICUBIC),T.ToTensor()])
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):
        x,y=self.raw[self.indices[i]]
        return self.base(x),y


def loader(raw,indices,batch,shuffle=False):
    return DataLoader(Indexed(raw,indices),batch_size=batch,shuffle=shuffle,drop_last=False,num_workers=0)


def transforms():
    from kornia import augmentation as K
    train=K.AugmentationSequential(K.RandomResizedCrop((224,224),resample='bicubic'),K.RandomHorizontalFlip(p=.5),K.Normalize((.5,)*3,(.5,)*3))
    test=K.AugmentationSequential(K.Resize((256,256),resample='bicubic'),K.CenterCrop((224,224)),K.Normalize((.5,)*3,(.5,)*3))
    return train,test


def manifest(raw,cpt,classes,cfg,path):
    path=Path(path)
    identity={k:cfg[k] for k in ('dataset','seed','alpha','clients')}
    if path.exists():
        m=json.loads(path.read_text())
        assert m['identity']==identity
        return m
    m={'identity':identity,'class_order':list(range(classes)),'cpt':cpt,
       'partitions':upstream_split(raw[0].targets,raw[1].targets,cpt,cfg['clients'],cfg['alpha'],cfg['seed']),
       'labels':{'train':list(map(int,raw[0].targets)),'test':list(map(int,raw[1].targets))}}
    for split in ('train','test'):
        ids=[i for task in m['partitions'][split] for client in task for i in client]
        assert sorted(ids)==list(range(len(raw[0 if split=='train' else 1])))
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(m))
    return m


def current_training_indices(m, task):
    ids=[i for client in m['partitions']['train'][task] for i in client]
    low,high=task*m['cpt'],(task+1)*m['cpt']
    if any(not low <= m['labels']['train'][i] < high for i in ids):
        raise ValueError('Future/old class in current training statistics')
    return ids
