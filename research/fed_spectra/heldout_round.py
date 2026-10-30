"""Fixed current-task split and runtime guards for a single shared round."""
import random
import torch


def split_clients(groups,seed):
    train=[];validation=[]
    for k,ids in enumerate(groups):
        ordered=sorted(ids)
        if len(set(ordered))!=len(ordered):raise ValueError('Duplicate client indices')
        random.Random(20261030+1009*seed+31*k).shuffle(ordered)
        n=max(1,len(ordered)//5) if len(ordered)>=2 else 0
        validation.append(ordered[:n]);train.append(ordered[n:])
    flat=[i for g in groups for i in g]
    if len(flat)!=len(set(flat)):raise ValueError('Overlapping client partitions')
    assert not set(i for g in train for i in g)&set(i for g in validation for i in g)
    assert sorted(i for g in train+validation for i in g)==sorted(flat)
    return train,validation


class GuardedLoader:
    def __init__(self,loader,raw,training,validation):
        self.loader=loader;self.raw=raw;self.training=set(training);self.validation=set(validation)
        assert not self.training&self.validation
        self.allow_validation=False;self.last_validation=False
        self.training_access_images=0;self.validation_access_images=0;self.validation_model_forward_images=0
        self.validation_grad_violations=0;self.training_validation_accesses=0
    def __call__(self,raw,indices,batch,shuffle=False):
        if raw is not self.raw:raise AssertionError('Only current train dataset is allowed')
        ids=set(indices)
        if not ids<=self.training|self.validation:raise AssertionError('Old/future/test indices prohibited')
        is_validation=bool(ids&self.validation)
        if is_validation and (not self.allow_validation or shuffle or not ids<=self.validation):
            self.training_validation_accesses+=1;raise AssertionError('Validation input entered training/statistics or mixed batch')
        source=self.loader(raw,indices,batch,shuffle)
        owner=self
        class Batches:
            def __len__(self):return len(source)
            def __iter__(self):
                for x,y in source:
                    owner.last_validation=is_validation
                    if is_validation:owner.validation_access_images+=len(y)
                    else:owner.training_access_images+=len(y)
                    yield x,y
        return Batches()
    def model_guard(self,module,args):
        if self.last_validation:
            if torch.is_grad_enabled():
                self.validation_grad_violations+=1;raise AssertionError('Validation model forward with gradients enabled')
            self.validation_model_forward_images+=len(args[0])
