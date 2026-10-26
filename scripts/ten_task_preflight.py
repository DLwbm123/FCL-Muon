"""Read-only full-split and admission checks before the shared production clock."""
import json,os,time
from pathlib import Path
from torchvision.datasets import CIFAR100

root=Path(os.environ['PROJECT_ROOT'])
assert not (root/'reports/job.json').exists()
tests=json.loads((root/'reports/TESTS.json').read_text());assert tests['passed']
raw=[CIFAR100(root/'data',train=t,download=False) for t in [True,False]]
checks=[]
for seed in range(3):
    for method in ['regmean','lorm']:
        cfg=json.loads((root/f'configs/ten_{method}_seed{seed}.json').read_text())
        assert cfg['tasks']==10 and cfg['seed']==seed and cfg['alpha']==.05
        assert cfg['rounds']==cfg['epochs']==5 and cfg['clients']==10 and cfg['batch']==16
        m=json.loads((root/cfg['manifest']).read_text())
        assert m['identity']=={k:cfg[k] for k in ['dataset','seed','alpha','clients']}
        for split,data,count,per_task in [('train',raw[0],50000,5000),('test',raw[1],10000,1000)]:
            assert list(data.targets)==m['labels'][split]
            tasks=m['partitions'][split];assert len(tasks)==10
            flat=[]
            for t,groups in enumerate(tasks):
                assert len(groups)==10
                ids=[i for group in groups for i in group]
                assert len(ids)==per_task and all(t*10<=data.targets[i]<(t+1)*10 for i in ids)
                if split=='train':assert all(len(g)>0 for g in groups)
                flat+=ids
            assert len(flat)==len(set(flat))==count and set(flat)==set(range(count))
        checks.append({'method':method,'seed':seed,'complete_ten_task_manifest':True})
weights=json.loads((root/'weights.json').read_text());assert Path(weights['path']).is_file()
result={'passed':True,'checked':time.time(),'tests':tests,'data':checks,'weight_readable':True,
        'fresh_study_no_old_run_reuse':True,'shared_resources_copied':False,
        'hash_policy':'Existing built-in LoRM digest retained; no extra preflight hashing'}
(root/'reports/PREFLIGHT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
