"""Entry for the single authorized Head-RegMean continuation; env-only paths."""
import hashlib,json,os,time
from pathlib import Path
import torch
import timm
from torchvision.datasets import CIFAR100
from fcl.data import loader,transforms
from fcl.model import MuLoRAFL
from fcl.checkpoint import rng_state
from research.head_regmean.pilot import Pilot,observe

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=Path(os.environ['PROJECT_ROOT']);original=Path(os.environ['EXPERIMENT_ROOT'])
    cfg=json.loads((root/'research/head_regmean/PILOT_PROTOCOL.json').read_text())
    job=json.loads((root/'reports/job.json').read_text());clock={k:job[k] for k in ['started','deadline']}
    source=original/cfg['source_checkpoint'];before=(source.stat().st_size,source.stat().st_mtime_ns)
    legacy=torch.load(source,map_location='cpu',weights_only=False)
    mf=original/'manifests'/cfg['manifest'];manifest=json.loads(mf.read_text())
    assert manifest['identity']=={k:cfg[k] for k in ['dataset','seed','alpha','clients']}
    raw=[CIFAR100(original/'data',train=v,download=False) for v in [True,False]]
    assert [list(d.targets) for d in raw]==[manifest['labels']['train'],manifest['labels']['test']]
    meta=json.loads((original/'weights.json').read_text())
    identity={'manifest':digest(mf),'weights':meta['sha256'],'source_code':digest(root/'research/head_regmean/pilot.py'),
        'solver':digest(root/'research/head_regmean/solver.py'),'source_checkpoint':cfg['source_checkpoint']}
    def model():return MuLoRAFL(timm.create_model(cfg['backbone'],pretrained=False,num_classes=0),100,cfg['rank'],cfg['cosine_scale']).cuda()
    tr,te=transforms();tr=tr.cuda();te=te.cuda();out=root/'runs/pilot'
    p=Pilot(model(),raw,manifest,loader,tr,te,cfg,out,identity,clock)
    assert not (out/'latest.pt').exists(),'Existing continuation must not be restarted automatically'
    p.import_round1(legacy)
    # Validate the imported state itself (not a fresh image evaluation).
    assert all(torch.equal(v.cpu(),legacy['model'][k]) for k,v in p.model.state_dict().items())
    assert torch.equal(rng_state()['torch'],legacy['rng']['torch'])
    assert all(torch.equal(a,b) for a,b in zip(rng_state()['cuda'],legacy['rng']['cuda']))
    (root/'reports/IMPORT_QUALIFICATION.json').write_text(json.dumps({'passed':True,'source':cfg['source_checkpoint'],
        'model_exact':True,'RNG_exact':True,'round':1,'task':0,'history_commits':0,'original_clock':clock},indent=2))
    del legacy
    p.run()
    records=p.s['training_records'];assert len(records)==cfg['new_training_clients']
    assert sum(r['visited'] for r in records)==cfg['new_training_image_visits']
    assert len({(r['task'],r['round'],r['client']) for r in records})==len(records)
    assert p.s['merges']==[0,1,2] and p.s['history'].committed=={0,1,2}
    assert len(p.s['certificates'])==14 and all(c['deployed_fp32']['passed'] for c in p.s['certificates'])
    assert all(not torch.count_nonzero(b) for b in p.model.b_parameters())
    # Reopen the final merged checkpoint and reproduce full final confusion.
    restored=Pilot(model(),raw,manifest,loader,tr,te,cfg,out,identity,clock);restored.restore()
    restored.s['task']=2
    final=restored.evaluate()
    assert final['confusion']==p.s['stages'][-1]['confusion'],'Pre-/post-merge final confusion mismatch'
    assert (source.stat().st_size,source.stat().st_mtime_ns)==before
    qualification={'qualified':True,'new_training_clients':len(records),
        'new_image_visits':sum(r['visited'] for r in records),'new_optimizer_steps':sum(r['steps'] for r in records),
        'imported_R1_training_clients':10,'tasks_completed':3,'merge_counts':p.s['merges'],
        'history_commits':sorted(p.s['history'].committed),'checkpoint_reload_confusion_equal':True,
        'source_checkpoint_unchanged':True,'numerical_certificates_passed':14,
        'communication_continuation_bytes':p.s['communication'],
        'statistics_samples':sum(r['samples'] for r in p.s['stats_records']),
        'clock':clock,'completed':time.time()}
    (out/'QUALIFICATION.json').write_text(json.dumps(qualification,indent=2))
    (out/'COMPLETE.json').write_text(json.dumps({'completed':time.time(),'tasks':3,'clock':clock}))
    print(json.dumps(qualification),flush=True)

if __name__=='__main__':main()
