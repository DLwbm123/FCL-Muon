"""One seed, one shared task-2 state, two prespecified head-loss branches."""
import copy,gc,json,os,time
from pathlib import Path
import torch,timm
from torch.nn.attention import sdpa_kernel,SDPBackend
from torchvision.datasets import CIFAR100
from fcl.data import loader,transforms
from fcl.checkpoint import restore_rng
from fcl.model import MuLoRAFL
from research.head_regmean.head_loss import HeadLossPilot

root=Path(os.environ['PROJECT_ROOT']);base=Path(os.environ['EXPERIMENT_ROOT']);seed=int(os.environ['LANE_SEED'])
protocol=json.loads((root/'research/head_regmean/HEAD_LOSS_PROTOCOL.json').read_text())
lane=next(v for v in protocol['lanes'] if v['seed']==seed)
source_path=base/lane['source'];source_before=(source_path.stat().st_size,source_path.stat().st_mtime_ns)
source=torch.load(source_path,map_location='cpu',weights_only=False)
clock=json.loads((root/'reports/job.json').read_text());clock={k:clock[k] for k in ['started','deadline']}
assert clock['started']<=source['clock']['started'] and clock['deadline']<=source['clock']['deadline']
assert source['head_count']==1 and source['state']['task']==1 and source['state']['phase']=='task_start'
assert source['state']['merges']==[0] and source['state']['history'].committed=={0}
assert json.loads((source_path.parent/'QUALIFICATION.json').read_text())['qualified']
cfg={k:source['config'][k] for k in ['dataset','clients','alpha','seed','tasks','cpt','rounds','epochs','batch','rank','lr','cosine_scale','backbone']}
cfg.update(aggregation='regmean',precision='FP32 deterministic algorithms, math SDPA, TF32 off',head_loss_mode='current')
assert cfg['seed']==seed and cfg['tasks']==3 and cfg['cpt']==10
m=json.loads((base/lane['manifest']).read_text())
assert m['identity']=={k:cfg[k] for k in ['dataset','seed','alpha','clients']}
raw=[CIFAR100(base/'data',train=t,download=False) for t in [True,False]]
assert [list(v.targets) for v in raw]==[m['labels']['train'],m['labels']['test']]
torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
identity={'source_checkpoint':lane['source'],'source_identity':source['identity'],'source_clock':source['clock'],
          'experiment':'shared-state head-loss intervention; original failed exact-pair gate remains failed'}
lane_path=root/f'reports/seed{seed}.json';state={'seed':seed,'status':'preparing','branches':[],'source':lane['source']}
def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
def record():write(lane_path,state)
record()

def make(snapshot,mode,out):
    config={**cfg,'head_loss_mode':mode}
    model=MuLoRAFL(timm.create_model(cfg['backbone'],pretrained=False,num_classes=0),100,cfg['rank'],cfg['cosine_scale']).cuda()
    for _ in range(snapshot['head_count']):model.add_head(cfg['cpt'])
    model.load_state_dict(snapshot['model'],strict=True)
    for k,a in model.adapters.items():a.merged=snapshot['merged_flags'][k]
    tr,te=transforms();p=HeadLossPilot(model,raw,m,loader,tr.cuda(),te.cuda(),config,out,identity,clock)
    p.s=copy.deepcopy(snapshot['state']);restore_rng(snapshot['rng'])
    return p

with sdpa_kernel(SDPBackend.MATH):
    shared_out=root/f'runs/seed{seed}/shared';shared_out.mkdir(parents=True,exist_ok=False)
    p=make(source,'current',shared_out)
    # Re-evaluate the inherited T1 once under the new fixed numeric mode. Keep
    # the old report as provenance; both branches inherit the same new result.
    original_T1=copy.deepcopy(p.s['stages'][0]);p.s['task']=0;replayed_T1=p.evaluate();p.s['task']=1
    p.s['matrix'][0]=replayed_T1['per_task'];p.s['stages'][0]=replayed_T1
    p.step() # exactly one common T2 initialization/RRS, no parameter update
    assert p.s['task']==1 and p.s['round']==0 and p.s['phase']=='round_start'
    write(shared_out/'PREPARATION.json',{'source_original_T1':original_T1,'numeric_mode_T1':replayed_T1,
        'source_T1_confusion_equal':original_T1['confusion']==replayed_T1['confusion'],
        'common_RRS_images':5000,'source_replay_test_images':1000,'optimizer_steps':0})
    del p,source;gc.collect();torch.cuda.empty_cache()
    shared=torch.load(shared_out/'latest.pt',map_location='cpu',weights_only=False)
    expected={}
    for mode in ['current','all_seen']:
        p=make(shared,mode,shared_out)
        dl=p.loader(raw[0],p.ids('train',1,0),cfg['batch'],True)
        x,y=next(iter(dl));x=p.train_tf(x.cuda());y=y.cuda();p.model.zero_grad(set_to_none=True)
        p.backward_loss(x,y)
        expected[mode]={'x':x.cpu().clone(),'y':y.cpu().clone(),
            'B':[v.grad.cpu().clone() for v in p.model.b_parameters()],'head':p.model.rows[-1].grad.cpu().clone()}
        assert all(v.grad is None for v in p.model.rows[:-1])
        del p,dl,x,y;gc.collect();torch.cuda.empty_cache()
    a,b=expected['current'],expected['all_seen']
    checks={'input_exact':torch.equal(a['x'],b['x']),'labels_exact':torch.equal(a['y'],b['y']),
        'direct_B_gradients_exact':all(torch.equal(x,y) for x,y in zip(a['B'],b['B'])),
        'head_gradient_changed':not torch.equal(a['head'],b['head'])}
    write(root/f'reports/FIRST_BATCH_seed{seed}.json',{'passed':all(checks.values()),'checks':checks,
        'optimizer_steps':0,'training_images_forward_backward':2*len(a['y']),'common_snapshot':'shared/latest.pt',
        'source_original_exact_pair_gate':'failed, not modified'})
    if not all(checks.values()):state['status']='first_batch_qualification_failed';record();raise RuntimeError(checks)
    for mode in ['current','all_seen']:
        out=root/f'runs/seed{seed}/{mode}';out.mkdir(parents=True,exist_ok=False)
        p=make(shared,mode,out);p.expected_first=expected[mode];p.save()
        start_records=len(p.s['training_records']);start_stats=len(p.s['stats_records']);start_certs=len(p.s['certificates'])
        before_comm=dict(p.s['communication'])
        row={'mode':mode,'status':'running','started':time.time()};state['branches'].append(row);state['status']='running';record()
        p.run();new=p.s['training_records'][start_records:]
        assert len(new)==100 and sum(v['visited'] for v in new)==250000 and p.first_batch_verified
        assert len({(v['task'],v['round'],v['client']) for v in new})==100
        assert len(p.s['certificates'])-start_certs==10 and all(v['deployed_fp32']['passed'] for v in p.s['certificates'][start_certs:])
        assert p.s['merges']==[0,1,2] and p.s['history'].committed=={0,1,2}
        assert all(not torch.count_nonzero(b) for b in p.model.b_parameters())
        p.restore();p.s['task']=2;final=p.evaluate()
        assert final['confusion']==p.s['stages'][-1]['confusion'] and final['centered']==p.s['stages'][-1]['centered']
        centered_matrix=[[v['all_seen_accuracy'] for v in stage['centered']['per_task']] for stage in p.s['stages']]
        qualification={'qualified':True,'new_training_visits':250000,'new_client_rounds':100,
            'new_optimizer_steps':sum(v['steps'] for v in new),'new_statistics_visits':sum(v['samples'] for v in p.s['stats_records'][start_stats:]),
            'common_RRS_images_separately_counted':5000,'preflight_backward_images_per_mode':len(expected[mode]['y']),
            'new_communication_bytes':{k:v-before_comm.get(k,0) for k,v in p.s['communication'].items()},
            'new_certificates_passed':10,'actual_first_batch_matches_preflight':True,'final_checkpoint_reload_confusion_equal':True,
            'centered_matrix':centered_matrix,'centered_forgetting':sum(max(centered_matrix[j][i] for j in range(i,3))-centered_matrix[-1][i] for i in range(2))/2,
            'clock':clock,'source_clock':identity['source_clock'],'head_loss_mode':mode}
        write(out/'QUALIFICATION.json',qualification);write(out/'COMPLETE.json',{'completed':time.time(),'qualified':True})
        row.update(status='completed',ended=time.time());record()
        del p;gc.collect();torch.cuda.empty_cache()
    assert (source_path.stat().st_size,source_path.stat().st_mtime_ns)==source_before
    state.update(status='completed',ended=time.time(),source_signature_unchanged=True);record()
