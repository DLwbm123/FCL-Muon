"""T2 paired local-to-global head response matching, with unchanged old-response B loss."""
import copy,gc,json,os,time
from pathlib import Path
import torch,timm
from torch.nn.attention import sdpa_kernel,SDPBackend
from torchvision.datasets import CIFAR100
from fcl.data import loader,transforms
from fcl.checkpoint import restore_rng
from fcl.model import MuLoRAFL
from research.head_regmean.head_loss import HeadLossPilot,separated_backward
from research.head_regmean.old_response import OldResponsePilot
from research.head_regmean.transport_head import TransportHeadPilot

root=Path(os.environ['PROJECT_ROOT']);base=Path(os.environ['EXPERIMENT_ROOT']);seed=int(os.environ['LANE_SEED'])
protocol=json.loads((root/'research/head_regmean/TRANSPORT_HEAD_PROTOCOL.json').read_text())
lane=next(v for v in protocol['lanes'] if v['seed']==seed)
source_path=base/lane['source'];control=base/lane['control']
source_before=(source_path.stat().st_size,source_path.stat().st_mtime_ns)
source=torch.load(source_path,map_location='cpu',weights_only=False)
control_q=json.loads((control/'QUALIFICATION.json').read_text())
assert control_q['qualified'] and control_q['head_loss_mode']=='centered_all_seen' and control_q['actual_first_batch_matches_preflight']
assert control_q['final_checkpoint_reload_confusion_equal'] and (control/'COMPLETE.json').exists()
assert source['head_count']==2 and source['state']['task']==1 and source['state']['round']==0 and source['state']['phase']=='round_start'
assert source['state']['merges']==[0] and source['state']['history'].committed=={0}
assert source['config']['head_loss_mode']=='current'
clock=json.loads((root/'reports/job.json').read_text());clock={k:clock[k] for k in ['started','deadline']}
assert clock==source['clock']==control_q['clock']
cfg=copy.deepcopy(source['config'])
assert cfg['seed']==seed and cfg['tasks']==3 and cfg['cpt']==10 and cfg['aggregation']=='regmean'
cfg.update(tasks=2,old_response_coefficient=1.,head_aggregation_frame='paired_deployed_global')
assert cfg['precision']=='FP32 deterministic algorithms, math SDPA, TF32 off'
m=json.loads((base/lane['manifest']).read_text());assert m['identity']=={k:cfg[k] for k in ['dataset','seed','alpha','clients']}
raw=[CIFAR100(base/'data',train=t,download=False) for t in [True,False]]
assert [list(v.targets) for v in raw]==[m['labels']['train'],m['labels']['test']]
torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
identity={'source_checkpoint':lane['source'],'source_identity':source['identity'],'source_clock':source['clock'],
          'reused_control':lane['control'],'experiment':'T2 paired local/global response transport with unchanged old-response B loss; prior gates remain failed'}
state={'seed':seed,'status':'preparing','source':lane['source'],'reused_control':lane['control']};lane_path=root/f'reports/seed{seed}.json'
def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
def record():write(lane_path,state)
record()

def make(preserving,out):
    model=MuLoRAFL(timm.create_model(cfg['backbone'],pretrained=False,num_classes=0),100,cfg['rank'],cfg['cosine_scale']).cuda()
    for _ in range(source['head_count']):model.add_head(cfg['cpt'])
    model.load_state_dict(source['model'],strict=True)
    for k,a in model.adapters.items():a.merged=source['merged_flags'][k]
    tr,te=transforms();cls=TransportHeadPilot if preserving else HeadLossPilot
    p=cls(model,raw,m,loader,tr.cuda(),te.cuda(),{**cfg,'head_loss_mode':'centered_all_seen'},out,identity,clock)
    if preserving:p.teacher=copy.deepcopy(model).eval().requires_grad_(False)
    p.s=copy.deepcopy(source['state']);restore_rng(source['rng']);return p

with sdpa_kernel(SDPBackend.MATH):
    probe=root/f'runs/seed{seed}/preflight';probe.mkdir(parents=True,exist_ok=False)
    expected={}
    for preserving in [False,True]:
        p=make(preserving,probe);dl=p.loader(raw[0],p.ids('train',1,0),cfg['batch'],True)
        x,y=next(iter(dl));x=p.train_tf(x.cuda());y=y.cuda();p.model.zero_grad(set_to_none=True);p.backward_loss(x,y)
        expected[preserving]={'x':x.cpu().clone(),'y':y.cpu().clone(),'B':[v.grad.cpu().clone() for v in p.model.b_parameters()],'head':p.model.rows[-1].grad.cpu().clone()}
        assert all(v.grad is None for v in p.model.rows[:-1])
        if preserving:initial_zero=p.s['preservation_training']['mse_sum']==0.
        del p,dl,x,y;gc.collect();torch.cuda.empty_cache()
    a,b=expected[False],expected[True]
    checks={'initial_preservation_loss_zero':initial_zero,'input_exact':torch.equal(a['x'],b['x']),'labels_exact':torch.equal(a['y'],b['y']),
            'direct_B_gradients_exact':all(torch.equal(x,y) for x,y in zip(a['B'],b['B'])),
            'head_gradient_exact':torch.equal(a['head'],b['head'])}
    p=make(True,probe);x=b['x'].cuda();y=b['y'].cuda()
    with torch.no_grad():
        v=p.model.b_parameters()[1]
        v.add_(.001*torch.sin(torch.arange(v.numel(),device=v.device,dtype=v.dtype).reshape_as(v)))
    p.model.zero_grad(set_to_none=True);separated_backward(p.model,x,y,1,cfg['cpt'],'centered_all_seen')
    ref_B=[v.grad.detach().clone() for v in p.model.b_parameters()];ref_head=p.model.rows[-1].grad.detach().clone()
    p.model.zero_grad(set_to_none=True);p.backward_loss(x,y)
    checks.update(perturbed_preservation_mse_positive=p.s['preservation_training']['mse_sum']>0,
        perturbed_B_gradient_changed=any(not torch.equal(v.grad,g) for v,g in zip(p.model.b_parameters(),ref_B)),
        perturbed_head_gradient_exact=torch.equal(p.model.rows[-1].grad,ref_head),
        teacher_has_no_gradients=all(v.grad is None for v in p.teacher.parameters()))
    del p,x,y,ref_B,ref_head,v;gc.collect();torch.cuda.empty_cache()
    write(root/f'reports/FIRST_BATCH_seed{seed}.json',{'passed':all(checks.values()),'checks':checks,'optimizer_steps':0,
          'backward_images':4*len(a['y']),'extra_teacher_forward_images':2*len(a['y']),'reused_control_qualified':True,'original_failure_retained':True})
    if not all(checks.values()):state['status']='qualification_failed';record();raise RuntimeError(checks)
    out=root/f'runs/seed{seed}/transport_head';out.mkdir(parents=True,exist_ok=False)
    p=make(True,out);p.expected_first=b
    p.expected_first_round=torch.load(control/'round_t1_r1.pt',map_location='cpu',weights_only=False)
    p.save()
    nr=len(p.s['training_records']);ns=len(p.s['stats_records']);nc=len(p.s['certificates']);before_comm=dict(p.s['communication'])
    state.update(status='running',started=time.time());record();p.run();new=p.s['training_records'][nr:]
    assert len(new)==50 and len({(v['task'],v['round'],v['client']) for v in new})==50
    assert sum(v['visited'] for v in new)==125000 and p.first_batch_verified
    assert len(p.s['certificates'])-nc==5 and all(v['deployed_fp32']['passed'] for v in p.s['certificates'][nc:])
    assert p.s['merges']==[0,1] and p.s['history'].committed=={0,1}
    assert all(not torch.count_nonzero(v) for v in p.model.b_parameters())
    p.restore();p.s['task']=1;final=p.evaluate()
    assert final['confusion']==p.s['stages'][-1]['confusion'] and final['centered']==p.s['stages'][-1]['centered']
    assert all(torch.equal(v.cpu(),source['model'][k]) for k,v in p.teacher.state_dict().items())
    assert torch.equal(p.model.rows[0].cpu(),source['model']['rows.0'])
    assert all(v.grad is None for v in p.teacher.parameters())
    assert p.s['preservation_training']['visits']==125000 and p.s['preservation_training']['positive_mse_batches']>0
    changed=sum(not torch.equal(v.cpu(),source['model'][k]) for k,v in p.model.state_dict().items() if k.startswith('backbone.'))
    assert changed>0
    assert p.s['transport_first_round_pair']['passed'] and not p.s['transport_endpoints']
    newstats=p.s['stats_records'][ns:]
    assert sum(v['samples'] for v in newstats)==50000
    assert {v['kind'] for v in newstats}=={'paired_local_endpoint','paired_deployed_global'}
    assert all(v['paired_samples']==5000 and v['paired_encoder_forward_images']==10000 for v in p.s['certificates'][nc:])
    assert all(v['paired_feature_squared_drift_mean']>0 for v in p.s['certificates'][nc:])
    cm=[[v['all_seen_accuracy'] for v in s['centered']['per_task']] for s in p.s['stages']]
    q={'qualified':True,'head_loss_mode':'centered_all_seen','new_training_visits':125000,'new_client_rounds':50,
       'new_optimizer_steps':sum(v['steps'] for v in new),'new_B_optimizer_steps':sum(v['steps'] for v in new),'new_statistics_visits':sum(v['samples'] for v in p.s['stats_records'][ns:]),
       'new_communication_bytes':{k:v-before_comm.get(k,0) for k,v in p.s['communication'].items()},'new_certificates_passed':5,
       'actual_first_batch_matches_preflight':True,'final_checkpoint_reload_confusion_equal':True,
       'centered_matrix':cm,'centered_forgetting':cm[0][0]-cm[1][0],
       'teacher_state_exact':True,'teacher_gradients_none':True,'old_head_exact':True,
       'changed_backbone_tensors':changed,'preservation_training':p.s['preservation_training'],
       'first_round_B_and_raw_head_exact_pair':p.s['transport_first_round_pair'],
       'paired_samples':25000,'paired_statistics_encoder_forward_images':50000,
       'paired_feature_squared_drift_mean_by_round':[v['paired_feature_squared_drift_mean'] for v in p.s['certificates'][nc:]],
       'raw_response_MSE_by_round':[v['raw_response_MSE'] for v in p.s['certificates'][nc:]],
       'raw_FedAvg_response_MSE_by_round':[v['raw_FedAvg_response_MSE'] for v in p.s['certificates'][nc:]],
       'head_aggregation_frame':'paired_deployed_global',
       'old_response_coefficient':1.,'new_teacher_forward_images':125000,'preflight_extra_teacher_forward_images':2*len(b['y']),
       'clock':clock,'reused_control':lane['control'],'reused_control_new_optimizer_steps':0,
       'new_common_RRS_images':0,'new_T1_replay_images':0,'preflight_backward_images':4*len(b['y'])}
    assert (source_path.stat().st_size,source_path.stat().st_mtime_ns)==source_before
    write(out/'QUALIFICATION.json',q);write(out/'COMPLETE.json',{'completed':time.time(),'qualified':True})
    state.update(status='completed',ended=time.time(),source_signature_unchanged=True);record()
