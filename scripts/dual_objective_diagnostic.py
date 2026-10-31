"""Draft fixed-round dual-objective diagnostic. Not a registered experiment."""
import copy
import json
import os
import time
from pathlib import Path

import timm
import torch
from fcl.checkpoint import restore_rng, rng_state
from fcl.data import loader, transforms
from fcl.math import mulora_ssb
from fcl.model import MuLoRAFL
from research.head_regmean.pilot import Pilot
from research.fed_spectra.heldout_round import GuardedLoader, split_clients
from research.continual_muon.current_cifar import CurrentCIFAR
from research.continual_muon.current_task_guard import CurrentTaskGuard
from research.continual_muon.dual_fit_gradients import gradients
from research.continual_muon.dual_objective_server import guarded_candidate
from research.continual_muon.server_candidate import candidate
from scripts.server_response_diagnostic import evaluate
from scripts.real_boundary_worker import same_rng, write


def main():
    root, source, endpoint = [Path(os.environ[k]) for k in ('PROJECT_ROOT', 'SOURCE_ROOT', 'ENDPOINT_ROOT')]
    report = root / 'reports'
    job = json.loads((report / 'job.json').read_text())
    clock = {k: job[k] for k in ('started', 'deadline')}
    if not clock['started'] <= time.time() < clock['deadline']:
        raise TimeoutError('original deadline exhausted')
    with (report / 'WORKER_CLAIM.json').open('x') as f:
        json.dump({'clock': clock, 'started': time.time()}, f)
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.benchmark = False;torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False;torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    x = torch.load(endpoint / 'runs/diagnostic/latest.pt', map_location='cpu', weights_only=False)
    s, cfg = x['state'], copy.deepcopy(x['config'])
    expected = {'seed': 0, 'dataset': 'seq-cifar100', 'alpha': .05, 'clients': 10,
                'rounds': 5, 'epochs': 5, 'batch': 16, 'rank': 32, 'lr': .0002,
                'aggregation': 'regmean', 'selection_mode': 'prefix', 'retain_round_checkpoints': False}
    if any(cfg.get(k) != v for k, v in expected.items()):raise ValueError('source config mismatch')
    if (x['schema'] != 'cosine-head-regmean-pilot-1' or x['head_count'] != 3 or
            (s['task'], s['round'], s['phase']) != (2, 0, 'local_done') or
            s['history'].committed != {0, 1} or x['clock'] == clock or
            any(len(s[k]) != n for k,n in [('agg_B',24),('heads',10),('grams',10),('sizes',10),('client_states',10)])):
        raise ValueError('source endpoint mismatch')
    manifest = json.loads((source / cfg['manifest']).read_text())
    weights_info = json.loads((source / 'weights.json').read_text())
    if (manifest['identity'] != {k:cfg[k] for k in ('dataset','seed','alpha','clients')} or
            x['identity']['manifest'] != cfg['manifest'] or
            x['identity']['weight_identity'] != weights_info['sha256'] or
            x['identity']['source_revision'] != cfg['source_revision']):
        raise ValueError('source identity mismatch')
    original = manifest['partitions']['train'];fit, heldout = split_clients(original[2], 0)
    if (sum(map(len,fit)),sum(map(len,heldout))) != (4005,995):raise ValueError('split mismatch')
    records = [v for v in s['training_records'] if v['task'] == 2 and v['round'] == 0]
    if (len(records) != 10 or sum(v['visited'] for v in records) != 20025 or
            any(v['N'] != len(fit[v['client']]) for v in records)):
        raise ValueError('source training receipt mismatch')
    pairs = [(sorted(a)[:32], sorted(b)[32:64]) if len(b)>32 else ([],[]) for a,b in zip(fit,heldout)]
    if (sum(len(a) for a,b in pairs), sum(len(b) for a,b in pairs),sum(bool(b) for a,b in pairs)) != (224,224,7):
        raise ValueError('new probe split mismatch')
    indices = {t:[i for g in original[t] for i in g] for t in (1,2)}
    groups = copy.deepcopy(original);groups[2] = fit
    train_manifest = {'identity':manifest['identity'],'partitions':{'train':groups},
                      'labels':{'train':manifest['labels']['train']}}
    raw = CurrentCIFAR(source / 'data/cifar-100-python/train', train_manifest['labels']['train'], indices)
    taskguard = CurrentTaskGuard(loader, raw, indices);raw.enter_t3();taskguard.enter_t3()
    guard = GuardedLoader(taskguard,raw,[i for g in fit for i in g],[i for _,g in pairs for i in g])
    try:
        model = MuLoRAFL(timm.create_model(cfg['backbone'],pretrained=False,num_classes=0),100,cfg['rank'],cfg['cosine_scale']).cuda()
        for _ in range(3):model.add_head(cfg['cpt'])
        model.load_state_dict(x['model'],strict=True)
        for name,a in model.adapters.items():
            a.merged = x['merged_flags'][name]
            if a.merged:raise ValueError('source adapter unexpectedly merged')
        train_tf,stat_tf = transforms()
        pilot = Pilot(model,[raw,None],train_manifest,guard,train_tf.cuda(),stat_tf.cuda(),cfg,
                      root / 'runs/diagnostic',x['identity'],clock)
        pilot.s = copy.deepcopy(s);restore_rng(x['rng'])
        if len(x['rng']['cuda']) != 1 or not torch.equal(torch.cuda.get_rng_state(),x['rng']['cuda'][0]):
            raise ValueError('source RNG restoration failed')
        anchor_rng = rng_state();i = torch.arange(768,dtype=torch.float32)[:,None]+1
        j = torch.arange(32,dtype=torch.float32)[None,:]+1
        q = torch.linalg.qr(torch.sin(i*j*.013),mode='reduced').Q
        err = float(torch.linalg.vector_norm(mulora_ssb(q)-mulora_ssb(q.cuda()).cpu()))
        if not torch.isfinite(torch.tensor(err)) or err > 1e-4 or not same_rng(anchor_rng,rng_state()):
            raise ValueError('SSB preflight failed')
        write(report/'REAL_PREFLIGHT.json',{'passed':True,'cpu_gpu_ssb_l2':err,'image_accesses':0})
        origin = [v.detach().cpu().clone() for v in model.b_parameters()]
        if len(origin)!=24 or any(torch.count_nonzero(v) for v in origin):raise ValueError('source round start B mismatch')
        pilot.fuse();baseline = [v.detach().cpu().clone() for v in model.b_parameters()]
        if (pilot.s['round'],pilot.s['phase']) != (1,'round_start') or guard.validation_access_images:
            raise ValueError('round restore or heldout isolation failure')
        ga,gc,fit_receipt = gradients(pilot,pairs,0)
        weights = [len(g)/sum(map(len,fit)) for g in fit]
        u = [b-a for a,b in zip(origin,baseline)];hs=[];qs=[];seed_receipts=[]
        for j, update in enumerate(u):
            pilot.budget();name = list(model.adapters)[j//2]
            a = model.adapters[name].A.detach().cpu().double()
            h = a @ pilot.s['history'].raw[name].double() @ a.T
            ms = [pilot.s['client_states'][k]['m'][j] for k in range(cfg['clients'])]
            adjusted, receipt = candidate(update,ms,weights,h,angle=.1,damping=1.,tolerance=1e-10)
            hs.append(h);qs.append(adjusted.double()-update.double());seed_receipts.append(receipt)
        deployed, receipt = guarded_candidate(u,qs,ga,gc,hs,angle=.1,history_tolerance=1e-10)
        deployed = [a+b for a,b in zip(origin,deployed)]
        if guard.validation_access_images or guard.training_validation_accesses:
            raise ValueError('confirmation accessed before candidate frozen')
        write(report/'CANDIDATE_FROZEN.json',{'accepted':receipt['accepted'],'reason':receipt['reason'],
              'heldout_reads_at_freeze':0,'time':time.time()})
        matrices = model.b_parameters();heads = [v.detach().clone() for v in model.rows]
        buffers = {k:v.detach().clone() for k,v in model.named_buffers()}
        old_grads = [None if v.grad is None else v.grad.detach().clone() for v in model.parameters()]
        saved_rng = rng_state();metrics={}
        for part,name in enumerate(('fit','confirmation')):
            if part:guard.allow_validation = True
            before,bp = evaluate(pilot,pairs,part)
            with torch.no_grad():
                for target,value in zip(matrices,deployed):target.copy_(value)
            after,ap = evaluate(pilot,pairs,part)
            with torch.no_grad():
                for target,value in zip(matrices,baseline):target.copy_(value)
            if len(bp)!=len(ap) or any(a[0]!=b[0] for a,b in zip(bp,ap)):raise ValueError('pairing mismatch')
            flips = {'correct_to_wrong':0,'wrong_to_correct':0,'wrong_to_different_wrong':0}
            for (y,a),(_,b) in zip(bp,ap):
                if a==y and b!=y:flips['correct_to_wrong']+=1
                elif a!=y and b==y:flips['wrong_to_correct']+=1
                elif a!=y and b!=y and a!=b:flips['wrong_to_different_wrong']+=1
            metrics[name]={'baseline':before,'candidate':after,'paired_flips':flips,
                           'all_loss_difference':after['all']-before['all'],
                           'current_loss_difference':after['current']-before['current'],
                           'accuracy_difference':after['accuracy']-before['accuracy']}
        va,vc,val_receipt = gradients(pilot,pairs,1)
        delta = [b.double()-a.double() for a,b in zip(baseline,deployed)]
        dot = lambda gs:sum(float((g*d).sum()) for g,d in zip(gs,delta))
        directional = {'fit_all':dot(ga),'fit_current':dot(gc),'confirmation_all':dot(va),'confirmation_current':dot(vc)}
        if (guard.training_access_images != 672 or guard.validation_access_images != 672 or
                guard.training_validation_accesses or raw.reads != {1:0,2:1344} or raw.reads != taskguard.visits):
            raise ValueError('data accounting mismatch')
        if (not same_rng(saved_rng,rng_state()) or any(not torch.equal(a,b) for a,b in zip(heads,model.rows)) or
                any(not torch.equal(buffers[k],v) for k,v in model.named_buffers()) or
                any(not torch.equal(a.detach().cpu(),b) for a,b in zip(matrices,baseline))):
            raise ValueError('model or RNG not restored')
        for p,old in zip(model.parameters(),old_grads):
            if (old is None)!=(p.grad is None) or (old is not None and not torch.equal(old,p.grad)):
                raise ValueError('existing gradient not restored')
        checks={'candidate_accepted':bool(receipt['accepted'])}
        for name in ('fit','confirmation'):
            m=metrics[name];checks[name+'_coverage']=bool(m['baseline']['weight_coverage']>=.95)
            checks[name+'_all_improved']=bool(m['all_loss_difference'] < -1e-6)
            checks[name+'_current_not_worse']=bool(m['current_loss_difference'] <= 1e-6)
        checks['confirmation_accuracy_not_lower']=bool(metrics['confirmation']['accuracy_difference'] >= -1e-12)
        write(report/'RESULT.json',{'status':'completed','local_conditions_passed':all(checks.values()),
            'checks':checks,'candidate_receipt':receipt,'metrics':metrics,'directional':directional,
            'fit_gradient_receipt':fit_receipt,'confirmation_gradient_receipt':val_receipt,
            'student_forward_images':1344,'student_backward_images':896,'old_train_images':0,'test_images':0,
            'new_training_images':0,'fit_reads':672,'confirmation_reads':672,'heldout_reads_at_freeze':0,
            'theoretical_momentum_upload_bytes':23592960,'theoretical_gradient_upload_bytes':66060288,
            'theoretical_candidate_broadcast_7_clients_bytes':16515072,
            'theoretical_candidate_broadcast_10_clients_bytes':23592960,'network_bytes_measured':None,
            'seed_reason_counts':{r:sum(v['reason']==r for v in seed_receipts) for r in sorted({v['reason'] for v in seed_receipts})},
            'elapsed_seconds':time.time()-clock['started'],'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),
            'limitations':['single round and seed; not full FCL or old-class safety',
                           'confirmation unused in source round but possibly seen in other research or pretraining',
                           'client final momenta are not common-anchor gradients or full trajectory bias']})
    finally:
        raw.close()


if __name__ == '__main__':main()
