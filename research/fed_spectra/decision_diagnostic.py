"""Paired aggregate decision changes; no sample-level outputs or training."""
import torch
from fcl.checkpoint import rng_state,restore_rng

METRICS=('all_seen_ce','current_ce','competition_ce','true_minus_max_old')
GROUPS=('all_seen_correct','current_correct_stolen','current_wrong')


def decisions(z,y,old):
    current=z[:,old:].argmax(-1)+old
    pred=z.argmax(-1)
    correct=pred==y
    stolen=(current==y)&(pred<old)
    group=torch.where(correct,0,torch.where(stolen,1,2))
    assert torch.all((group!=2)|(current!=y))
    all_ce=torch.logsumexp(z,1)-z.gather(1,y[:,None]).squeeze(1)
    cur_ce=torch.logsumexp(z[:,old:],1)-z.gather(1,y[:,None]).squeeze(1)
    metrics=torch.stack((all_ce,cur_ce,all_ce-cur_ce,z.gather(1,y[:,None]).squeeze(1)-z[:,:old].max(1).values),1)
    return pred,current,group,metrics


def pair_statistics(a,b,y,old):
    ap,ac,ag,av=decisions(a,y,old);bp,bc,bg,bv=decisions(b,y,old)
    transition=torch.bincount(ag*3+bg,minlength=9).reshape(3,3)
    masks={'prediction_changed':ap!=bp,'correct_to_wrong':(ap==y)&(bp!=y),
      'wrong_to_correct':(ap!=y)&(bp==y),'wrong_to_different_wrong':(ap!=y)&(bp!=y)&(ap!=bp),
      'current_prediction_changed':ac!=bc,'old_winner_changed':(ap<old)!=(bp<old)}
    return {'transition_counts':transition.cpu().tolist(),'counts':{k:int(v.sum()) for k,v in masks.items()},
      'baseline_metric_sums':av.double().sum(0).cpu().tolist(),
      'candidate_metric_sums':bv.double().sum(0).cpu().tolist(),
      'group_delta_sums':torch.stack([(bv-av)[ag==g].double().sum(0) for g in range(3)]).cpu().tolist()}


def measure_decision_pair(p,probes,baseline,candidate):
    state=rng_state();modes={m:m.training for m in p.model.modules()}
    saved=[v.detach().clone() for v in p.model.b_parameters()]
    saved_buffers={k:v.detach().clone() for k,v in p.model.named_buffers()}
    p.model.eval();total=sum(len(p.ids('train',p.s['task'],k)) for k in range(p.cfg['clients']))
    rows=[];images=0
    def deploy(values):
        for b,v in zip(p.model.b_parameters(),values):b.copy_(v)
    try:
        with torch.no_grad():
            for k,ids in enumerate(probes):
                if not ids:continue
                sums=None
                for x,y in p.loader(p.raw[0],ids,p.cfg['batch']):
                    p.budget();p.check_labels(y);x=p.test_tf(x.to(p.device));y=y.to(p.device)
                    deploy(baseline);a=p.model(x)
                    deploy(candidate);b=p.model(x)
                    v=pair_statistics(a,b,y,p.s['task']*p.cfg['cpt'])
                    if sums is None:sums=v
                    else:
                        for key in ('transition_counts','group_delta_sums'):
                            sums[key]=[[aa+bb for aa,bb in zip(ar,br)] for ar,br in zip(sums[key],v[key])]
                        for key in ('baseline_metric_sums','candidate_metric_sums'):
                            sums[key]=[aa+bb for aa,bb in zip(sums[key],v[key])]
                        for key in sums['counts']:sums['counts'][key]+=v['counts'][key]
                    images+=len(y)
                assert sum(map(sum,sums['transition_counts']))==len(ids)
                for j in range(4):
                    assert abs(sum(v[j] for v in sums['group_delta_sums'])-(sums['candidate_metric_sums'][j]-sums['baseline_metric_sums'][j]))<=2e-6*len(ids)
                rows.append(dict(client=k,images=len(ids),weight=len(p.ids('train',p.s['task'],k))/total,**sums))
            coverage=sum(r['weight'] for r in rows)
            transition=[[sum(r['weight']*r['transition_counts'][i][j]/r['images']/coverage for r in rows) for j in range(3)] for i in range(3)]
            groups=[]
            for i,name in enumerate(GROUPS):
                weight=sum(transition[i]);contribution={key:sum(r['weight']*r['group_delta_sums'][i][j]/r['images']/coverage for r in rows) for j,key in enumerate(METRICS)}
                groups.append({'baseline_group':name,'weight':weight,'metric_delta_contribution':contribution,'conditional_mean_delta':{k:v/weight if weight else None for k,v in contribution.items()}})
            result={'clients':rows,'coverage':coverage,'transition_fraction':transition,'groups':groups,
              'counts':{key:sum(r['counts'][key] for r in rows) for key in rows[0]['counts']},
              'weighted_fractions':{key:sum(r['weight']*r['counts'][key]/r['images']/coverage for r in rows) for key in rows[0]['counts']},
              'baseline_metrics':{key:sum(r['weight']*r['baseline_metric_sums'][j]/r['images']/coverage for r in rows) for j,key in enumerate(METRICS)},
              'candidate_metrics':{key:sum(r['weight']*r['candidate_metric_sums'][j]/r['images']/coverage for r in rows) for j,key in enumerate(METRICS)},
              'images':images,'student_forward_images':2*images,'student_backward_images':0,'teacher_forward_images':0}
            net=sum(transition[i][0] for i in range(3))-sum(transition[0])
            assert abs(net-result['weighted_fractions']['wrong_to_correct']+result['weighted_fractions']['correct_to_wrong'])<1e-12
            return result
    finally:
        with torch.no_grad():
            deploy(saved)
            assert all(torch.equal(v,saved_buffers[k]) for k,v in p.model.named_buffers())
        for m,mode in modes.items():m.training=mode
        restore_rng(state)
