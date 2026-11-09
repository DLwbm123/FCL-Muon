"""Train-only pseudo-old-class extrapolation at initial and completed-C0 states.

The late reference has already learned these classes: it is a stress test of
coordinate transfer, not an unseen-class or chronological evaluation result.
No diagnostic outcomes tune the frozen parameters or select a test seed.
"""
import copy,gc,json,os
from pathlib import Path
import torch
from torch.nn import functional as F
from research.head_regmean.pilot import observe
from .runtime import setup,build,validate
from .statistics import projection
from research.adamw_t10.storage import write


def main():
    setup();root,f=validate();clock={k:f[k] for k in ('started','deadline')};rows=[]
    os.environ['ARM']='unified'
    for state in ('initial','late_C0_merged'):
        p=build(clock);p.cfg=copy.deepcopy(p.cfg);p.cfg['route']['mode']='rrs'
        p.out=root/('src/runs/diagnostic_'+state);p.out.mkdir(exist_ok=False)
        if state!='initial':
            x=torch.load(f['diagnostic_late_checkpoint'],map_location='cpu',weights_only=False)
            base={k:v for k,v in x['model'].items() if not k.startswith('rows.')}
            missing,extra=p.model.load_state_dict(base,strict=False)
            if set(missing)!={'lda_weight','lda_bias','lda_classes'} or extra:raise ValueError('late source incompatible')
            for a in p.model.adapters.values():a.merged=True
            del x,base
        p.initialize_task()
        ids=[];labels=p.manifest['labels']['train']
        for c in range(10):ids.extend([i for i in p.ids('train',0) if labels[i]==c][:8])
        def features():
            out=[];ys=[]
            with observe(p.model):
                for x,y in p.loader(p.raw[0],ids,4):
                    p.budget();out.append(F.normalize(p.model.backbone.forward_features(p.test_tf(x.cuda()))[:,0],dim=1).cpu().double());ys.append(y)
            return torch.cat(out),torch.cat(ys)
        z0,y=features();fit=y<5
        params=[{'params':p.model.b_parameters(),'lr':.0002},{'params':[p.model.rows[-1]],'lr':.001}]
        opt=torch.optim.AdamW(params,weight_decay=0)
        fit_ids=[i for i in ids if labels[i]<5]
        for x,target in p.loader(p.raw[0],fit_ids,4):
            p.budget();opt.zero_grad();p.backward_loss(p.test_tf(x.cuda()),target.cuda());opt.step()
        zr,_=features();delta=zr-z0
        for name,q in (('identity',0),('lowrank64',64),('full_affine',768)):
            if not q:prediction=z0
            else:
                R=projection(768,q);h=z0[fit]@R;hm=h.mean(0);dm=delta[fit].mean(0)
                hc=h-hm;dc=delta[fit]-dm
                ridge=.1*(hc.T@hc).trace()/(len(h)*h.shape[1])+1e-6
                U=torch.linalg.solve(hc.T@hc+len(h)*ridge*torch.eye(h.shape[1]),hc.T@dc).T
                prediction=z0+(z0@R-hm)@U.T+dm
            true_means=torch.stack([zr[y==c].mean(0) for c in range(10)])
            predicted_means=torch.stack([prediction[y==c].mean(0) for c in range(10)])
            held=y>=5;old=z0[held];truth=zr[held];pred=prediction[held]
            true_sc=sum((zr[y==c]-true_means[c]).T@(zr[y==c]-true_means[c]) for c in range(5,10))
            pred_sc=sum((prediction[y==c]-predicted_means[c]).T@(prediction[y==c]-predicted_means[c]) for c in range(5,10))
            rows.append(dict(state=state,mapping=name,fit_classes=list(range(5)),pseudo_old_classes=list(range(5,10)),
                fit_images=int(fit.sum()),held_train_images=int(held.sum()),
                held_feature_MSE=float((pred-truth).square().mean()),
                held_mean_error=float((predicted_means[5:]-true_means[5:]).norm()),
                held_scatter_relative_error=float((pred_sc-true_sc).norm()/true_sc.norm()),
                held_nearest_mean_accuracy=float((torch.cdist(truth,predicted_means).argmin(1)==y[held]).double().mean())))
        del p;gc.collect();torch.cuda.empty_cache()
    write(root,root/'src/reports/diagnostic.json',dict(status='completed',train_only=True,rows=rows,
        limitation='late completed C0 already learned diagnostic classes; pseudo-old train-side stress test, not chronological old-data guarantee',
        decision='negative findings retained; no parameter tuning, no gate on intermediate test scores'))


if __name__=='__main__':main()
