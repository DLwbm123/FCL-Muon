"""Frozen synthetic boundary projection and Muon noncommutativity diagnostic."""
import json, math, signal, time
from pathlib import Path
import numpy as np

signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('hard limit')))
signal.alarm(60)
started=time.time()

def project(m, old, new):
    if not all(np.isfinite(x).all() for x in (m,old,new)):
        raise ValueError('non-finite')
    return m@old@new.T

def ssb(x):
    x=x.astype(np.float32).copy(); transpose=x.shape[0]>x.shape[1]
    if transpose: x=x.T
    x=x/(np.linalg.norm(x)+np.float32(1e-7))
    for _ in range(5):
        gram=x@x.T
        x=np.float32(3.4445)*x+(np.float32(-4.7750)*gram+np.float32(2.0315)*(gram@gram))@x
    return x.T if transpose else x

rows=[]
for seed in range(5):
    rng=np.random.default_rng(seed);m=rng.normal(size=(6,4));q,_=np.linalg.qr(rng.normal(size=(8,8)))
    old=q[:,:4].T;perp=q[:,4:].T; rot,_=np.linalg.qr(rng.normal(size=(4,4)))
    for theta in (0,math.pi/4,math.pi/2):
        new=math.cos(theta)*old+math.sin(theta)*perp
        assert np.allclose(new@new.T,np.eye(4),atol=1e-12)
        x=m@old; a=project(m,old,new);recon=a@new;res=np.linalg.norm(x-recon); zero=np.linalg.norm(x)
        assert np.allclose((x-recon)@new.T,0,atol=1e-12)
        assert np.isclose(np.linalg.norm(x)**2,np.linalg.norm(recon)**2+res**2,atol=1e-10)
        assert np.isclose(res/zero,math.sin(theta),atol=1e-12)
        assert np.allclose(project(m,old,rot@new),a@rot.T,atol=1e-12)
        rows.append({'seed':seed,'angle_radians':theta,'relative_residual':res/zero,'zero_baseline_residual_norm':zero})
try: project(np.array([[float('nan')]]),np.ones((1,1)),np.ones((1,1)))
except ValueError: pass
else: raise AssertionError('nonfinite must be rejected')
rng=np.random.default_rng(4401);m1=rng.normal(size=(6,4)).astype(np.float32);m2=rng.normal(size=(6,4)).astype(np.float32)
cases={'same':[m1,m1], 'noncollinear':[m1,m2], 'cancellation':[m1,-m1]};out={}
for label,(a,b) in cases.items():
    weights=(.5,.5) if label=='cancellation' else (.3,.7)
    phi_average=ssb(weights[0]*a+weights[1]*b);average_phi=weights[0]*ssb(a)+weights[1]*ssb(b)
    row={'weights':weights,'difference_norm':float(np.linalg.norm(phi_average-average_phi)),'phi_of_mean_norm':float(np.linalg.norm(phi_average)),'mean_of_phi_norm':float(np.linalg.norm(average_phi))}
    if label in ('same','cancellation'):assert row['difference_norm']<1e-5
    if label=='cancellation':assert max(row['phi_of_mean_norm'],row['mean_of_phi_norm'])<1e-7
    out[label]=row
result={'status':'completed','protocol':'BOUNDARY_MATH_PROTOCOL_ZH.txt','zero_training':True,'zero_image_access':True,'zero_test_access':True,'synthetic_results':rows,'noncommutativity_cases':out,'limitations':['Synthetic geometry only; not evidence of continual-learning efficacy.','Terminal state checkpoints contain no task-boundary client momentum; historical momentum replay is unsupported.','SSB of averaged terminal momenta versus mean SSB is not full-trajectory bias.'],'elapsed_seconds':time.time()-started}
p=Path('reports/muon_boundary_math_20261030/RESULT.json');p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'cases':out,'elapsed_seconds':result['elapsed_seconds']}));signal.alarm(0)
