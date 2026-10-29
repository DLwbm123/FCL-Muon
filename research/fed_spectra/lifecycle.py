"""Milestone qualification on the original uninterrupted ten-task trajectory."""
import copy
import json
import os
import time
import numpy as np
import torch
from research.basic_repair.fcl_basic_repair_tools import center_task_logits


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def qualify(p,n):
    if p.s['task']!=n or p.s['phase'] not in ('task_start','complete'):
        raise ValueError('Qualification requires an exact committed task boundary')
    cfg=p.cfg;s=p.s;records=s['training_records'];rounds=cfg['rounds']
    expected={(t,r,k) for t in range(n) for r in range(rounds) for k in range(cfg['clients']) if p.ids('train',t,k)}
    assert {(v['task'],v['round'],v['client']) for v in records}==expected and len(records)==len(expected)
    visits=sum(len(p.ids('train',t))*rounds*cfg['epochs'] for t in range(n))
    assert sum(v['visited'] for v in records)==visits
    assert s['merges']==list(range(n)) and s['history'].committed==set(range(n))
    assert len(s['stages'])==n and len(p.model.rows)==n
    assert len(s['certificates'])==n*rounds and all(v['deployed_fp32']['passed'] for v in s['certificates'])
    assert all(a.merged for a in p.model.adapters.values()) and all(not torch.count_nonzero(b) for b in p.model.b_parameters())
    assert s['teacher_state'] is None and not s['spectra_endpoints']
    if p.spectral:
        assert len(s['spectra_geometry'])==len(s['spectra_rounds'])==n*rounds
        assert len(s['spectra_commits'])==n and len(s['spectra_probes'])==n*(rounds+1)
    if cfg['method']=='actual_update_global':
        assert len(s['actual_update_rounds'])==n*rounds and not s['actual_update_endpoints']
        assert not s['spectra_geometry'] and not s['spectra_probes']
    private=p.out/f'task{n}_private_logits.npz'
    with np.load(private) as f:before={k:f[k].copy() for k in f.files}
    stages=copy.deepcopy(s['stages'])
    # Restore the actual checkpoint; keep the original stage outputs separately
    # even if a validation assertion fails after writing the reload prediction.
    original=p.out/f'task{n}_original_logits.npz'
    if original.exists():raise ValueError('Unfinished earlier qualification')
    private.rename(original)
    try:
        p.restore();p.s['task']=n-1;final=p.evaluate()
        with np.load(private) as f:after={k:f[k].copy() for k in f.files}
        private.rename(p.out/f'task{n}_reload_private_logits.npz')
    finally:
        p.s['task']=n
        original.replace(private)
    assert np.array_equal(before['indices'],after['indices']) and np.array_equal(before['labels'],after['labels'])
    images=sum(len(p.ids('test',t)) for t in range(n))
    assert len(after['labels'])==images
    raw_equal=np.array_equal(before['logits'].argmax(1),after['logits'].argmax(1))
    centered_equal=np.array_equal(center_task_logits(before['logits'],cfg['cpt']).argmax(1),
                                  center_task_logits(after['logits'],cfg['cpt']).argmax(1))
    assert raw_equal and centered_equal and final['confusion']==stages[-1]['confusion'] and final['centered']==stages[-1]['centered']
    s=p.s
    q={'qualified':True,'method':cfg['method'],'tasks':n,'primary_readout':'raw','protocol_version':cfg['protocol_version'],
       'clock':p.clock,'config':cfg,'stages':stages,'client_rounds':len(records),'training_visits':visits,
       'new_training_visits':visits,'stats_visits':sum(v['samples'] for v in s['stats_records']),
       'merges':s['merges'],'history_committed':sorted(s['history'].committed),'certificates_passed':len(s['certificates']),
       'spectra_geometry':s['spectra_geometry'],'spectra_rounds':s['spectra_rounds'],
       'spectra_commits':s['spectra_commits'],'spectra_probes':s['spectra_probes'],
       'training_records':records,'logical_communication_bytes':s['communication'],
       'anchor_forward_images':s['anchor_forward_images'],'anchor_forward_calls':s['anchor_forward_calls'],
       'extra_probe_forward_images':sum(v['forward_images'] for v in s['spectra_probes']),
       'extra_probe_backward_images':sum(v['backward_images'] for v in s['spectra_probes']),
       'extra_probe_forward_calls':sum(v['forward_calls'] for v in s['spectra_probes']),
       'extra_probe_backward_calls':sum(v['backward_calls'] for v in s['spectra_probes']),
       'reload_images':images,'reload_raw_predictions_equal':raw_equal,'reload_centered_predictions_equal':centered_equal,
       'maximum_reload_logit_difference':float(np.abs(before['logits']-after['logits']).max()),
       'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated() if p.device.type=='cuda' else 0,
       'elapsed_seconds':time.time()-p.clock['started'],'checkpoint_bytes':(p.out/'latest.pt').stat().st_size,
       'limitations':['single development seed','current-input Fisher/KL is not an old-task risk guarantee',
                     'Full cumulative task update may exceed active rank','per-client covariance transport does not guarantee aggregate Gram',
                     'original LoRM is a system comparison; controlled Adam is the matched comparison']}
    path=p.out/f'QUALIFICATION_T{n}.json'
    if cfg['method']=='actual_update_global':
        q['actual_update_rounds']=s['actual_update_rounds']
        q['limitations']=['single development seed; no significance claim',
            'current-input KL is not an old-task risk guarantee',
            'shape correction does not guarantee classification improvement',
            'active rank per task is fixed; task-boundary merges are shared with control',
            'original LoRM is a system comparison; eta-zero arm is the matched comparison']
    if cfg.get('training_gate'):
        gates=s['training_gate_records'];assert len(gates)==n*rounds
        q['training_gate_records']=gates
        q['gate_student_forward_images']=sum(v['student_forward_images'] for v in gates)
        q['gate_teacher_forward_images']=sum(v['teacher_forward_images'] for v in gates)
        q['extra_probe_forward_images']+=q['gate_student_forward_images']+q['gate_teacher_forward_images']
        q['extra_probe_forward_calls']+=sum(v['forward_calls'] for v in gates)
        q['gate_accepted_rounds']=sum(v['accepted'] for v in gates)
        q['limitations'].append('Training-internal loss gate is not a held-out or old-task risk guarantee; extra forward and logical communication costs are reported.')
    if path.exists():raise ValueError('Do not overwrite a qualified milestone')
    write(path,q)
    checkpoint=p.out/f'task{n}.pt'
    if checkpoint.exists():raise ValueError('Milestone checkpoint exists')
    os.link(p.out/'latest.pt',checkpoint)
    if n==cfg['tasks']:
        write(p.out/'QUALIFICATION.json',q)
        write(p.out/'COMPLETE.json',{'qualified':True,'tasks':n,'completed':time.time()})
    return q
