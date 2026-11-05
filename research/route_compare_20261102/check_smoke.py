"""CPU-only real checkpoint model/optimizer/clock pre-freeze inspection."""
import json, os
from pathlib import Path
import timm, torch
from fcl.model import MuLoRAFL


def main():
    root=Path(os.environ['PROJECT_ROOT']);job=json.loads((root/'reports/job.json').read_text())
    out={}
    for optimizer in ('adamw','muon'):
        x=torch.load(root/f'runs/smoke_{optimizer}/latest.pt',map_location='cpu',weights_only=False)
        s=x['state'];cfg=x['config']
        if x['clock']!={k:job[k] for k in ('started','deadline')}:
            raise ValueError('smoke clock mismatch')
        if (s['task'],s['round'],s['phase'])!=(0,5,'ready_evaluation') or len(s['training_records'])!=50:
            raise ValueError('smoke boundary mismatch')
        if sum(v['visited'] for v in s['training_records'])!=125000 or x['head_count']!=1:
            raise ValueError('smoke counts mismatch')
        if cfg['route']['optimizer']!=optimizer or cfg['route']['mode']!='prefix':
            raise ValueError('smoke optimizer identity mismatch')
        if not x['rng']['cuda'] or s['history'].committed or len(s['pending'])!=12:
            raise ValueError('smoke history or RNG missing')
        net=MuLoRAFL(timm.create_model(cfg['backbone'],pretrained=False,num_classes=0),100,cfg['rank'],cfg['cosine_scale'])
        net.add_head(cfg['cpt']);net.load_state_dict(x['model'],strict=True)
        if any(x['merged_flags'].values()):raise ValueError('smoke should remain unmerged')
        if optimizer=='muon' and not any(torch.count_nonzero(v) for group in s['client_states'].values() for v in group['m']):
            raise ValueError('Muon client momentum missing')
        out[optimizer]={'strict_cpu_model_load':True,'t1_visits':125000,'client_rounds':50,
                        'cuda_rng_present':True,'phase':'ready_evaluation'}
        del net,x
    print(json.dumps({'status':'passed','models':out,'pixel_reads':0,'test_scores':0}))

if __name__=='__main__':main()
