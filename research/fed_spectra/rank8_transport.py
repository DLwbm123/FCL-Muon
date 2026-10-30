"""Fixed client-local low-rank wire representation, without rank selection."""
import torch


def encode_decode(gradient,rank=8):
    if gradient.ndim!=2 or rank<1 or rank>min(gradient.shape):raise ValueError('Invalid fixed rank')
    if gradient.dtype!=torch.float32 or not torch.isfinite(gradient).all():raise ValueError('Finite FP32 client gradient required')
    u,s,vh=torch.linalg.svd(gradient.double(),full_matrices=False)
    left=(u[:,:rank]*s[:rank]).float();right=vh[:rank].float()
    recovered=left.double()@right.double()
    energy=float(s.square().sum());kept=float(s[:rank].square().sum())
    return recovered,{'rank':rank,'full_bytes':gradient.numel()*4,'wire_bytes':(left.numel()+right.numel())*4,
        'retained_energy_fraction':kept/energy if energy else 1.,
        'relative_reconstruction_error':float((recovered-gradient.double()).norm())/max(energy**.5,1e-30)}


def conditions(rows):
    if sorted(r['seed'] for r in rows)!=[0,1,2]:raise ValueError('Three registered seeds required')
    def point(r):return r['compressed_candidate']['probes']['confirmation']
    return {'engineering':all(r['engineering_passed'] for r in rows),
        'coverage':all(point(r)['paired_decisions']['coverage']>=.95 for r in rows),
        'wire_le_27percent':all(r['wire_bytes']<=.27*r['full_bytes'] for r in rows),
        'selection_gate':all(r['selection_gate_passed'] for r in rows),
        'validation_retains_90percent_loss_gain':all(point(r)['combined_delta_vs_baseline']< -1e-6 and point(r)['combined_delta_vs_baseline']<=.9*r['full_candidate']['probes']['confirmation']['combined_delta_vs_baseline'] for r in rows),
        'validation_original_loss_safe':all(point(r)['original_objective_delta_vs_baseline']<=1e-6 for r in rows),
        'validation_accuracy_not_worse_than_full':all(point(r)['all_seen_accuracy_delta_vs_original']>=-1e-10 for r in rows),
        'mean_validation_accuracy_gain_positive':sum(point(r)['all_seen_accuracy_delta_vs_baseline'] for r in rows)/3>1e-10}
