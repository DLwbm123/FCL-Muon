"""Post-training class-incremental pair accounting for three tasks."""
import numpy as np


def compare(labels, prediction, baseline_prediction, cpt, t1_accuracy, t2_accuracies):
    y, pred, base = (np.asarray(v) for v in (labels, prediction, baseline_prediction))
    if y.ndim != 1 or y.shape != pred.shape or y.shape != base.shape or cpt <= 0 or len(t2_accuracies) != 2:
        raise ValueError('invalid paired predictions')
    masks = [(y >= i*cpt) & (y < (i+1)*cpt) for i in range(3)]
    if sum(int(m.sum()) for m in masks) != len(y) or any(not m.any() for m in masks):
        raise ValueError('invalid task label coverage')
    if any(not np.isfinite(v) or not 0 <= v <= 1 for v in [t1_accuracy, *t2_accuracies]):
        raise ValueError('invalid previous task accuracies')
    good, was_good = pred == y, base == y
    per_task = [float(good[mask].mean()) for mask in masks]
    forgetting = sum(max(([t1_accuracy] if i == 0 else []) + [t2_accuracies[i], per_task[i]]) - per_task[i]
                     for i in range(2)) / 2
    old = masks[0] | masks[1]
    return {'raw_accuracy': float(good.mean()), 'old_accuracy': float(good[old].mean()),
            'new_accuracy': per_task[2], 'forgetting': float(forgetting),
            'corrected_vs_zero': int((good & ~was_good).sum()),
            'broken_vs_zero': int((~good & was_good).sum()),
            'old_correct': int(good[old].sum()), 'new_correct': int(good[masks[2]].sum())}


def conditions(comparisons):
    base, direct, guarded = (comparisons[x] for x in ('zero', 'projected', 'guarded'))
    raw_net = guarded['corrected_vs_zero'] - guarded['broken_vs_zero']
    return {'raw_gain_at_least_6_of_3000': bool(raw_net >= 6),
            'old_accuracy_not_lower': bool(guarded['old_correct'] >= base['old_correct']),
            'new_accuracy_no_more_than_10_of_1000_lower': bool(guarded['new_correct'] >= base['new_correct'] - 10),
            'forgetting_not_higher': bool(guarded['forgetting'] <= base['forgetting'] + 1e-12),
            'raw_not_lower_than_unconditional_projection': bool(guarded['raw_accuracy'] + 1e-12 >= direct['raw_accuracy'])}
