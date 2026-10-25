import torch
from scripts.diagnose_checkpoint import summarize


def test_error_partition_with_old_new_tie_and_distinct_denominators():
    # 2 old and 4 new examples; two restricted-correct new examples are
    # stolen by old classes, including one exact tie (lower index wins).
    logits = torch.tensor([[4., 0., 1., 0.], [0., 1., 2., 0.],
                           [4., 0., 3., 1.], [0., 3., 1., 3.],
                           [0., 0., 1., 2.], [0., 0., 1., 3.]])
    labels = torch.tensor([0, 1, 2, 3, 2, 3])
    s = summarize(logits, labels, cpt=2)
    assert s['current_restricted_accuracy'] == .75
    assert s['current_all_seen_accuracy'] == .25
    assert s['current_competition_loss'] == .5
    assert s['current_correct_but_old_wins_count'] == 2
    assert s['current_to_old_rate'] == .5
    assert s['current_to_wrong_current_rate'] == .25
    assert s['old_to_current_rate'] == .5
    assert s['all_seen_accuracy'] == 2/6
    assert sum(map(sum, s['confusion_true_rows_pred_columns'])) == 6


def test_wrong_within_current_can_also_lose_to_old_without_double_counting():
    logits = torch.tensor([[3., 0., 0., 0.], [5., 0., 1., 2.]])
    s = summarize(logits, torch.tensor([0, 2]), cpt=2)
    assert s['current_to_old_rate'] == 1
    assert s['current_restricted_error'] == 1
    assert s['current_competition_loss'] == 0
    assert s['current_max_new_minus_max_old_quantiles']['restricted_correct'] is None
