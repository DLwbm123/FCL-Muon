"""Fixed disjoint training-side sample split for a single endpoint audit."""


def split(groups, max_samples=32):
    if max_samples < 2:
        raise ValueError('maximum must allow two nonempty splits')
    all_indices = [i for group in groups for i in group]
    if len(set(all_indices)) != len(all_indices):
        raise ValueError('client train partitions overlap')
    results = []
    for group in groups:
        chosen = sorted(group)[:max_samples]
        if len(chosen) < 2:
            results.append(([], []))
        else:
            half = len(chosen) // 2
            results.append((chosen[:half], chosen[half:]))
    return results
