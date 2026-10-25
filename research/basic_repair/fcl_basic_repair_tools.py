"""Read-only, parameter-free task-block intervention and fixed-feature head tools.

This module does not load a model, train an encoder, select hyperparameters,
or access the network. The CLI reads existing private logits and writes one
new aggregate report. Labels are used only in reporting, never in centering.
Fixed-feature head helpers require the SAME global encoder on every client.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def _scores(scores: np.ndarray, classes_per_task: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or not scores.size or not np.isfinite(scores).all():
        raise ValueError("Expected a nonempty finite [examples, classes] matrix")
    if classes_per_task < 1 or scores.shape[1] % classes_per_task:
        raise ValueError("Class count must be divisible by classes_per_task")
    return scores


def center_task_logits(scores: np.ndarray, classes_per_task: int = 10) -> np.ndarray:
    """Use fixed class groups, never an example's true/predicted task identity."""
    scores = _scores(scores, classes_per_task)
    blocks = scores.reshape(len(scores), -1, classes_per_task)
    result = blocks - blocks.mean(axis=2, keepdims=True)
    # Preserve ALL within-block decisions, including blocks unrelated to labels.
    if not np.array_equal(blocks.argmax(2), result.argmax(2)):
        raise ArithmeticError("Within-block decisions changed due to numerical error")
    return result.reshape(scores.shape)


def summarize(scores: np.ndarray, labels: np.ndarray, classes_per_task: int = 10) -> dict:
    scores = _scores(scores, classes_per_task)
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != len(scores):
        raise ValueError("Labels must match the example dimension")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Expected integer labels")
    classes = scores.shape[1]
    if (labels < 0).any() or (labels >= classes).any():
        raise ValueError("Label outside the seen class set")
    tasks = classes // classes_per_task
    predictions = scores.argmax(1)
    true_tasks = labels // classes_per_task
    pred_tasks = predictions // classes_per_task
    task_confusion = np.bincount(true_tasks * tasks + pred_tasks,
                                 minlength=tasks * tasks).reshape(tasks, tasks)
    per_task = []
    for task in range(tasks):
        subset = true_tasks == task
        if not subset.any():
            raise ValueError("Each seen task must have evaluation examples")
        lo = task * classes_per_task
        restricted = scores[subset, lo:lo + classes_per_task].argmax(1) + lo
        good = restricted == labels[subset]
        global_good = predictions[subset] == labels[subset]
        stolen = good & (pred_tasks[subset] != task)
        if int(good.sum()) - int(global_good.sum()) != int(stolen.sum()):
            raise ArithmeticError("Error decomposition failed")
        per_task.append({
            "task_zero_based": task, "n": int(subset.sum()),
            "restricted_accuracy": float(good.mean()),
            "all_seen_accuracy": float(global_good.mean()),
            "restricted_correct_but_other_task_wins": int(stolen.sum()),
        })
    new = true_tasks == tasks - 1
    old = ~new
    result = {
        "n": len(labels), "all_seen_accuracy": float((predictions == labels).mean()),
        "per_task": per_task, "task_group_confusion": task_confusion.tolist(),
        "current_all_seen_accuracy": float((predictions[new] == labels[new]).mean()),
        "current_restricted_accuracy": per_task[-1]["restricted_accuracy"],
        "current_to_first_task": float((pred_tasks[new] == 0).mean()) if tasks > 1 else None,
        "current_to_old": float((pred_tasks[new] < tasks - 1).mean()) if tasks > 1 else None,
        "old_all_seen_accuracy": float((predictions[old] == labels[old]).mean()) if old.any() else None,
        "old_to_current": float((pred_tasks[old] == tasks - 1).mean()) if old.any() else None,
    }
    return result


def fixed_feature_statistics(features: np.ndarray, labels: np.ndarray, n_classes: int) -> dict:
    """Raw sums of normalized features from one fixed global encoder.

    Cache features only on clients. The returned moments are sufficient for
    global prototypes and ridge. Do not multiply raw sums by client sizes again.
    An intercept is appended for ridge. No head fitting or validation selection
    is performed here.
    """
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    if features.ndim != 2 or not len(features) or not np.isfinite(features).all():
        raise ValueError("Nonempty finite feature matrix required")
    if labels.shape != (len(features),) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Integer labels matching features required")
    if n_classes < 1 or (labels < 0).any() or (labels >= n_classes).any():
        raise ValueError("Labels outside fixed current class set")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if (norms <= 1e-12).any():
        raise ValueError("Cannot normalize near-zero feature vectors")
    z = features / norms
    zz = np.concatenate([z, np.ones((len(z), 1))], axis=1)
    target = np.eye(n_classes)[labels]
    return {"n": len(z), "gram": zz.T @ zz, "cross": zz.T @ target,
            "class_sums": target.T @ z, "counts": target.sum(axis=0)}


def merge_statistics(parts: list[dict]) -> dict:
    if not parts:
        raise ValueError("At least one client statistics object is required")
    out = {key: np.array(parts[0][key], copy=True) for key in ["gram", "cross", "class_sums", "counts"]}
    out["n"] = int(parts[0]["n"])
    for part in parts[1:]:
        for key in ["gram", "cross", "class_sums", "counts"]:
            if part[key].shape != out[key].shape:
                raise ValueError("Client feature/class dimensions differ")
            out[key] += part[key]
        out["n"] += int(part["n"])
    return out


def global_prototype_rows(statistics: dict) -> np.ndarray:
    sums = statistics["class_sums"]
    norms = np.linalg.norm(sums, axis=1, keepdims=True)
    if (statistics["counts"] <= 0).any() or (norms <= 1e-12).any():
        raise ValueError("Prototype undefined for absent class or zero resultant")
    return sums / norms


def global_ridge(statistics: dict, regularization: float) -> np.ndarray:
    """Minimize mean square error + lambda*||feature weights||^2.

    Intercept is unpenalized. Returned [(d+1), classes] weights are LINEAR;
    normalizing their rows or columns would no longer be this ridge solution.
    Lambda must be fixed in advance or selected using valid training holdout.
    """
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("Require an explicit positive regularization")
    n = statistics["n"]
    gram = statistics["gram"] / n
    penalty = np.eye(len(gram)) * regularization
    penalty[-1, -1] = 0
    return np.linalg.solve((gram + gram.T) / 2 + penalty, statistics["cross"] / n)


def self_test() -> dict:
    rng = np.random.default_rng(193)
    scores = rng.normal(size=(72, 30))
    centered = center_task_logits(scores)
    assert np.array_equal(scores.reshape(72, 3, 10).argmax(2), centered.reshape(72, 3, 10).argmax(2))
    one = scores[:, :10]
    assert np.array_equal(one.argmax(1), center_task_logits(one).argmax(1))
    labels = np.tile(np.arange(30), 3)[:72]
    a, b = summarize(scores, labels), summarize(centered, labels)
    assert [x["restricted_accuracy"] for x in a["per_task"]] == [x["restricted_accuracy"] for x in b["per_task"]]

    # Exact cosine common-component ambiguity; both sets of rows have unit norm.
    x = np.array([.3, 0., np.sqrt(.91)])
    plus = np.array([[1., 0., 1.], [-1., 0., 1.]]) / np.sqrt(2)
    minus = plus.copy(); minus[:, 2] *= -1
    np.testing.assert_allclose(np.linalg.norm(plus, axis=1), 1.)
    np.testing.assert_allclose(np.linalg.norm(minus, axis=1), 1.)
    zp, zm = 16 * plus @ x, 16 * minus @ x
    np.testing.assert_allclose(zp[0] - zp[1], zm[0] - zm[1])
    assert np.r_[0., zp].argmax() == 1 and np.r_[0., zm].argmax() == 0

    # Unequal clients: raw sufficient-statistic addition equals pooled fitting.
    f = rng.normal(size=(71, 9)); y = np.arange(71) % 4
    chunks = [slice(0, 1), slice(1, 8), slice(8, 71)]
    distributed = merge_statistics([fixed_feature_statistics(f[s], y[s], 4) for s in chunks])
    pooled = fixed_feature_statistics(f, y, 4)
    for key in ["gram", "cross", "class_sums", "counts"]:
        np.testing.assert_allclose(distributed[key], pooled[key], atol=1e-12)
    wr, wp = global_ridge(distributed, .01), global_ridge(pooled, .01)
    np.testing.assert_allclose(wr, wp, atol=1e-12)
    np.testing.assert_allclose(global_prototype_rows(distributed), global_prototype_rows(pooled), atol=1e-12)
    return {"self_test": "passed", "scope": "Synthetic mathematical checks; no real data/model performance",
            "cosine_plus_logits": zp.tolist(), "cosine_minus_logits": zm.tolist(),
            "distributed_vs_pooled_ridge_max_abs": float(np.max(np.abs(wr - wp)))}


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--classes-per-task", type=int, default=10)
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
    if not args.inputs:
        if not args.self_test:
            parser.error("Provide --inputs/--output or --self-test")
        return
    if args.output is None:
        parser.error("--output is required with --inputs")
    if args.output.exists():
        raise FileExistsError("Output already exists; no overwrites are permitted")
    reports = []
    for path in args.inputs:
        before = _hash(path)
        with np.load(path, allow_pickle=False) as archive:
            scores, labels, indices = archive["logits"], archive["labels"], archive["indices"]
        if len(indices) != len(scores) or len(np.unique(indices)) != len(indices):
            raise ValueError("Expected unique sample identifiers matching logits")
        centered = center_task_logits(scores, args.classes_per_task)
        original = summarize(scores, labels, args.classes_per_task)
        changed = summarize(centered, labels, args.classes_per_task)
        assert before == _hash(path), "Input was changed during read-only analysis"
        reports.append({"input_basename": path.name, "input_sha256": before,
                        "original": original, "block_centered": changed,
                        "all_seen_accuracy_change_pp": 100 * (changed["all_seen_accuracy"] - original["all_seen_accuracy"])})
    output = {"rule": "Subtract the unweighted mean of EVERY fixed training-task logit block, then all-seen argmax",
              "sample_task_identity_used_for_prediction": False, "parameters_fit": False,
              "interpretation": "One prespecified diagnostic intervention; no performance claim before running on real logits",
              "classes_per_task": args.classes_per_task, "results": reports}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(output, stream, indent=2)
    print(json.dumps({"output": str(args.output), "files": len(reports)}))


if __name__ == "__main__":
    main()
