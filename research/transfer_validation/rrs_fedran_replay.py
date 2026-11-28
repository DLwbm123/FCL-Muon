"""Run the existing FedRAN head on a committed RRS T1 encoder."""
import json
import os
from pathlib import Path
from statistics import mean
import time

import torch

from research.adamw_t10.storage import allocated, save, write
from research.analytic_comparison.baselines import (
    fedran_packet,
    get_features,
    mathematical_checks,
    proposal,
    score,
    spectral_merge,
    spectral_ridge,
)
from research.stat_ablation.runtime import build, setup, validate
from research.transfer_validation.frozen_stage import restore_model


def main():
    setup()
    source, frozen = validate()
    root = Path(os.environ["COHORT_ROOT"])
    protocol = json.loads((root / "protocol.json").read_text())
    clock = json.loads((root / "clock.json").read_text())
    spec = protocol["classifier"]
    config = frozen["config"]
    if frozen["manifest_sha256"] != protocol["manifest_sha256"]:
        raise ValueError("Unexpected frozen manifest")
    for key in ("dataset", "seed", "tasks", "clients"):
        if config[key] != protocol[key]:
            raise ValueError(f"Unexpected configuration field: {key}")

    def check_budget():
        if time.time() >= clock["deadline"] - 30:
            raise TimeoutError("Original clock reached save reserve")

    mathematical_checks()
    pilot = build(clock)
    endpoint = restore_model(pilot, source / "src/runs/rrs_shared/T1.pt")
    if endpoint["task"] != 1 or endpoint["merges"] != [0]:
        raise ValueError("The frozen control requires the committed T1 encoder")
    backbone = pilot.model.backbone.eval().requires_grad_(False)
    projection = proposal(backbone.num_features, spec["width"], protocol["seed"])
    projection64 = projection.double()

    def cls(images):
        return backbone.forward_features(images)[:, 0]

    def features(split, indices):
        return get_features(cls, pilot.raw[split], indices, spec["feature_batch"], pilot.test_tf, check_budget)

    print(json.dumps({"validated_t1": True, "device": torch.cuda.get_device_name()}), flush=True)
    parts = pilot.manifest["partitions"]
    cpt = config["cpt"]
    summary = None
    stages = []
    logical_bytes = {"statistics_upload_bytes": 0, "head_download_bytes": 0}
    for task, clients in enumerate(parts["train"]):
        check_budget()
        if allocated(root) >= protocol["budgets"]["allocated_bytes"] - 100_000_000:
            raise RuntimeError("Allocated storage reached save reserve")
        task_summary = None
        task_sums = torch.zeros(spec["width"], cpt, device="cuda", dtype=torch.float64)
        for indices in clients:
            if not indices:
                continue
            z, labels = features(0, indices)
            current = labels - task * cpt
            if len(labels) != len(indices) or not bool(((0 <= current) & (current < cpt)).all()):
                raise ValueError("Training sample coverage differs from frozen task")
            local_summary, sums = fedran_packet((z.double() @ projection64).relu(), current, cpt, spec["rank"])
            local_summary = tuple(tensor.float().double() for tensor in local_summary)
            task_summary = spectral_merge(task_summary, local_summary, spec["rank"])
            task_sums += sums.float().double()
            logical_bytes["statistics_upload_bytes"] += 4 * (
                sum(tensor.numel() for tensor in local_summary) + spec["width"] * (task + 1) * cpt
            )
        summary = spectral_merge(summary, task_summary, spec["rank"])
        cumulative = task_sums if task == 0 else torch.cat((cumulative, task_sums), dim=1)
        classifier = spectral_ridge(summary, cumulative, spec["ridge"]).float()
        logical_bytes["head_download_bytes"] += len(clients) * classifier.numel() * classifier.element_size()

        indices = [i for groups in parts["test"][: task + 1] for group in groups for i in group]
        z, labels = features(1, indices)
        if len(labels) != len(indices) or set(labels.tolist()) != set(range((task + 1) * cpt)):
            raise ValueError("Evaluation sample coverage differs from frozen task")
        stage = score(z, labels, projection, classifier, stages, cpt, "fedran")
        stages.append(stage)
        raw_aia = mean(stage["all_seen"] for stage in stages)
        write(root, root / f"T{task + 1}.json", {"stage": stage, "raw_aia": raw_aia, "logical_bytes": logical_bytes})
        print(json.dumps({"task": task + 1, "raw_aia": raw_aia, "raw_accuracy": stage["all_seen"]}), flush=True)

    output = root / "latest.pt"
    save(root, output, {"classifier": classifier.cpu(), "projection": projection.cpu()})
    reloaded = torch.load(output, map_location="cuda", weights_only=True)
    z, labels = features(1, indices)
    if score(z, labels, reloaded["projection"], reloaded["classifier"], stages[:-1], cpt, "fedran") != stages[-1]:
        raise RuntimeError("Final classifier reload changed metrics")
    write(root, root / "COMPLETE.json", {
        "status": "completed",
        "tasks": len(stages),
        "raw_aia": raw_aia,
        "raw_final": stages[-1]["all_seen"],
        "centered_aia": mean(stage["centered"]["all_seen"] for stage in stages),
        "stages": stages,
        "logical_bytes": logical_bytes,
        "new_training_visits": 0,
        "new_optimizer_updates": 0,
        "final_checkpoint_reload_equal": True,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
    })


if __name__ == "__main__":
    main()
