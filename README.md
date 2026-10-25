# MuLoRA-FL (Shared-RRS + FedAvg)

Reconstruction of the user-provided MuLoRA paper's core mechanisms with minimal FCIL adaptation, plus the independently specified Takezawa FedMuon combination. This does not reproduce original paper tables. See [SOURCE_AUDIT.md](SOURCE_AUDIT.md), [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) and the frozen user prompt in `PROTOCOL_ZH.txt`; the later user correction recorded in SOURCE_AUDIT supersedes its shared-operator clause.

Code: `fcl.math`, `fcl.model`, `fcl.data`, `fcl.train`. Model registry names are `mulora_fl` and `mulora_fl_fedmuon`. The dedicated trainer replaces LoRM's training mechanics and uses the pinned upstream split and preprocessing.

## Setup

```sh
HTTPS_PROXY=http://127.0.0.1:7897 HTTP_PROXY=http://127.0.0.1:7897 NO_PROXY='' git -c http.proxy=http://127.0.0.1:7897 clone https://github.com/aimagelab/fed-mammoth.git sources/fed-mammoth
git -C sources/fed-mammoth checkout 6d57b5d361f743f3ff0cd1371409bd7a1158de64
python -m pip install -r requirements.txt
python -m pytest -q tests
```

Install the appropriate CUDA build of torch/torchvision for the hardware. Validated server uses Python3.12, torch2.7.1+cu128, torchvision0.22.1+cu128; all additional actual versions are recorded in `reports/environment.json`. Preserve the MIT upstream license. Original third-party snapshots are locally retained, excluded from publication, and not required to redistribute our code.

## Execution

The Python CLI supports `--config` and `--resume`. To respect neutral visible command lines on servers, use the environment-driven entry:

```sh
export PROJECT_ROOT=/path/to/project
export PYTHONPATH="$PROJECT_ROOT/deps:$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES=6
export ENTRY_MODULE=fcl.train
export RUN_CONFIG="$PROJECT_ROOT/configs/smoke_mulora_fl.json"
/path/to/python /path/to/neutral/e.py
```

`/path/to/neutral/e.py` is an installed copy of `scripts/entry.py`; its visible argv does not reveal the project or method. For exact communication-boundary recovery set `RESUME="$PROJECT_ROOT/runs/smoke_mulora_fl/latest.pt"` and reuse the same config. Resume preserves the original run clock. A failed costly run is not silently restarted.

`configs/smoke_*.json`: 2 clients, 2 tasks, 2 rounds, shortened batches/epochs, clearly non-scientific. `configs/pilot_*.json`: CIFAR first 3 official tasks, 10 clients, α=.05, seed0, full 5×5 budget. `configs/formal_*.json`: first full CIFAR α=.05 seed0 pair. Remaining full grid configs are named by dataset/alpha/seed/method. `queue.json` records the 40 stages: four smoke/pilot qualifications plus 36 full runs. Dataset class order is never shuffled. Every method pair uses the same manifest; its labels and split identities can be validated with `scripts/validate_manifests.py`.

Prepare weights and CIFAR using `ENTRY_FILE=scripts/prepare.py` (unset ENTRY_MODULE). The preparation script resolves the exact HF repository revision and hashes the actual weight file. ImageNet-R images must be prepared under `data/imagenet-r`; the original 24k/6k YAML split is loaded verbatim. No substitute backbone is permitted.

Queue runner: set `ENTRY_FILE=scripts/queue.py`, unset ENTRY_MODULE/RUN_CONFIG/RESUME, and run the neutral entry in a persistent process. Each config retains its first launch time, has a process-group hard deadline below 24 hours, and uses a conservative allocated-storage guard at 45GB. It stops on failure or unavailable resources. A previous failed state requires analysis before editing the queue state. Completed qualification stages are reviewed before allowing scientific jobs; metrics do not select hyperparameters or checkpoints.

## Results and status

`reports/STATUS.json` tracks actual stage status. A run is complete only with its `COMPLETE.json` receipt; PID or log activity alone is insufficient. `metrics.json` retains the all-seen task matrix, sample micro and task macro accuracies, mean stage micro and explicitly defined forgetting. `events.jsonl` retains samples/steps/LR, spectral diagnostics and runtime/communication data. Values are fractions.

```sh
python scripts/summarize.py runs
python scripts/validate_manifests.py manifests/*.json
```

No unrun score is populated. Smoke, pilot and formal results stay separate. Latest atomic checkpoint is retained per run to control storage. Shared raw datasets, weight cache and original-source snapshots remain outside Git.
