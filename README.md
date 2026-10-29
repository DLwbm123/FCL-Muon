# MuLoRA-FL (Shared-RRS + FedAvg)

**Actual-update global correction (2026-10-29): new five-task pilot running.** [Frozen protocol (中文)](research/fed_spectra/ACTUAL_UPDATE_PROTOCOL_ZH.txt), [source revision](reports/actual_update_global_20261029/SOURCE.json), [CPU checks](reports/actual_update_global_20261029/CPU_TESTS.json), [real GPU preflight](reports/actual_update_global_20261029/REAL_PREFLIGHT.json), [startup receipt](reports/actual_update_global_20261029/STARTUP.json). Fresh matched arms use eta=0 and eta=0.1, with identical RRS/local learning/head settings. Correction uses actual client increments after FedAvg and preserves aggregate update norm. T3 qualification is followed by T5 regardless of T3 scores. No performance benefit has yet been established; the previous stopped experiment below remains stopped.

**Previous FedSPECTRA run (2026-10-29): remains stopped by user.** [Stopped experiment summary (中文)](reports/fed_spectra_20261028/STOPPED_REPORT_ZH.txt) and [machine-readable results](reports/fed_spectra_20261028/STOPPED_SUMMARY.json). T8 is the latest common evaluated stage: raw accuracy Control 77.6375%, MVP 76.9875%, Full 72.4000%. Control/Full also have T9 results; MVP has no T9 evaluation. All three lack T10 completion/reload qualification. The added mechanisms have not shown a benefit over the matched control in the available stages. Earlier T3 reports remain historical snapshots, not current running status.

**Review and results:** [Pro text/source bundle and audit index](review/README.md), [completed three-method pilot comparison](reports/PILOT_COMPARISON_ZH.txt), [implementation/configuration audit](reports/IMPLEMENTATION_AUDIT_20261025_ZH.txt). The new history_spectral experiments remain user-paused pending Pro review; their numerical qualification has not passed.

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

An additionally authorized official LoRM three-task comparison is configured in `configs/pilot_lorm_official.json`. Its algorithm methods come directly from the pinned upstream code, using rank16, AdamW lr5e-4, alternating A/B updates, RegMean, the original linear head and default mixed precision. It loads the existing seed0/alpha.05 client manifest. Read `reports/LORM_PROTOCOL.json` for disclosed differences and additional Gram-forward cost. Install the LoRM dependency overlay, set `ENTRY_MODULE=fcl.lorm_train` for the CLI or `ENTRY_FILE=scripts/lorm_job.py` for its persisted-deadline supervisor, and pass the config through `RUN_CONFIG`. The LoRM job is limited to this pilot; it does not add a LoRM hyperparameter search or full experiment grid. All methods' primary stage results compare among all seen classes and use the same forgetting definition.

## Pro review of history_spectral

The research implementation is on branch `codex/history-spectral-v1`. Start with `research/history_spectral/REVIEW_FOR_PRO.txt`, then its full user protocol and source audit. The experiment family is **paused by the user pending Pro review**; `reports/USER_PAUSE.json` blocks research training and queue entry points. Existing faithful baselines and LoRM are independent.

The CPU numerical/lifecycle suite can be checked without a benchmark run:

```sh
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/test_history_math.py tests/test_history_lifecycle.py
python -m fcl.history_train --help
```

Future reviewed execution uses the same neutral launcher and config environment as the baseline, with `ENTRY_MODULE=fcl.history_train` and a config under `configs/history_spectral/`; `RESUME` restores a round boundary and `FORK_TASK1` loads a matching exported task-one checkpoint. Smoke and bounded-pilot manifests are under `research/history_spectral/`. The separate full research matrix is configuration-only and is never automatically launched. Do not remove the pause marker or start these commands until the user explicitly resumes experiments. See `reports/history_spectral/PRE_REVIEW_STATUS.json`: unit tests passed, but actual-step FP32 QP qualification remains unresolved; no performance claim follows from the pre-review smoke.
