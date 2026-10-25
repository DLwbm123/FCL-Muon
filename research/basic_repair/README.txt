Limited basic repair, 2026-10-25

User scope: focus on the user's method foundation versus LoRM. Follow the supplied
Pro repair bundle; recover one original MuLoRA T1 if no T1 model exists, then use
one frozen global encoder for head diagnostics. history_spectral remains paused.
No new FedMuon training, no full matrix, no automatic second encoder experiment.

The supplied prompt and NumPy tools are retained here for provenance. The shared
ChatGPT URL timed out when opened; the attached bundle was read in full.
The lock file predates launching training or obtaining the new head results.
One conditional recommendation in the prompt could otherwise lead to selecting
a head after looking at test: we instead SAVE H1 UNCONDITIONALLY as the sole
predeclared repair candidate. H0/H1/H2/H3 are all reported, with no test-based
hyperparameter, checkpoint, iteration, or replacement-head selection.

Training preparation:
  Run scripts/prepare_basic_repair.py from the repository. It exports fcl from
  commit00b6508 to runs/basic_repair_deployment, changing ONLY snapshot retention.
  Shared data/manifests/weights/sources remain external read-only resources.
  Copy scripts/basic_repair_{job,centering,heads}.py and this research directory.
  The server job runs each phase through the existing neutral entry. See the
  locked protocol and reports/basic_repair_20261025 for provenance and status.
  Source hashes are recorded because the supplied protocol explicitly requests
  them. Do not publish input images, features, private logits or checkpoints.

Three preflight checks passed in the isolated frozen-source environment:
  1. Supplied raw prototype/ridge pooling and block-centering mathematical checks.
  2. Unequal-client gradient sums agree with one pooled global cosine-head update.
  3. Detached endpoint serialization leaves Torch/NumPy RNG states unchanged.
The training loop performs no additional forward pass between optimizer updates.
It retains round boundaries using hard links and stores local B/head endpoints.
All added diagnostic feature extraction happens after completing T1 training.

H0: original learned cosine head on adapted T1 encoder.
H1: normalized raw class feature sums across clients; class counts are not
    multiplied into raw sums again. Save a separately named checkpoint always.
H2: one server cosine head, initialized from H0, 1000 full-batch steps, lr.001,
    AdamW/wd0. Each step adds client gradient sums / total N, then updates once.
    This is extra head optimization, not part of the faithful 5x5-epoch budget.
H3: normalized CLS plus intercept; FP64 ridge normal equations, lambda.01,
    intercept unpenalized. Resulting linear weights are not normalized.
Pretrained encoder: H1/H3 only, same feature/data/transform rules. It does not
use the IN21K classification head as a CIFAR classifier.

Clients all use the SAME global encoder for these statistics. Full original
training data are used; no after-the-fact subset is called unseen validation.
All fixed readouts are evaluated for diagnosis and reported. No scores select
parameters or trigger a second encoder-training run.

The class6/class7 issue is within T1. Block centering preserves every within-
block argmax and cannot repair that ordering. Full confusion, old and new
accuracy changes must be retained even where an intervention makes them worse.

The original separate server queue is not stopped or modified. This diagnostic
job has its own persisted first-start deadline (85500s) and12GB new-storage
guard, shared across centering, one encoder run and all head fitting/analysis.
Only one existing heartbeat is updated; no additional monitor is created.
