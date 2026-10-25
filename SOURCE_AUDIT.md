# Sources and adaptation audit — 2026-10-25

This project reconstructs core algorithms and adapts them to one FCIL protocol. It does not reproduce either original paper's experimental tables.

## Frozen upstream

- FCIL: https://github.com/aimagelab/fed-mammoth commit `6d57b5d361f743f3ff0cd1371409bd7a1158de64`, verified by Git checkout. Unmodified snapshot retained in `sources/fed-mammoth`; MIT license retained there. The actual `_split_fcil` function is extracted by AST and executed on sample indices. This preserves its NumPy Dirichlet/PyTorch categorical draws, clipping, minimum six train samples, and train-proportional test assignment including rounding/padding. It already has a 10,000 outer retry assertion. Model/data/training/args/global constants and ImageNet-R YAML were inspected. New trainer reuses the exact split and preprocessing sequence but replaces LoRM training and metrics; it is not an upstream CLI model registration.
- MuLoRA: user-provided local PDF is the primary paper, anonymous ICLR 2027 submission, title “MuLoRA: Spectrally Balanced Low-Rank Adaptation for Continual Learning”. Extract retained locally. arXiv https://arxiv.org/abs/2610.02283v1 and HTML checked. The attachment and v1 agree on the core RRS/SSB formulas and Appendix D defaults inspected; byte identity is not asserted. Author public repository listing and arXiv page checked 2026-10-25; no official MuLoRA implementation confirmed. This is a bounded search, not proof none exists. `junkangLiu0/FedMuon` is unrelated to the required Takezawa algorithm and is not used.
- FedMuon: Takezawa, Koloskova, Jiang, Stich, ICLR 2026; https://arxiv.org/abs/2509.26337 and official proceedings PDF/ZIP. Downloaded supplemental ZIP SHA256 exactly `dfc176670be5a63a8256d521229819ef428ca0529555f0efd58f6194b066904b`. Original extracted snapshot retained locally. No standalone license located in the supplemental tree: do not redistribute this snapshot without establishing rights. Our optimizer is an independent formula implementation.

## FedMuon code findings

`code/optimizer/fedmuon_optimizer.py` has an outer loop over all parameters enclosing another loop over all parameters: a group of n matrices receives n updates per step. The independent implementation iterates once and is tested with two matrices and two rounds. `use_muon=False` actually applies corrected momentum SGD, not Adam. In `average`, full participation overwrites controls by end momentum and averages them. Under partial participation stale global controls are averaged, rather than explicitly applying Algorithm 1 control increments; this path is not adopted or claimed validated. `local_muon_optimizer.py`, `muon.py` and `evaluate.py` were inspected; auxiliary Adam variants must not be confused with the FedMuon file's SGD branch.

## Configuration provenance

| Choice | Source |
|---|---|
| Historical whitening, inverse-transpose map, QR, fixed A, B-only learning and task-end merge | MuLoRA §3, Appendix B |
| Frozen past-feature moments are historical-response proxies, not recomputed old features | MuLoRA Appendix B.1 |
| ViT-B/16 IN21K, rank32, cosine classifier, dataset-specific B LR | MuLoRA Appendix D |
| Exact FP64 eigensolver | user reference requirement; paper describes randomized SVD |
| 10 tasks, 10 clients, 5 rounds × 5 epochs, batch16, seed0/1/2, α .5/.1/.05 | user protocol; differs from MuLoRA central 20 epochs/batch48 |
| Q and V in 12 blocks, scale1, patch-only statistics | explicit user completion; original internal projection placement unconfirmed |
| Cosine scale16, old rows frozen, current-only CE, AdamW head lr.001/wd0, reset each round | explicit user completion; upstream LoRM has a linear head |
| NS coefficients 3.4445/-4.7750/2.0315, five iterations, FP32 Frobenius+1e-7, sqrt(max dimension) scaling | Takezawa supplemental operator; FedMuon only; MuLoRA Eq.3 omits the dimension multiplier |
| Raw momentum beta.9, no Nesterov; correction before NS, controls frozen per round | Algorithm 1 plus user protocol |
| Sample weighting, full participation, variable local steps from five epochs | FCIL adaptation; original theorem uses uniform objective/fixed K |
| Per-client state across rounds, reset when A changes | explicit user task-coordinate adaptation |
| epsilon = 1e-6 × max(mean diag historical S, mean diag current S, 1e-12), ×10 retry jitter | project numerical completion, training-only; logged per layer/task |
| Statistics deterministic upstream test-form transform applied ONLY to current train images | explicit engineering completion |
| FP32 backbone/training, FP32 momentum/NS, FP64 statistics/decomposition | controlled project precision; shared across methods |

Unlike LoRM, this project has no alternating A/B training, RegMean, historical adapter average or inherited 48-matrix list. There are 24 B matrices (12 Q, 12 V), each 768×32; twelve shared Q/V input statistics and A bases, since each fused QKV receives the identical input tensor. CIFAR partition/class identity does not depend on model RNG. All reported test predictions compete across every seen class. Hyperparameters are fixed, with no selection based on test scores.

## User correction — supersedes initial shared-operator requirement

Primary baseline is **MuLoRA-FL (Shared-RRS + FedAvg)**. Its SSB follows attachment Eq.(3): B -= lr * Y_J, with no sqrt(dimension) multiplier. The PDF specifies symbolic a,b,c,J,δ and calls G momentum, without explicit momentum recurrence or numerical NS defaults. Project completions are beta=.9 exponential momentum without Nesterov, NS coefficients (3.4445,-4.7750,2.0315), J=5, δ=1e-7, FP32; these are documented practical Muon defaults, not verified original MuLoRA values. FedMuon combination independently retains Takezawa sqrt(max(d_out,r)) scaling and control correction. Thus the current pair changes operator scale as required by source priority, not exclusively controls. A unified-operator ablation is not run. Single-client centralized-core equivalence is a required gate using identical initialization, batches, schedule, state semantics and task transitions. Statistics sum and shared-A broadcast are explicit federation choices and communication costs.

## Authorized official LoRM three-task comparison

The user additionally authorized one LoRM pilot with the same existing client partition and training budget. `fcl.lorm_upstream` loads the original files without changing their algorithm methods, bypassing only unrelated package registry scanning. `fcl.lorm_train` calls the official begin/end task, begin/end round, observe, Gram collection, spatial aggregation and temporal merge methods. README CIFAR alpha=.05 standard config is rank16, lr5e-4, AdamW/wd0, alternating B/A, original 48 matrix positions, linear head, backbone Gram alpha0/head alpha.5, FP64 merge solves, one augmented Gram pass per round. Official main default precision16-mixed and matmul precision medium are retained. These differ from the MuLoRA FP32/QV/r32/cosine setup and are disclosed as a standard-method comparison, not a single-factor ablation. Both use the identical IN21K weight file and existing n10/alpha.05/seed0 manifest, 3 tasks, 5 rounds×5 full optimizer epochs and all-seen evaluation. Required additional Gram forwards are algorithm overhead, not additional optimizer epochs. No hyperparameter search or test-selected checkpoint.

First LoRM smoke attempt stopped before the first optimizer step because the orchestration harness had moved the initial server A/B to CPU before broadcasting to CUDA clients. Fixed the harness to move the server to its device before round broadcast; official source unchanged. Original failure log is retained remotely as `reports/lorm-smoke-attempt1.log`. This is an entry/device error, not a scientific negative result or a changed algorithm. LoRM uses a separate dependency overlay, preserving the running MuLoRA/FedMuon environment.

## History-spectral research implementation (user specification v1.1)

Research branch `codex/history-spectral-v1` preserves baseline parent `00b6508`. See `research/history_spectral/SOURCE_AUDIT.txt` and the full user protocol in that directory. New raw centered functional-history statistics are distinct from the original RRS statistics. Research exact-polar, one-step historical-gradient polar, spectral QP and Frobenius QP mechanisms use separate registry names; faithful finite-NS and Takezawa definitions remain unchanged. Deployed research code is isolated from the existing running baseline/LoRM source directory.

The user's latest instruction pauses all new research experiments pending Pro correctness review. Mathematical tests passed, but the abbreviated FP32 spectral smoke did NOT pass numerical qualification:35/192 actual parameter increments failed strict convergence/feasibility certification, maximum relative gap approximately1.55e-4 vs target1e-4. Preserve this finding; do not present the implementation as numerically qualified or start a pilot. No new research pilot/formal result exists. Pause markers and the existing single heartbeat enforce the user's instruction.
