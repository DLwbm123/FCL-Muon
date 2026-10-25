# Pro correctness review and existing results

The history_spectral experiment family is **paused by the user pending Pro review**.
Publication is not authorization to resume training, calibration, or evaluation.

- [Complete Pro text bundle](FCL_history_spectral_Pro_review_97ba2b6.txt): full adopted specification, implementation, mathematical/lifecycle tests, baseline diff, and review questions. Frozen source commit `97ba2b6`.
- [Frozen source ZIP](FCL_history_spectral_source_97ba2b6.zip): tracked source/configuration/documentation at the same commit. No datasets, weights, third-party PDFs or source snapshots.
- [Current implementation and configuration audit](../reports/IMPLEMENTATION_AUDIT_20261025_ZH.txt): static source inspection and existing-log analysis requested after the three pilots completed. No retuning, retraining, or new model evaluation.
- [Numerical diagnosis and Pro review response](../reports/history_spectral/numerics_audit/DIAGNOSIS_ZH.txt) and [updated numerical-code addendum](NUMERICS_ADDENDUM_20261025.txt): 42 mathematical/lifecycle/core tests passed. Only synthetic inputs; original real-data smoke remains unqualified and paused.
- [Completed three-method pilot analysis](../reports/PILOT_COMPARISON_ZH.txt), [metrics and full task matrices](../reports/pilot_comparison.json), [completion audit](../reports/pilot_completion_audit.json), [existing training-log summary](../reports/PILOT_LOG_AUDIT.json).
- [History-spectral pre-review status](../reports/history_spectral/PRE_REVIEW_STATUS.json): prior abbreviated smoke is separate from the faithful pilots. Spectral numerical qualification did not pass; 35/192 actual increments were flagged, maximum relative gap approximately 1.55e-4 against 1e-4. No research pilot or formal result exists.

The repository tree contains the latest code. The original text bundle and ZIP preserve source version `97ba2b6`; the numerical-code addendum covers subsequent solver/certificate and diagnostic changes on `codex/history-spectral-numerics-review`. Read both when reviewing the latest implementation. Completed pilot results are single-seed, first-three-task comparisons with common data and optimizer-training budget; architecture, optimizer, precision and statistics overhead differ between standard methods. They do not establish a single-mechanism causal effect or a full ten-task conclusion.

The audit found no identified core implementation error explaining the performance gap, but it does not prove complete correctness. In particular, the previous “new-class fitting” description must be read as global all-seen accuracy on new-class samples, not a measured local fitting failure. Please review the audit's unresolved diagnostic limits alongside the source.
