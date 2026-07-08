# SRP-Augmentation Project Status

Last updated: 2026-07-08

## Current Stage

The repository is in the focused experimental-comparison stage. The training
pipeline, standard augmentation baselines, similarity-guided methods, neighbor
construction, and anchor-score selection are implemented.

Implemented pieces:

- CIFAR-10 and CIFAR-100 k-shot splits with fixed validation data.
- ResNet50 and ViT training through `src/train.py`.
- Baselines: none, MixUp, CutMix, and AugMix.
- Proposed variants: SimMixUp and SimCutMix.
- Similarity-guided pairing with class-aware/class-agnostic neighbors, rank
  windows, mix probability, warmup, and anchor-score gating.
- Canonical experiment outputs under `results/experiments/<dataset>/<model>/k<k>/<method>/...`.

## Canonical References

Use these files as the current source of truth:

- Detailed result narrative: `notes/experiment_results_summary_v1.md`
- Experiment artifact folder: `results/experiments/`
- Sortable run index: `results/experiments/manifest.csv`
- Experiment folder map: `results/experiments/README.md`

Older duplicate notes and baseline-only summaries were removed to keep the repo
clean.

## Current Evidence

The local experiment catalog contains 53 complete summary/CSV metric pairs after
deduplicating two equivalent reruns. Three non-exact reruns are preserved
under `legacy/` for provenance. Most results are CIFAR-100 with `subset_seed=0`
and `train_seed=0`, so they are still preliminary single-seed evidence.

Main observations:

- CutMix remains the strongest standard baseline in the 100-epoch baseline grid.
- The best guided result so far is SimCutMix on CIFAR-100, ResNet50, k=100,
  class-agnostic K40 ranks 21-40, with `40.02%` test accuracy.
- The repeated class-agnostic SimCutMix ranks 1-20 setup remains promising at
  `38.65%` test accuracy.
- Anchor-gated SimMixUp did not improve over ungated guided mixing in the tested
  configurations.
- Similarity-guided methods need multi-seed validation before making final
  claims.

## Known Gaps

- The k=20 same-budget comparison is incomplete locally: MixUp and CutMix
  summary/CSV files are missing, and SimCutMix k=20 currently exists only as a
  teammate-provided summary.
- The k=50 MixUp result needs reconciliation: the local canonical summary
  reports `24.08%` test accuracy, while the teammate-provided result reports
  `25.16%`.
- Eight historical summaries reference checkpoints that are not present locally,
  and one class-aware SimMixUp summary references a missing K20 neighbor file.
  The metric CSV/JSON pairs are present, but exact restoration is incomplete for
  those historical runs.
- Multi-seed aggregation is still missing.

## Next Tasks

1. Add or rerun the missing k=20 same-budget MixUp, CutMix, and SimCutMix runs.
2. Reconcile the k=50 MixUp discrepancy and decide which result is authoritative.
3. Build final same-budget tables for k=20, k=50, and k=100.
4. Run multi-seed aggregation for the final selected settings.
5. Generate final plots from the centralized experiment folder.

## Bottom Line

The experimental stage is close to a clean final comparison, but the k=20 gap,
k=50 MixUp discrepancy, and missing multi-seed aggregation should be resolved
before making final claims about the proposed method.
