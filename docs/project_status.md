# SRP-Augmentation Project Status

Last verified: 2026-07-26

## Current Stage

The repository is in the focused experimental-comparison stage. The data,
training, standard augmentation, similarity-guided pairing, neighbor
construction, anchor scoring, and result-writing pipelines are implemented.

The immediate research goal is to reconcile the remaining result discrepancies,
select final same-budget configurations, and validate them across multiple
seeds. The immediate repository-maintenance goal is to consolidate historical
results and make the environment and artifact policy more reproducible.

For setup and system orientation, use the
[project README](../README.md) and [architecture guide](architecture.md).

## Sources of Truth

Use each document for one purpose:

| Source | Owns |
|---|---|
| `README.md` | Setup, entry points, and first-run instructions. |
| `docs/architecture.md` | System components and data flow. |
| `docs/reproducibility.md` | Reproduction requirements and limitations. |
| `docs/project_status.md` | Current work, known gaps, and next tasks. |
| `notes/experiment_results_summary_v1.md` | Detailed scientific result interpretation. |
| `results/experiments/README.md` | Experiment-folder conventions. |
| `results/experiments/manifest.csv` | Generated sortable index of run summaries. |

The manifest is derived data. Regenerate it after adding or moving canonical
summaries rather than editing it manually.

## Implemented Scope

- CIFAR-10 and CIFAR-100 k-shot splits with a fixed validation set.
- ResNet50 and ViT training through `src/train.py`.
- Standard methods: none, MixUp, CutMix, and AugMix.
- Proposed methods: SimMixUp and SimCutMix.
- Class-aware, class-agnostic, and different-label neighbor modes.
- Neighbor rank windows, uniform or weighted pairing, mix probability, and
  warmup.
- Optional anchor-score gating and dynamic neighbor pools.
- Canonical metrics, summaries, best-validation checkpoints, manifest
  generation, and comparison plots.

## Repository Evidence Inventory

The canonical `results/experiments/` tree currently contains:

- 66 complete `summary.json` and `metrics.csv` pairs;
- 54 ResNet50 runs and 12 ViT runs;
- 38 k=100, 12 k=20, 12 k=50, and 4 k=450 runs;
- 8 runs in the current `standard_cifar_recipe` layout;
- 58 runs in the earlier canonical layout, including 3 explicitly retained
  `legacy/` runs.

The generated manifest contains 66 unique experiment IDs and indexes all
current canonical summaries.

The two former `experiments_v2` different-label SimMixUp runs were consolidated
under the canonical k=100 SimMixUp tree on 2026-07-26. Their original output
root is retained in each configuration, and explicit consolidation metadata
records the previous summary path. Their matching checkpoints were verified
against the stored configuration, epoch, and validation score before being
placed beside the canonical summaries.

## Current Scientific Evidence

The following observations come from
`notes/experiment_results_summary_v1.md`. That summary predates some of the
newer standard-recipe additions, so it remains the versioned interpretation
rather than a claim that every current run has already been re-analysed.

- CutMix was the strongest standard baseline in the documented 100-epoch
  baseline grid.
- The best documented guided result was SimCutMix on CIFAR-100, ResNet50,
  k=100, class-agnostic K40 ranks 21–40, with `40.02%` test accuracy.
- The repeated class-agnostic SimCutMix ranks 1–20 setup remained promising at
  `38.65%` test accuracy.
- The documented same-budget k=20 and k=50 SimCutMix comparisons were positive:
  `16.49%` versus `13.36%` CutMix at k=20, and `28.04%` versus `26.07%`
  CutMix at k=50.
- Anchor-gated SimMixUp did not improve over ungated guided mixing in the tested
  configurations.
- Similarity-guided methods still require multi-seed validation before final
  claims.

## Known Research Gaps

- The k=20 local SimCutMix result and the teammate-provided summary disagree:
  the local canonical file reports `16.49%` test accuracy, while the teammate
  summary reports `18.14%`.
- The k=50 MixUp result needs reconciliation: the local canonical summary
  reports `24.08%`, while the teammate-provided result reports `25.16%`.
- Multi-seed aggregation is still missing.
- The detailed results summary has not yet incorporated and interpreted the
  eight newer standard-recipe runs.

## Known Reproducibility and Artifact Gaps

- Eight historical summaries reference checkpoints that are not present
  locally.
- One class-aware SimMixUp summary references a missing K20 neighbor payload.
- Large `.pt` artifacts are ignored by Git, so a fresh clone does not include
  local checkpoints, embeddings, or neighbor payloads.
- Historical summaries record experiment configuration but not the Git commit,
  Python version, PyTorch/CUDA versions, or GPU.
- `requirements.txt` is an install specification rather than an exact
  environment lock.
- Some result paths exceed the traditional Windows 260-character limit.
- The local artifact audit identifies 14 checkpoints requiring a retention
  decision: 9 unreferenced shared historical checkpoints (`2.39 GiB`) and 5
  checkpoints from incomplete run folders (`1.24 GiB`).
- Ten high-confidence smoke or failed checkpoints (`2.04 GiB`) were deleted on
  2026-07-26 after path, configuration, size, and hash validation. Their
  provenance is retained in
  `results/experiments/artifact_cleanup_log.json`.

The CSV and JSON records for affected historical runs are preserved. Missing
support artifacts should be reported explicitly rather than silently
reconstructed or substituted.

## Next Research Tasks

1. Reconcile the k=20 SimCutMix and k=50 MixUp discrepancies.
2. Incorporate the standard-recipe runs into the detailed result narrative.
3. Build final same-budget tables for k=20, k=50, and k=100.
4. Select final configurations and run multiple subset and training seeds.
5. Generate final aggregate tables and figures.

## Next Repository-Maintenance Tasks

1. Review the 14 checkpoint retention candidates before archiving or deleting
   anything.
2. Decide which final-run checkpoints need durable external storage.
3. Introduce a locked experiment environment and record environment metadata
   in future summaries.
4. Shorten future run paths while preserving historical records.

## Bottom Line

The code path is implemented and covered by tests, and the repository has a
usable single-seed evidence base. Final scientific claims depend on resolving
the documented discrepancies and running multi-seed comparisons. Historical
artifact availability and environment capture remain limitations of exact
reproduction.
