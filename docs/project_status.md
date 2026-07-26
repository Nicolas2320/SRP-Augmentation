# SRP-Augmentation Project Status

Last verified: 2026-07-26

## Current Stage

The repository is in the focused experimental-comparison stage. The data,
training, standard augmentation, similarity-guided pairing, neighbor
construction, anchor scoring, and result-writing pipelines are implemented and
covered by tests.

The immediate research goal is to complete the scheduled-training baseline
matrix, preselect final configurations, and validate them across multiple
seeds. Historical no-LR-schedule experiments have been separated from the
active evidence set.

For setup and system orientation, use the
[project README](../README.md) and [architecture guide](architecture.md).

## Sources of Truth

| Source | Owns |
|---|---|
| `README.md` | Setup, entry points, and first-run instructions. |
| `docs/architecture.md` | System components and data flow. |
| `docs/reproducibility.md` | Reproduction requirements and limitations. |
| `docs/project_status.md` | Current work, known gaps, and next tasks. |
| `notes/current_results_summary.md` | Active-result interpretation. |
| `results/experiments/README.md` | Experiment-folder conventions. |
| `results/experiments/manifest.csv` | Generated sortable index of active run summaries. |

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
  generation, artifact auditing, and comparison plots.

## Active Evidence Inventory

The canonical `results/experiments/` tree contains:

- 20 complete `summary.json` and `metrics.csv` pairs;
- 8 ResNet50 scheduled-training runs and 12 ViT baseline runs;
- 6 k=100, 5 k=20, 5 k=50, and 4 k=450 runs;
- zero incomplete run folders or unmatched checkpoints requiring retention
  review.

The generated manifest contains 20 unique experiment IDs and indexes all
active summaries.

On 2026-07-26, 46 earlier ResNet50 runs without the current learning-rate
schedule, including K100 ablations and legacy runs, were moved to:

```text
../SRP-old_experiments/historical_no_lr_schedule/
```

The sibling archive is intentionally outside the Git repository. It contains a
46-run manifest, the original 66-run manifest, documentation snapshots, 52
locally available `.pt` payloads, and neighbor support used only by archived
experiments. The active cleanup ledger records the move and destination.

## Current Scientific Evidence

- Scheduled ResNet50 SimCutMix reaches `46.16%` test accuracy at k=100,
  compared with `43.20%` for scheduled CutMix.
- At k=450, scheduled CutMix reaches `72.70%`, compared with `71.60%` for
  SimCutMix. This negative proposal result remains part of the active evidence.
- The k=20 and k=50 SimCutMix results are `16.40%` and `28.65%`, respectively,
  but matching scheduled baselines have not yet been run.
- CutMix is the strongest recorded ViT baseline at k=20, k=50, and k=100.
- Similarity-guided methods still require multi-seed validation before final
  claims.

See [Current Experiment Results](../notes/current_results_summary.md) for the
active tables and missing comparison matrix.

## Known Research Gaps

- Scheduled none, MixUp, CutMix, and optionally AugMix baselines are missing at
  k=20 and k=50.
- Scheduled none, MixUp, and optionally AugMix baselines are missing at k=100.
- K450 AugMix is missing if AugMix remains in the final baseline set.
- Multi-seed aggregation is still missing.

## Known Reproducibility and Artifact Gaps

- Two active ViT AugMix summaries, at k=20 and k=50, reference checkpoints that
  are not present locally. Their metrics and summaries are complete.
- Large `.pt` artifacts are ignored by Git, so a fresh clone does not include
  local checkpoints, embeddings, or neighbor payloads.
- Existing summaries record experiment configuration but not the Git commit,
  Python version, PyTorch/CUDA versions, or GPU.
- `requirements.txt` is an install specification rather than an exact
  environment lock.
- Twenty-four ignored smoke, failed, incomplete, superseded, or unmatched
  checkpoints were permanently deleted in two reviewed batches on 2026-07-26.
  Their paths, sizes, and hashes are retained in
  `results/experiments/artifact_cleanup_log.json`.
- The current artifact audit reports zero retention-review candidates and zero
  noncanonical result locations.

## Next Research Tasks

1. Complete the scheduled-training baseline matrix.
2. Preselect one final proposal configuration per comparison budget using
   validation evidence rather than test-score cherry-picking.
3. Run multiple subset and training seeds.
4. Generate final aggregate tables and figures.

## Next Repository-Maintenance Tasks

1. Decide which final-run checkpoints need durable external storage.
2. Introduce a locked experiment environment and record environment metadata
   in future summaries.

## Bottom Line

The active result tree now contains only scheduled ResNet50 comparisons and the
retained ViT baseline grid. Final scientific claims depend on completing
matched baselines and running multi-seed comparisons. Environment capture
remains a limitation of exact reproduction.
