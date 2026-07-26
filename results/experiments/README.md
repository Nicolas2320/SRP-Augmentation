# Experiment Folder Guide

`results/experiments/` is the canonical home for experiment records and shared
support artifacts.

For scientific interpretation, read
[`notes/experiment_results_summary_v1.md`](../../notes/experiment_results_summary_v1.md).
For known discrepancies and maintenance work, read
[`docs/project_status.md`](../../docs/project_status.md).

## Quick Navigation

| Location | Purpose |
|---|---|
| `manifest.csv` | Generated spreadsheet-friendly index of canonical summaries. |
| `artifact_cleanup_log.json` | Provenance ledger for intentionally removed local artifacts. |
| `cifar100/` | Dataset/model/k/method run records. |
| `shared/neighbors/` | Reusable embeddings, neighbor payloads, and metadata. |
| `shared/anchor_scores/` | Anchor uncertainty and rarity scores. |
| `shared/checkpoints/` | Historical or unmatched local checkpoints. |
| `shared/figures/` | Generated comparison figures. |

The two result pairs formerly stored below `results/experiments_v2/` were
consolidated into the canonical k=100 SimMixUp tree on 2026-07-26. Their
summaries retain explicit provenance fields for the previous locations.

## Completed Run Contract

A completed canonical run has:

```text
<run-directory>/
  metrics.csv
  summary.json
  checkpoint_best.pt
```

| File | Contents | Git policy |
|---|---|---|
| `metrics.csv` | Per-epoch learning rate, loss, and accuracy. | Commit for evidence. |
| `summary.json` | Run configuration, best epoch, and final test evaluation. | Commit for evidence. |
| `checkpoint_best.pt` | Best-validation model, optimizer, and scheduler state. | Local/ignored. |

A run can still be inspected when its historical checkpoint is unavailable,
provided its metrics and summary pair is intact. Such missing artifacts must be
documented in `docs/project_status.md`.

## Current Output Layout

The current `src/train.py` writes a recipe-aware path:

```text
results/experiments/
  <dataset>/
    <model>/
      k<k>/
        standard_cifar_recipe/
          <optimizer-batch-lr-weight-decay-schedule>/
            baselines/
              <augmentation>/
                e<epochs>_s<subset-seed>_t<train-seed>/
            simmixup/
              <guided-settings>/
                e<epochs>_s<subset-seed>_t<train-seed>/
            simcutmix/
              <guided-settings>/
                e<epochs>_s<subset-seed>_t<train-seed>/
```

The recipe component makes optimizer and schedule differences visible instead
of allowing incompatible runs to share a folder.

## Earlier Canonical Layout

Runs created before the recipe-aware output change remain valid in the shorter
layout:

```text
results/experiments/
  <dataset>/
    <model>/
      k<k>/
        baselines/
        simmixup/
        simcutmix/
        legacy/
```

Do not move these runs merely to make the tree visually uniform. Their paths
are referenced by summaries, notes, and the manifest. Any future migration
should update those references together and verify the resulting catalog.

Use `legacy/` only for non-exact historical reruns retained for provenance.
Exact or equivalent duplicates should be classified during artifact cleanup
before any deletion.

## Guided Setting Components

Guided run paths encode the settings most useful during browsing:

```text
<guided-mode>_K<saved-neighbor-count>/
  r<first-rank>-<last-rank>/
    <sampling>_nk<count>_a<alpha>_mp<mix-probability>_w<warmup>/
```

Anchor-gated SimMixUp adds an anchor-selection component. The full
configuration remains authoritative in `summary.json`; folder names are a
navigation aid.

## Shared Artifacts

| Folder | Meaning |
|---|---|
| `shared/neighbors/` | Embeddings and filtered neighbor payloads reused by guided methods. |
| `shared/anchor_scores/` | Per-anchor scores used by targeted mixing. |
| `shared/checkpoints/` | Historical local checkpoints not stored beside a canonical summary. |
| `shared/checkpoints/unmatched/` | Checkpoints without a matching local CSV/summary pair. |
| `shared/figures/` | Generated plots from `src/graphs/plot_graphs.py`. |

Large `.pt` files and `.png` figures are ignored by Git. Metadata JSON and
small result records may be committed so their origin and expected paths remain
visible.

## Manifest

`manifest.csv` is generated from canonical `summary.json` files and is intended
for filtering and comparison without opening every directory.

Regenerate it after adding or moving run records:

```powershell
python src\experiments\build_manifest.py
```

As of the 2026-07-26 result consolidation, the canonical tree and manifest both
contain 66 experiments with unique IDs.

Do not edit the manifest manually. Correct the source summary or the manifest
builder instead.

## Artifact Audit

Run the read-only artifact audit with:

```powershell
python src\experiments\audit_artifacts.py --details
```

It checks:

- canonical summary/metrics pairs;
- paths recorded by summaries and shared metadata;
- local `.pt` and `.pth` classification;
- checkpoint files that need a manual retention decision;
- result files outside the canonical root.

The default command only prints findings. `--json-output <path>` can save a
machine-readable local snapshot; the audit never deletes or moves artifacts.

## Figures

Generate the standard comparison figures with:

```powershell
python src\graphs\plot_graphs.py
```

The script reads summaries and epoch metrics, excludes explicit `legacy/` runs
where applicable, and writes figures under `shared/figures/`.

## Maintenance Checklist

When adding a completed run:

1. Confirm that `metrics.csv` and `summary.json` are both present.
2. Check that paths recorded inside `summary.json` point to the intended run and
   support artifacts.
3. Preserve the subset seed, training seed, and full recipe.
4. Regenerate `manifest.csv`.
5. Update the scientific result narrative only after validating the comparison
   budget and seed policy.
6. Document missing checkpoints or neighbor payloads.

When reorganizing historical results, use a separate cleanup change and verify
every reference before deleting or moving artifacts.
