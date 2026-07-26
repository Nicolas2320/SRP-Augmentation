# Experiment Folder Guide

`results/experiments/` is the canonical home for experiment records and shared
support artifacts.

For scientific interpretation, read
[`notes/current_results_summary.md`](../../notes/current_results_summary.md).
For known discrepancies and maintenance work, read
[`docs/project_status.md`](../../docs/project_status.md).

## Quick Navigation

| Location | Purpose |
|---|---|
| `manifest.csv` | Generated spreadsheet-friendly index of canonical summaries. |
| `artifact_cleanup_log.json` | Provenance ledger for intentionally removed local artifacts. |
| `cifar100/` | Dataset/model/k/method run records. |
| `shared/neighbors/` | Reusable embeddings, neighbor payloads, and metadata. |
| `shared/figures/` | Generated comparison figures; created only when plotting is run. |

Earlier no-LR-schedule ResNet50 runs, K100 ablations, legacy records, and anchor
scores were moved to the external `historical_no_lr_schedule` archive on
2026-07-26. The move and destination are recorded in
`artifact_cleanup_log.json`.

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

The current `src/train.py` writes a concise, collision-safe path:

```text
results/experiments/
  <dataset>/
    <model>/
      k<k>/
        <method>/
          <guided-mode>_k<saved-neighbors>_r<rank-window>/  # guided only
            e<epochs>_s<subset-seed>_t<train-seed>_c<config-id>/
```

Baseline methods omit the guided component. The eight-character config ID is
derived from all scientific configuration fields and prevents different
optimizer, schedule, or augmentation settings from overwriting one another.
The complete readable configuration remains in `summary.json`.

## Earlier Records and Archive

The 12 retained ViT baseline records predate the concise layout and remain
under:

```text
results/experiments/cifar100/vit/k<k>/baselines/<method>/e<epochs>_s0_t0/
```

Their folders are already short, and their full recorded configuration remains
in each summary. Earlier no-LR-schedule ResNet50 runs are no longer part of the
active canonical tree. They are preserved under
`SRP-old_experiments/historical_no_lr_schedule`, which contains a 46-run
archive manifest, the original 66-run manifest, documentation snapshots, and
archived-only neighbor support. Do not mix archived results with active
scheduled-training results unless the comparison explicitly uses the same
recipe.

## Guided Setting Components

Guided run paths encode the settings most useful during browsing:

```text
<guided-mode>_k<saved-neighbor-count>_r<first-rank>-<last-rank>/
```

Sampling, alpha, mix probability, warmup, anchor gating, and dynamic-pool
settings remain authoritative in `summary.json` and are covered by the config
ID. Folder names are a navigation aid.

## Shared Artifacts

| Folder | Meaning |
|---|---|
| `shared/neighbors/` | Embeddings and filtered neighbor payloads reused by guided methods. |
| `shared/figures/` | Generated plots from `src/graphs/plot_graphs.py`; absent until regenerated. |

Large `.pt` files and `.png` figures are ignored by Git. Metadata JSON and
small result records may be committed so their origin and expected paths remain
visible.

## Manifest

`manifest.csv` is generated from canonical `summary.json` files and is intended
for filtering and comparison without opening every directory. Its `config_id`
column matches the suffix used by concise-layout runs.

Regenerate it after adding or moving run records:

```powershell
python src\experiments\build_manifest.py
```

As of the 2026-07-26 archive cleanup, the canonical tree and manifest both
contain 20 experiments with unique IDs: 8 scheduled-training ResNet50 runs and
12 ViT baseline runs.

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

The script reads the active summaries and epoch metrics and writes figures
under `shared/figures/`.

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
