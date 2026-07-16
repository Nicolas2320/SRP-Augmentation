# Central Experiment Folder

This folder is the central home for experiment outputs.

All local experiment artifacts have been consolidated here. The top-level
`results/` folder should only contain this `experiments/` directory.

## Files

| File | Purpose |
|---|---|
| `manifest.csv` | Generated spreadsheet-friendly index of every local `summary.json` run. Useful for sorting/filtering experiments without opening every JSON file. |
| `README.md` | This folder map. |

Regenerate the manifest after adding new results:

```powershell
python src\experiments\build_manifest.py
```

## Layout

Canonical runs are organized by dataset, model, k-shot setting, and method:

```text
results/experiments/
  cifar100/
    resnet50/
      k100/
        baselines/
          cutmix/e50_s0_t0/
            metrics.csv
            summary.json
            checkpoint_best.pt
        simcutmix/
        simmixup/
          ungated/
          anchor_gated/
        legacy/
    vit/
  shared/
```

Use `legacy/` only for non-exact reruns retained for provenance. Exact or
equivalent duplicates should be deduplicated.

Shared artifacts live under:

| Folder | Meaning |
|---|---|
| `results/experiments/shared/neighbors` | Neighbor payloads reused by guided methods. |
| `results/experiments/shared/anchor_scores` | Anchor-score artifacts. |
| `results/experiments/shared/checkpoints/unmatched` | Local checkpoints that do not currently have a matching CSV/summary pair. |
| `results/experiments/shared/figures` | Generated figures. |

Current interpretation, incomplete runs, and known missing historical artifacts
are tracked in `docs/project_status.md` and
`notes/experiment_results_summary_v1.md`.

Example future command:

```powershell
python src/train.py --dataset cifar100 --model resnet50 --k 20 --subset-seed 0 --augmentation cutmix --epochs 50
```

The default `src/train.py` output root is `results/experiments`. The script
builds the method/k-specific subfolder automatically.

## Maintenance

After adding or moving experiment outputs, regenerate the manifest and confirm
the row count:

```powershell
python src\experiments\build_manifest.py
```
