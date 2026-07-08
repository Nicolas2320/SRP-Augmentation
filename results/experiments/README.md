# Central Experiment Folder

This folder is the central home for experiment outputs.

All local experiment artifacts have been consolidated here. The top-level
`results/` folder should only contain this `experiments/` directory.

## Files

| File | Purpose |
|---|---|
| `manifest.csv` | Generated spreadsheet-friendly index of every local `*_summary.json` run. Useful for sorting/filtering experiments without opening every JSON file. |
| `README.md` | This folder map. |

Regenerate the manifest after adding new results:

```powershell
python src\experiments\build_manifest.py
```

## Collections

| Folder | Meaning |
|---|---|
| `results/experiments/baseline_100e/metrics` | Main 100-epoch baseline grid plus early guided runs. |
| `results/experiments/initial_simmixup_v2/metrics` | Initial 50-epoch ResNet50 k=100 baseline and SimMixUp runs. |
| `results/experiments/initial_simcutmix_v3/metrics` | Initial 50-epoch SimCutMix run. |
| `results/experiments/simmixup_ablation_v2` | Expanded SimMixUp ablations and neighbor/checkpoint artifacts. |
| `results/experiments/anchor_gated_v3` | Anchor-gated SimMixUp outputs. |
| `results/experiments/final_stage_v1` | Newer same-budget CutMix, k=50 comparison, and SimCutMix ablations. |
| `results/experiments/shared/neighbors` | Shared legacy neighbor artifacts. |
| `results/experiments/shared/anchor_scores` | Shared anchor-score artifacts. |
| `results/experiments/shared/checkpoints` | Checkpoints from older top-level runs that shared `results/checkpoints`. |
| `results/experiments/shared/figures` | Generated figures. |

Current interpretation, incomplete runs, and known missing historical artifacts
are tracked in `docs/project_status.md` and
`notes/experiment_results_summary_v1.md`.

## Recommended Future Layout

For new runs, use one named collection under this central folder:

```text
results/experiments/
  baseline_100e/
    metrics/
    checkpoints/
  final_stage_v1/
    metrics/
    checkpoints/
    neighbors/
  manual_runs/
    metrics/
    checkpoints/
```

Example future command:

```powershell
python src/train.py --dataset cifar100 --model resnet50 --k 20 --subset-seed 0 --augmentation cutmix --epochs 50 --output-root results/experiments/final_stage_v1
```

The default `src/train.py` output root is now
`results/experiments/manual_runs`, so ad hoc runs also stay inside the central
folder.

## Maintenance

After adding or moving experiment outputs, regenerate the manifest and confirm
the row count:

```powershell
python src\experiments\build_manifest.py
```
