# ResNet50 Comparison v1

Last verified: 2026-10-09.

This document defines the fixed command recipe implemented by the local
`scripts/run_resnet_comparison.ps1` runner and the cluster
`scripts/run_resnet_comparison.slurm` runner. It records where existing
evidence deviates from that recipe. It is a common-recipe comparison, not a
claim that the hyperparameters are optimal for every method or initialization.

## Scientific Scope

- Dataset: CIFAR-100.
- Classifier: standard torchvision ResNet50.
- Input: 224x224 with ImageNet normalization.
- Initializations: random (`scratch`) and ImageNet (`pretrained`).
- All classifier parameters are trainable.
- Subset seed: 0.
- Training seed: 0.
- Data budgets: k=20/50/100/450.
- Selection: best validation accuracy.
- Test: evaluated once from the selected checkpoint.

For guided methods, `scratch` describes classifier initialization only. The
partner-selection encoder is still a frozen ImageNet-pretrained ResNet50.

## Shared Training Recipe

| Setting | k=20/50/100 | k=450 |
|---|---:|---:|
| Epochs | 100 | 50 |
| Batch size | 32 | 32 |
| Optimizer | SGD | SGD |
| Learning rate | 0.01 | 0.1 |
| Momentum | 0.9 | 0.9 |
| Nesterov | Yes | Yes |
| Weight decay | 0.0005 | 0.0005 |
| LR milestones | 30/55/75 | 15/30/40 |
| LR gamma | 0.1 | 0.2 |

## Method Parameters

| Display label | CLI value | Fixed settings |
|---|---|---|
| Raw | `raw` | Deterministic resize/normalize only |
| Crop + flip | `none` | No additional method |
| MixUp | `mixup` | alpha=1, probability=1 |
| CutMix | `cutmix` | alpha=1, probability=0.5 |
| AugMix | `augmix` | severity=3, width=3, random depth, alpha=1; no JSD loss |
| SimMixUp | `simmixup` | alpha=1, guided probability=1, no warm-up |
| SimCutMix | `simcutmix` | alpha=1, guided probability=1, no warm-up |

All methods except `raw` include random crop and horizontal flip.

The runner currently requests class-agnostic K40 neighbors, ranks 21-40,
uniform sampling for both guided methods. Baseline CutMix mixes only half of
batches while SimCutMix mixes every batch, so that comparison changes both
partner selection and mixing frequency. It is not a pure pairing ablation.

## Execution

Run from the repository root. Omit `-Execute` to print the Python command
without training:

```powershell
.\scripts\run_resnet_comparison.ps1 -Initialization pretrained -K 20 -Method cutmix
.\scripts\run_resnet_comparison.ps1 -Initialization pretrained -K 20 -Method cutmix -Execute
```

Parameters:

- `-Initialization`: `scratch` or `pretrained`.
- `-K`: 20, 50, 100, or 450.
- `-Method`: `raw`, `none`, `mixup`, `cutmix`, `augmix`, `simmixup`, or
  `simcutmix`.

The full design is 56 cells (2 initializations x 4 budgets x 7 methods).

On the ISMLL cluster, submit from the repository root. Use `smoke` for a
one-epoch validation-only infrastructure check, or `full` for the recipe above:

```bash
sbatch scripts/run_resnet_comparison.slurm pretrained 20 none smoke
sbatch scripts/run_resnet_comparison.slurm pretrained 20 cutmix full
```

The smoke mode writes below `results/smoke/` and is not comparison evidence.
See [ISMLL cluster workflow](cluster_workflow.md) for setup, monitoring, and
artifact-handling instructions.

Guided execution currently requires local files such as:

```text
results/experiments/shared/neighbors/cifar100/k20_seed0/
  neighbors_class_agnostic_K40.pt
```

Those payloads are not present in the verified checkout and must be regenerated
before executing a guided command.

## Output Layout

```text
results/comparison_v1/<initialization>/cifar100/resnet50/k<k>/<method>/
  [class_agnostic_k40_r<window>/]
  e<epochs>_s0_t0_c<config-id>/
```

Each completed run should contain `metrics.csv`, `summary.json`, and a local
`checkpoint_best.pt`.

## Current Coverage

| Initialization | k=20 | k=50 | k=100 | k=450 | Total records |
|---|---:|---:|---:|---:|---:|
| Pretrained | 7/7 | 7/7 | 7/7 | 0/7 | 21 |
| Scratch group | 7/7 | 7/7 | 7/7 | 7/7 | 28 |

The scratch rows include imported historical records and are not all products
of the current 224x224 runner. See the cohort notes below.

## Evidence Deviations and Cohorts

### Imported historical scratch records

Most scratch ResNet50 results were created with an older CIFAR-adapted
32x32 ResNet50 (3x3 stride-1 stem, no initial max-pool, CIFAR normalization)
and later imported. Their summaries are marked
`cohort=historical_from_scratch`. Historical AugMix runs additionally retain
their original 100-epoch LR 0.1, milestones 30/60/80 recipe.

Current scratch records use the standard 224x224 architecture. The plotter
uses cohort and recipe fields to avoid matching or averaging incompatible
runs.

### Pretrained guided rank windows

The tracked pretrained evidence does not exactly match the runner for
SimMixUp:

- Pretrained SimCutMix at k=20/50/100 uses class-agnostic ranks 21-40.
- Pretrained SimMixUp at k=20/50/100 uses class-agnostic ranks 1-10.

Therefore, rerunning the current script with `-Method simmixup` would create a
different configuration (ranks 21-40), not reproduce the tracked pretrained
SimMixUp result. Use the recorded `summary.json` configuration when exact
reproduction is intended. This discrepancy should be resolved before the
multi-seed phase by either updating the runner to the selected method-specific
configuration or rerunning SimMixUp under the shared window.

## Figure Generation

```powershell
python src\graphs\plot_graphs.py --experiments-dir results\comparison_v1 --output-dir docs\figures
```

Outputs are replaced in `docs/figures/scratch/` and
`docs/figures/pretrained/`. Each contains:

- `runs.csv`;
- `matched_test_accuracy.png`;
- `matched_validation_curves.png`;
- `matched_overfitting_gap.png`;
- `available_test_accuracy_vs_k.png`;
- `experiment_coverage.png`.

The plotter keeps different initializations, architecture cohorts, and shared
training recipes separate. Method-specific settings are labeled but do not
make a run a repeated seed of another configuration.

## Reporting Rules

- Call every current value a single-run result.
- Report initialization and cohort with the number.
- Do not combine scratch and pretrained accuracies.
- Do not interpret epoch-to-epoch variation as seed uncertainty.
- Do not claim ranks 1-10 or 21-40 are globally optimal.
- Do not attribute CutMix/SimCutMix differences solely to partner selection
  while their application probabilities differ.
- Use validation evidence, not test accuracy, to select configurations for
  multi-seed confirmation.
