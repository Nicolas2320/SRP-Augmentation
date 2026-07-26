# Current Experiment Results

Last verified: 2026-07-26

## Scope

The active manifest contains 20 complete CIFAR-100 experiments:

- 8 ResNet50 runs using the current scheduled recipe and concise run layout;
- 12 ViT baseline runs covering none, MixUp, CutMix, and AugMix at k=20,
  k=50, and k=100.

Earlier ResNet50 runs without the current learning-rate schedule, K100
ablations, and legacy results are stored outside the active project at:

```text
../SRP-old_experiments/historical_no_lr_schedule/
```

That sibling directory is intentionally outside the Git repository. It contains
its own 46-run manifest and the original 66-run manifest.

## Scheduled ResNet50 Results

| k | Method | Epochs | Best validation | Test |
|---:|---|---:|---:|---:|
| 20 | SimCutMix | 100 | 16.32% | 16.40% |
| 50 | SimCutMix | 100 | 29.64% | 28.65% |
| 100 | CutMix | 100 | 44.06% | 43.20% |
| 100 | SimCutMix | 100 | 46.40% | 46.16% |
| 450 | None | 50 | 71.08% | 71.07% |
| 450 | MixUp | 50 | 68.40% | 69.04% |
| 450 | CutMix | 50 | 72.58% | 72.70% |
| 450 | SimCutMix | 50 | 71.48% | 71.60% |

At k=100, SimCutMix is 2.96 percentage points above the scheduled CutMix
result. At k=450, CutMix is 1.10 points above SimCutMix. Both outcomes should
remain visible; proposal results are not removed merely because a baseline is
stronger.

The k=20 and k=50 SimCutMix results do not yet have matching scheduled
baselines. They must not be compared directly with the archived no-schedule
baseline grid.

## ViT Baseline Results

| k | None | MixUp | CutMix | AugMix |
|---:|---:|---:|---:|---:|
| 20 | 9.44% | 13.14% | 13.97% | 12.57% |
| 50 | 16.24% | 20.36% | 20.75% | 19.29% |
| 100 | 22.00% | 26.17% | 27.69% | 23.21% |

CutMix is the strongest recorded ViT baseline at all three k values.

## Missing Comparisons

For a complete scheduled ResNet50 comparison matrix, run:

- none, MixUp, CutMix, and optionally AugMix at k=20 and k=50;
- none, MixUp, and optionally AugMix at k=100;
- AugMix at k=450 only if it remains part of the final baseline set.

All active results currently use one subset seed and one training seed. Final
claims require preselected configurations and multiple seeds.
