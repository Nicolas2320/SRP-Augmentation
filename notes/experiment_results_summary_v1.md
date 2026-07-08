# Experiment Results Summary v1

Generated: 2026-07-08

Branch: `summary-v1`

This note consolidates the experiment result files currently present under
`results/`. It expands the earlier `notes/experiments_v2_results.md` note and
keeps each result source visible, because several nominally similar runs appear
in more than one metrics folder.

## Scope and Method

- Result sources: `*_summary.json` files plus their matching epoch-level CSV
  files.
- Complete metric pairs found: 55 summary JSON files and 55 matching CSV files.
- Missing summary/CSV pairs: none.
- Anchor-score support file found: 1 CSV under `results/experiments/shared/anchor_scores/`.
- Reported `train_acc`, `train_loss`, and `val_loss` are taken from the CSV row
  at the JSON `best_epoch`.
- Reported `val_acc`, `test_acc`, and `test_loss` are taken from the JSON
  summary, where `test_acc` is the test evaluation of the best-validation
  checkpoint.
- `gap` means `train_acc - val_acc` at the best-validation epoch.
- All complete summaries are CIFAR-100, `subset_seed=0`, and `train_seed=0`.

Important caveat: these are still single-seed results. They are good for
preliminary direction finding, but not enough for final claims without multi-seed
aggregation.

## Experiment Folders

| Folder | Role | Complete runs |
|---|---|---:|
| `results/experiments/baseline_100e/metrics` | Main 100-epoch baseline grid plus one early SimMixUp run | 25 |
| `results/experiments/initial_simmixup_v2/metrics` | Initial 50-epoch ResNet50 k=100 baseline and SimMixUp runs | 4 |
| `results/experiments/initial_simcutmix_v3/metrics` | Initial 50-epoch SimCutMix run | 1 |
| `results/experiments/simmixup_ablation_v2/metrics` | Expanded 50-epoch SimMixUp ablations | 13 |
| `results/experiments/anchor_gated_v3/metrics` | Anchor-gated 50-epoch SimMixUp sweep | 5 |
| `results/experiments/final_stage_v1/metrics` | New same-budget CutMix, k=50 comparison, and SimCutMix ablations | 7 |
| `results/experiments/shared/anchor_scores` | Anchor uncertainty/rarity score CSV used for targeted mixing | 1 support file |

## Final-Stage Completion Check

The planned final-stage command set was partly completed. The result files show
that the k=100 same-budget CutMix run, the k=50 comparison, and the small k=100
SimCutMix ablation were completed. The k=20 comparison is still missing.

| Planned experiment | Status | Evidence |
|---|---|---|
| k=100 CutMix, 50 epochs | complete | `results/experiments/final_stage_v1/metrics/cifar100_resnet50_k100_seed0_cutmix_epochs50_summary.json` |
| k=100 SimCutMix K40 ranks 1-20 repeat | exact final-stage K40 file missing | K60 ranks 1-20 exists and has the same first 20 neighbors as the older K40 file, so it is effectively equivalent for this rank window |
| k=20 MixUp, 50 epochs | missing | no `results/experiments/final_stage_v1/metrics/cifar100_resnet50_k20_seed0_mixup_epochs50_summary.json` |
| k=20 CutMix, 50 epochs | missing | no `results/experiments/final_stage_v1/metrics/cifar100_resnet50_k20_seed0_cutmix_epochs50_summary.json` |
| k=20 SimCutMix, 50 epochs | teammate summary only | teammate reports `18.14%` test accuracy, but no local summary JSON or CSV exists under `results/experiments/final_stage_v1/metrics` |
| k=50 MixUp, 50 epochs | complete, with discrepancy | local file reports `24.08%` test accuracy; teammate summary for the same path reports `25.16%` |
| k=50 CutMix, 50 epochs | complete | `results/experiments/final_stage_v1/metrics/cifar100_resnet50_k50_seed0_cutmix_epochs50_summary.json` |
| k=50 SimCutMix K60 ranks 1-20, 50 epochs | complete | `results/experiments/final_stage_v1/metrics/cifar100_resnet50_k50_seed0_simcutmix_class_agnostic_neighbors_class_agnostic_K60_nk20_r1-20_uniform_alpha1_mp1_warm0_epochs50_summary.json` |
| k=100 SimCutMix K40 ranks 21-40 | complete | best current guided run, `40.02%` test accuracy |
| k=100 SimCutMix K60 ranks 1-20 | complete | effectively repeats the K40 ranks 1-20 setup, `38.65%` test accuracy |
| k=100 SimCutMix K60 ranks 21-40 | complete | `38.34%` test accuracy |

## Teammate-Provided External Summaries

These summaries were shared directly by a teammate and are not yet counted in
the local `55` complete JSON/CSV metric pairs above. They are useful for
orientation, but the local note should treat them as provisional until the
matching summary JSON and CSV files are added or the local files are reconciled.

| Run | Teammate result | Local status | Action needed |
|---|---:|---|---|
| k=20 SimCutMix K60 ranks 1-20, 50 epochs | test `18.14%`, val `16.58%`, best epoch 43 | no local summary JSON or CSV found | add the teammate's summary JSON and matching CSV, or rerun locally |
| k=50 MixUp, 50 epochs | test `25.16%`, val `24.98%`, best epoch 37 | local summary for the same metrics path reports test `24.08%`, val `24.88%`, best epoch 40 | reconcile which file is authoritative; replace/update local files only if the teammate run is the intended result |

## High-Level Findings

CutMix is still the strongest standard 100-epoch baseline in every baseline
cell. This holds for both ResNet50 and ViT and for k = 20, 50, and 100.

The best guided result is now SimCutMix K40 ranks 21-40 on CIFAR-100 ResNet50
k=100 after 50 epochs. It reaches `40.02%` test accuracy, which is `+3.38 pp`
over the 50-epoch MixUp baseline, `+2.45 pp` over the new same-budget 50-epoch
CutMix baseline, and `+1.27 pp` over the 100-epoch CutMix baseline. This is the
clearest positive result so far for similarity-guided mixing.

The new 50-epoch CutMix baseline matters: ResNet50 k=100 CutMix reaches
`37.57%` at 50 epochs, below its older 100-epoch result of `38.75%`. This makes
the new SimCutMix K40 ranks 21-40 result stronger than the older comparison,
because it beats CutMix at the same epoch budget rather than merely approaching a
longer-budget baseline.

The k=50 final-stage comparison is also positive for SimCutMix. At k=50,
SimCutMix K60 ranks 1-20 reaches `28.04%`, compared with `26.07%` for CutMix and
`24.08%` for MixUp under the same 50-epoch budget. This suggests the effect may
not be limited to k=100. For k=20, there is a teammate-provided SimCutMix result
(`18.14%`), but the local MixUp/CutMix baselines and matching SimCutMix files
are still missing.

Class-agnostic SimMixUp remains mostly close to MixUp. Across the 50-epoch
ResNet50 k=100 runs, most SimMixUp variants sit around `35.5%` to `36.9%` test
accuracy. The best SimMixUp result is still `37.69%` with class-agnostic K60
neighbors and ranks 21-40, which is useful but weaker than the new SimCutMix
results.

Class-aware SimMixUp performs poorly in the current result set. The class-aware
K20 run reaches only `31.75%`, which is `-4.89 pp` below the 50-epoch MixUp
baseline. In these runs, forcing same-class neighbors appears to reduce useful
diversity rather than improve label consistency.

Anchor-gated SimMixUp did not help in the tested configuration. The five
anchor-gated runs all underperform the 50-epoch MixUp baseline and the strongest
ungated SimMixUp run. Softer score-probability gating is less harmful than hard
top-fraction gating, but none of the tested anchor settings improves the result.

The recurring pattern is now more specific: SimMixUp parameter changes mostly
cluster near MixUp, while SimCutMix rank-window choice appears much more
consequential. The K40 ranks 21-40 SimCutMix run is the first result that clearly
moves beyond the baseline cluster.

## 100-Epoch Baseline Grid

Values are test accuracy percentages from the best-validation checkpoint.

| Model | k | None | MixUp | CutMix | AugMix | Best |
|---|---:|---:|---:|---:|---:|---|
| resnet50 | 20 | 11.75 | 14.68 | 15.46 | 13.55 | cutmix (15.46) |
| resnet50 | 50 | 22.32 | 25.62 | 27.04 | 26.69 | cutmix (27.04) |
| resnet50 | 100 | 31.88 | 36.67 | 38.75 | 37.15 | cutmix (38.75) |
| vit | 20 | 9.44 | 13.14 | 13.97 | 12.57 | cutmix (13.97) |
| vit | 50 | 16.24 | 20.36 | 20.75 | 19.29 | cutmix (20.75) |
| vit | 100 | 22.00 | 26.17 | 27.69 | 23.21 | cutmix (27.69) |

Baseline interpretation:

- ResNet50 is stronger than ViT in these from-scratch, low-data CIFAR-100 runs.
- No-augmentation runs overfit heavily, especially for ResNet50.
- MixUp improves consistently over no augmentation and substantially reduces the
  generalization gap.
- CutMix is consistently the best baseline and usually has the smallest gap,
  especially for ViT.
- AugMix helps over no augmentation for larger k values, but it does not beat
  CutMix and often keeps a large train-validation gap.

## 100-Epoch Baseline Details

| Source | Model | k | Aug. | Best epoch | Train acc | Val acc | Test acc | Test loss | Gap |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| metrics | resnet50 | 20 | none | 70 | 98.45 | 13.30 | 11.75 | 7.9251 | 85.15 |
| metrics | resnet50 | 20 | mixup | 46 | 76.47 | 15.28 | 14.68 | 4.4190 | 61.19 |
| metrics | resnet50 | 20 | cutmix | 87 | 73.79 | 16.12 | 15.46 | 3.9728 | 57.67 |
| metrics | resnet50 | 20 | augmix | 99 | 93.40 | 14.46 | 13.55 | 5.6774 | 78.94 |
| metrics | resnet50 | 50 | none | 80 | 99.94 | 22.22 | 22.32 | 6.9225 | 77.72 |
| metrics | resnet50 | 50 | mixup | 50 | 70.18 | 25.06 | 25.62 | 3.3042 | 45.12 |
| metrics | resnet50 | 50 | cutmix | 93 | 73.70 | 27.16 | 27.04 | 3.2143 | 46.54 |
| metrics | resnet50 | 50 | augmix | 75 | 96.14 | 26.64 | 26.69 | 4.7051 | 69.50 |
| metrics | resnet50 | 100 | none | 96 | 99.73 | 32.46 | 31.88 | 5.9852 | 67.27 |
| metrics | resnet50 | 100 | mixup | 29 | 68.67 | 36.58 | 36.67 | 2.6481 | 32.09 |
| metrics | resnet50 | 100 | cutmix | 33 | 60.34 | 39.74 | 38.75 | 2.5208 | 20.60 |
| metrics | resnet50 | 100 | augmix | 87 | 97.27 | 39.24 | 37.15 | 3.8504 | 58.03 |
| metrics | vit | 20 | none | 61 | 93.00 | 10.10 | 9.44 | 6.9626 | 82.90 |
| metrics | vit | 20 | mixup | 89 | 50.59 | 14.04 | 13.14 | 4.2252 | 36.55 |
| metrics | vit | 20 | cutmix | 100 | 29.31 | 14.20 | 13.97 | 3.9351 | 15.11 |
| metrics | vit | 20 | augmix | 88 | 76.30 | 12.32 | 12.57 | 5.5058 | 63.98 |
| metrics | vit | 50 | none | 22 | 23.88 | 16.58 | 16.24 | 3.6123 | 7.30 |
| metrics | vit | 50 | mixup | 98 | 62.21 | 20.42 | 20.36 | 3.7218 | 41.79 |
| metrics | vit | 50 | cutmix | 96 | 26.93 | 20.86 | 20.75 | 3.4192 | 6.07 |
| metrics | vit | 50 | augmix | 98 | 88.24 | 18.56 | 19.29 | 5.1023 | 69.68 |
| metrics | vit | 100 | none | 97 | 96.21 | 23.08 | 22.00 | 7.0647 | 73.13 |
| metrics | vit | 100 | mixup | 100 | 57.42 | 25.78 | 26.17 | 3.3269 | 31.64 |
| metrics | vit | 100 | cutmix | 88 | 34.41 | 27.66 | 27.69 | 3.0426 | 6.75 |
| metrics | vit | 100 | augmix | 95 | 86.81 | 22.74 | 23.21 | 4.7199 | 64.07 |

## 50-Epoch ResNet50 k=100 Guided and Reference Runs

The table below ranks all 50-epoch CIFAR-100 ResNet50 k=100 runs by test
accuracy. Deltas use the best available 50-epoch MixUp result, `36.64%`, and
the new 50-epoch CutMix result, `37.57%`, as reference points.

| Rank | Source | Method | Setting | Best epoch | Train acc | Val acc | Test acc | Test loss | Gap | Delta vs 50e MixUp | Delta vs 50e CutMix |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | final_stage | simcutmix | class-agnostic; class_agnostic_K40; ranks 21-40; nk=20; alpha=1; mix_prob=1; warmup=0 | 27 | 72.97 | 41.36 | 40.02 | 2.3745 | 31.61 | +3.38 pp | +2.45 pp |
| 2 | final_stage | simcutmix | class-agnostic; class_agnostic_K60; ranks 1-20; nk=20; alpha=1; mix_prob=1; warmup=0 | 40 | 79.54 | 39.20 | 38.65 | 2.5572 | 40.34 | +2.01 pp | +1.08 pp |
| 3 | metrics_v3 | simcutmix | class-agnostic; class_agnostic_K40; ranks 1-20; nk=20; alpha=1; mix_prob=1; warmup=0 | 40 | 79.54 | 39.20 | 38.65 | 2.5572 | 40.34 | +2.01 pp | +1.08 pp |
| 4 | final_stage | simcutmix | class-agnostic; class_agnostic_K60; ranks 21-40; nk=20; alpha=1; mix_prob=1; warmup=0 | 34 | 75.01 | 38.84 | 38.34 | 2.4457 | 36.17 | +1.70 pp | +0.77 pp |
| 5 | experiments_v2 | simmixup | class-agnostic; class_agnostic_K60; ranks 21-40; nk=20; alpha=1; mix_prob=1; warmup=0 | 28 | 78.66 | 38.14 | 37.69 | 2.6007 | 40.52 | +1.05 pp | +0.12 pp |
| 6 | final_stage | cutmix | standard baseline | 44 | 71.36 | 38.04 | 37.57 | 2.6396 | 33.32 | +0.93 pp | +0.00 pp |
| 7 | metrics_v2 | simmixup | class-agnostic; class_agnostic_K20; ranks 1-20; nk=20; alpha=1; mix_prob=1; warmup=0 | 39 | 80.73 | 37.30 | 37.41 | 2.5371 | 43.43 | +0.77 pp | -0.16 pp |
| 8 | experiments_v2 | simmixup | class-agnostic; class_agnostic_K40; ranks 21-40; nk=20; alpha=1; mix_prob=1; warmup=0 | 28 | 78.48 | 37.44 | 36.92 | 2.5621 | 41.04 | +0.28 pp | -0.65 pp |
| 9 | metrics | simmixup | class-agnostic; class_agnostic_K40; ranks 21-40; nk=20; alpha=1; mix_prob=1; warmup=0 | 39 | 79.25 | 36.92 | 36.73 | 2.7082 | 42.33 | +0.09 pp | -0.84 pp |
| 10 | experiments_v2 | mixup | standard baseline | 28 | 66.59 | 36.54 | 36.64 | 2.6172 | 30.05 | +0.00 pp | -0.93 pp |
| 11 | metrics_v2 | mixup | standard baseline | 28 | 66.59 | 36.54 | 36.64 | 2.6172 | 30.05 | +0.00 pp | -0.93 pp |
| 12 | experiments_v2 | simmixup | class-agnostic; class_agnostic_K60; ranks 41-60; nk=20; alpha=1; mix_prob=1; warmup=0 | 28 | 76.70 | 36.68 | 36.57 | 2.5456 | 40.02 | -0.07 pp | -1.00 pp |
| 13 | experiments_v2 | simmixup | class-agnostic; class_agnostic_K40; ranks 21-40; nk=20; alpha=1; mix_prob=1; warmup=5 | 46 | 81.37 | 36.02 | 36.41 | 2.8019 | 45.35 | -0.23 pp | -1.16 pp |
| 14 | experiments_v2 | simmixup | class-agnostic; class_agnostic_K40; ranks 21-40; nk=20; alpha=1; mix_prob=0.5; warmup=0 | 30 | 89.54 | 36.58 | 36.35 | 2.7392 | 52.96 | -0.29 pp | -1.22 pp |
| 15 | experiments_v2 | simmixup | class-agnostic; class_agnostic_K40; ranks 21-40; nk=20; alpha=0.7; mix_prob=1; warmup=0 | 49 | 83.54 | 36.92 | 36.32 | 2.5538 | 46.62 | -0.32 pp | -1.25 pp |
| 16 | experiments_v3 | simmixup | class-agnostic; class_agnostic_K60; ranks 21-40; nk=20; alpha=1; mix_prob=1; warmup=0; anchor=score_probability; top=0.2; power=0.5 | 34 | 85.08 | 36.42 | 36.20 | 2.7696 | 48.66 | -0.44 pp | -1.37 pp |
| 17 | experiments_v3 | simmixup | class-agnostic; class_agnostic_K60; ranks 21-40; nk=20; alpha=1; mix_prob=1; warmup=0; anchor=score_probability; top=0.2; power=0.7 | 49 | 86.67 | 36.40 | 36.07 | 2.8142 | 50.27 | -0.57 pp | -1.50 pp |
| 18 | experiments_v2 | simmixup | class-agnostic; class_agnostic_K40; ranks 21-40; nk=20; alpha=0.4; mix_prob=1; warmup=0 | 48 | 88.14 | 36.18 | 36.06 | 2.5931 | 51.96 | -0.58 pp | -1.51 pp |
| 19 | experiments_v2 | simmixup | class-agnostic; class_agnostic_K20; ranks 1-20; nk=20; alpha=1; mix_prob=1; warmup=0 | 39 | 82.33 | 36.64 | 35.94 | 2.6958 | 45.69 | -0.70 pp | -1.63 pp |
| 20 | experiments_v2 | simmixup | class-agnostic; class_agnostic_K60; ranks 21-60; nk=40; alpha=1; mix_prob=1; warmup=0 | 49 | 78.24 | 36.26 | 35.90 | 3.2581 | 41.98 | -0.74 pp | -1.67 pp |
| 21 | experiments_v2 | simmixup | class-agnostic; class_agnostic_K40; ranks 21-40; nk=20; alpha=1; mix_prob=1; warmup=10 | 38 | 79.48 | 36.56 | 35.70 | 2.5972 | 42.92 | -0.94 pp | -1.87 pp |
| 22 | experiments_v2 | simmixup | class-agnostic; class_agnostic_K40; ranks 21-40; nk=20; alpha=1; mix_prob=0.75; warmup=0 | 30 | 83.73 | 36.90 | 35.47 | 2.6945 | 46.83 | -1.17 pp | -2.10 pp |
| 23 | experiments_v3 | simmixup | class-agnostic; class_agnostic_K60; ranks 21-40; nk=20; alpha=1; mix_prob=1; warmup=0; anchor=score_probability; top=0.2; power=1 | 48 | 87.58 | 35.80 | 34.70 | 3.6894 | 51.78 | -1.94 pp | -2.87 pp |
| 24 | experiments_v3 | simmixup | class-agnostic; class_agnostic_K60; ranks 21-40; nk=20; alpha=1; mix_prob=1; warmup=0; anchor=score_probability; top=0.2; power=2 | 36 | 91.34 | 35.00 | 34.22 | 3.0599 | 56.34 | -2.42 pp | -3.35 pp |
| 25 | experiments_v3 | simmixup | class-agnostic; class_agnostic_K60; ranks 21-40; nk=20; alpha=1; mix_prob=1; warmup=0; anchor=top_fraction; top=0.2; power=1 | 50 | 94.16 | 32.24 | 31.79 | 3.6663 | 61.92 | -4.85 pp | -5.78 pp |
| 26 | metrics_v2 | simmixup | class-aware; class_aware_K20; ranks 1-20; nk=20; alpha=1; mix_prob=1; warmup=0 | 28 | 95.46 | 31.32 | 31.75 | 4.5125 | 64.14 | -4.89 pp | -5.82 pp |
| 27 | experiments_v2 | none | standard baseline | 46 | 98.79 | 32.06 | 31.47 | 5.6705 | 66.73 | -5.17 pp | -6.10 pp |
| 28 | metrics_v2 | none | standard baseline | 31 | 97.01 | 30.00 | 29.19 | 5.5395 | 67.01 | -7.45 pp | -8.38 pp |

## 50-Epoch ResNet50 k=50 Final-Stage Comparison

This is the first same-budget lower-k comparison for SimCutMix. SimCutMix wins
the k=50 group. A teammate-provided k=20 SimCutMix result exists, but the local
k=20 same-budget comparison is still incomplete.

| Method | Setting | Best epoch | Train acc | Val acc | Test acc | Test loss | Gap | Delta vs CutMix |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| simcutmix | class-agnostic; K60; ranks 1-20; nk=20; alpha=1; mix_prob=1; warmup=0 | 43 | 76.49 | 27.72 | 28.04 | 3.1799 | 48.77 | +1.97 pp |
| cutmix | standard baseline | 42 | 37.49 | 26.02 | 26.07 | 3.2853 | 11.47 | +0.00 pp |
| mixup | standard baseline | 40 | 70.97 | 24.88 | 24.08 | 3.4081 | 46.09 | -1.99 pp |

Note: a teammate-provided k=50 MixUp summary reports `25.16%` test accuracy for
the same metrics path. The table above keeps the local file value, `24.08%`,
until that discrepancy is reconciled.

## Strategy-Level Notes

### No Augmentation

No augmentation produces high training accuracy and poor validation/test
accuracy. The pattern is strongest in ResNet50, where train accuracy reaches
`98%` to `100%` while test accuracy remains `11.75%`, `22.32%`, and `31.88%`
for k = 20, 50, and 100. It is useful as an overfitting reference, not as a
competitive strategy.

### MixUp

MixUp improves all 100-epoch baseline cells over no augmentation. For ResNet50,
the gains over no augmentation are `+2.93 pp`, `+3.30 pp`, and `+4.79 pp` for
k = 20, 50, and 100. For ViT, the gains are `+3.70 pp`, `+4.12 pp`, and
`+4.17 pp`. MixUp also gives a much smaller train-validation gap than no
augmentation, which makes it the main regularized reference for the guided
experiments.

### CutMix

CutMix is the best standard method across the entire 100-epoch grid. The
ResNet50 k=100 CutMix result, `38.75%`, is the strongest standard baseline and
the main long-budget reference for guided methods. The new 50-epoch ResNet50
k=100 CutMix run reaches `37.57%`, which is the fair same-budget baseline for
the current k=100 guided runs. ViT also benefits strongly from CutMix, with low
gaps of `15.11`, `6.07`, and `6.75` percentage points across k values.

### AugMix

AugMix is helpful compared with no augmentation at k=50 and k=100, but it is not
as reliable as CutMix. It often keeps very high train accuracy and a large gap,
especially for ResNet50 and ViT k=100. In this result set, AugMix looks more
like an auxiliary baseline than the leading choice.

### SimMixUp

Class-agnostic SimMixUp is the most explored guided strategy. The strongest run
uses K60 class-agnostic neighbors, ranks 21-40, uniform sampling, alpha=1,
mix_prob=1, and no warmup. It reaches `37.69%` test accuracy, which is better
than 50-epoch MixUp but still below 100-epoch CutMix.

Most SimMixUp variants are tightly clustered. In `results/experiments/simmixup_ablation_v2`, the
11 SimMixUp ablations range from `35.47%` to `37.69%`, with a mean around
`36.35%`. This means changes to alpha, mix probability, warmup, and rank window
did not create a consistently better recipe in the current single-seed setup.

The gap behavior is also important. The 50-epoch MixUp baseline has a gap of
`30.05` points. Most SimMixUp variants have gaps around `40` to `53` points,
and the class-aware run reaches `64.14` points. That suggests guided MixUp is
usually less regularizing than ordinary MixUp even when its test accuracy is
similar.

### SimCutMix

SimCutMix is now the most promising guided strategy. The final-stage results
show that changing the rank window matters:

| k | Setting | Test acc | Delta vs same-budget CutMix | Read |
|---:|---|---:|---:|---|
| 100 | K40 ranks 21-40 | 40.02 | +2.45 pp | Best current result |
| 100 | K60 ranks 1-20 | 38.65 | +1.08 pp | Equivalent to the older K40 ranks 1-20 run because the first 20 neighbors match |
| 100 | K60 ranks 21-40 | 38.34 | +0.77 pp | Still beats 50-epoch CutMix |
| 50 | K60 ranks 1-20 | 28.04 | +1.97 pp | Positive lower-k result |

The strongest setting, K40 ranks 21-40, reaches `40.02%` test accuracy at 50
epochs. It beats the same-budget CutMix run (`37.57%`) and the older 100-epoch
CutMix run (`38.75%`) in this single-seed setup. This makes SimCutMix the clear
priority for the next experimental stage. The main remaining local gap is k=20:
MixUp and CutMix are still missing, and the teammate SimCutMix result still needs
matching local files before it can be counted in the main local summary.

### Anchor-Gated SimMixUp

The anchor-gated sweep used the anchor score file
`results/experiments/shared/anchor_scores/cifar100/k100_seed0/cifar100_resnet50_k100_seed0_none50_resnet50img_uw0p7_rw0p3.csv`.
That file contains 10,000 anchors with uncertainty, rarity, and combined score
columns. The filename indicates a score recipe based on uncertainty weight 0.7
and rarity weight 0.3.

Results:

| Anchor selection | Score power | Test acc | Gap |
|---|---:|---:|---:|
| score_probability | 0.5 | 36.20 | 48.66 |
| score_probability | 0.7 | 36.07 | 50.27 |
| score_probability | 1.0 | 34.70 | 51.78 |
| score_probability | 2.0 | 34.22 | 56.34 |
| top_fraction | 1.0 | 31.79 | 61.92 |

Softer score-probability selection is better than hard top-fraction selection,
and lower powers are better than sharper powers. However, all tested anchor
settings are below the 50-epoch MixUp baseline and below ungated SimMixUp. The
current evidence says targeted anchor mixing is not helping in this form.

## Parameter Patterns in SimMixUp Ablations

These summaries use the 11 SimMixUp runs under `results/experiments/simmixup_ablation_v2/metrics`.

| Ablation grouping | n | Min test | Mean test | Max test | Read |
|---|---:|---:|---:|---:|---|
| K20 source, ranks 1-20 | 1 | 35.94 | 35.94 | 35.94 | Below MixUp in this rerun |
| K40 source, mostly ranks 21-40 | 7 | 35.47 | 36.18 | 36.92 | Around MixUp, no stable lift |
| K60 source | 3 | 35.90 | 36.72 | 37.69 | Best individual SimMixUp run |
| Rank window 21-40 | 8 | 35.47 | 36.36 | 37.69 | Best rank window, but still variable |
| Rank window 41-60 | 1 | 36.57 | 36.57 | 36.57 | Similar to MixUp |
| nk=40, ranks 21-60 | 1 | 35.90 | 35.90 | 35.90 | More neighbors did not help here |
| alpha=1.0 | 9 | 35.47 | 36.33 | 37.69 | Default alpha remains best by max |
| alpha=0.7 | 1 | 36.32 | 36.32 | 36.32 | No improvement |
| alpha=0.4 | 1 | 36.06 | 36.06 | 36.06 | No improvement |
| mix_prob=1.0 | 9 | 35.70 | 36.39 | 37.69 | Full mixing is not clearly worse |
| mix_prob=0.75 | 1 | 35.47 | 35.47 | 35.47 | Worse in this run |
| mix_prob=0.5 | 1 | 36.35 | 36.35 | 36.35 | Similar to MixUp |
| warmup=0 | 9 | 35.47 | 36.36 | 37.69 | Best result used no warmup |
| warmup=5 | 1 | 36.41 | 36.41 | 36.41 | Slightly below MixUp |
| warmup=10 | 1 | 35.70 | 35.70 | 35.70 | Worse in this run |

Overall, the ablations do not reveal a clean monotonic tuning rule. The best
SimMixUp run uses later neighbors from the K60 file, but the average behavior is
still close to MixUp.

## Current Best Results

| Category | Best run | Test acc |
|---|---|---:|
| Best standard baseline overall | ResNet50 k=100 CutMix, 100 epochs | 38.75 |
| Best same-budget 50e standard baseline | ResNet50 k=100 CutMix, 50 epochs | 37.57 |
| Best guided run overall | ResNet50 k=100 SimCutMix K40 ranks 21-40, 50 epochs | 40.02 |
| Best SimMixUp run | ResNet50 k=100 SimMixUp K60 ranks 21-40, 50 epochs | 37.69 |
| Best anchor-gated SimMixUp run | score_probability, top=0.2, power=0.5 | 36.20 |
| Best lower-k guided run | ResNet50 k=50 SimCutMix K60 ranks 1-20, 50 epochs | 28.04 |
| Best ViT baseline | ViT k=100 CutMix, 100 epochs | 27.69 |

## Recommended Next Experiments

1. Finish the k=20 final-stage comparison: add/rerun MixUp and CutMix, and add
   the teammate SimCutMix JSON/CSV or rerun SimCutMix locally under the same
   50-epoch budget.
2. Repeat the strongest SimCutMix setting, k=100 K40 ranks 21-40, across at
   least three seeds and compare it to same-budget CutMix and MixUp.
3. Repeat the k=50 comparison across seeds if the k=20 results also look
   promising.
4. Add a same-budget 50-epoch no-augmentation reference only if the final table
   needs a complete baseline set.
5. Avoid expanding SimMixUp ablations until SimCutMix has been tested across
   seeds, because SimCutMix is now the clear lead method.
6. If anchor gating is revisited, avoid hard top-fraction selection first. Try
   softer score-probability schedules, lower anchor top percentages, and lower
   effective mix probabilities.
7. Add qualitative diagnostics for neighbor quality and mixed images. The strong
   K40 ranks 21-40 result makes it especially useful to inspect whether mid-rank
   neighbors provide more helpful diversity than nearest neighbors.

## Bottom Line

The current experiment set supports three practical conclusions. First, CutMix is
the strongest established standard baseline. Second, class-agnostic SimCutMix is
now more than a near miss: the K40 ranks 21-40 run beats both 50-epoch and
100-epoch CutMix in the current single-seed k=100 setup, and the k=50 SimCutMix
comparison is also positive. Third, most SimMixUp and anchor-gated variants
still produce results near MixUp, so the next stage should focus on completing
the local k=20 comparison, reconciling the k=50 MixUp discrepancy, and repeating
the strongest SimCutMix settings across seeds.
