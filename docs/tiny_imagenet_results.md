# Tiny ImageNet Confirmation Results

Last audited: 2026-11-14.

Accuracy is measured on the labeled official Tiny ImageNet validation
partition, used here as the held-out test set. The official unlabeled test
partition was never used. Complete cells report mean ± sample standard
deviation across three paired seeds.

## Training from Scratch

| k | Crop+Flip | CutMix | SimCutMix | Status |
|---:|---:|---:|---:|---|
| 20 | 12.71 ± 0.63 | 12.00 ± 0.90 | **14.87 ± 1.07** | Final, 3 seeds |
| 100 | 42.64 ± 0.79 | 43.14 ± 0.64 | **44.70 ± 0.42** | Final, 3 seeds |
| 450 | **60.94 ± 0.69** | 60.56 ± 1.04 | 60.49 ± 0.62 | Final, 3 seeds |

## ImageNet-Pretrained Classifier

| k | Crop+Flip | CutMix | SimCutMix | Status |
|---:|---:|---:|---:|---|
| 20 | **67.49 ± 0.55** | 66.84 ± 0.86 | 65.65 ± 0.55 | Final, 3 seeds |
| 100 | 65.95 ± 0.36 | **68.77 ± 0.25** | 67.69 ± 0.26 | Final, 3 seeds |
| 450 | 65.44 ± 0.13 | **65.82 ± 0.14** | 65.54 ± 0.37 | Final, 3 seeds |

## Paired Effects

- Scratch k=20: SimCutMix − Crop+Flip +2.16 ± 0.54 pp; wins 3/3.
- Scratch k=100: SimCutMix − Crop+Flip +2.06 ± 0.43 pp; wins 3/3.
- Scratch k=100: SimCutMix − CutMix +1.56 ± 0.46 pp; wins 3/3.
- Scratch k=450: SimCutMix − Crop+Flip -0.45 ± 1.12 pp; wins 1/3.
- Pretrained k=20: SimCutMix − Crop+Flip -1.85 ± 0.01 pp; wins 0/3.
- Pretrained k=100: SimCutMix − Crop+Flip +1.74 ± 0.60 pp; wins 3/3.
- Pretrained k=100: SimCutMix − CutMix -1.08 ± 0.26 pp; wins 0/3.
- Pretrained k=450: SimCutMix − Crop+Flip +0.11 ± 0.24 pp; wins 2/3.

## Interpretation

Tiny ImageNet independently reproduces the useful scratch low/moderate-data
regime: SimCutMix leads at k=20 and k=100. The advantage disappears at full
data. With an ImageNet-pretrained classifier, guided pairing is not the best
method at k=20 or k=450, and standard CutMix also wins k=100 across all three
paired seeds.

The k=450 recipe differs in learning rate and epoch count, so changes between
k=100 and k=450 are not a pure data-size ablation.

## Provenance

- Validation accuracy recomputed from each saved checkpoint matches the value
  used to select that checkpoint.
- Concurrent jobs may report `git_dirty=true` when another job updates tracked
  result summaries; the scientific configuration is recorded in every
  summary.
