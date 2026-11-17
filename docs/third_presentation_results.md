# Final CIFAR-100 ResNet50 Results

Last audited: 2026-11-14.

Test accuracy is mean ± sample standard deviation across three paired seeds.
The matrix contains 144 tested outcomes: two classifier initializations, six
data budgets, four methods, and three seeds. SimCutMix uses frozen
ImageNet-ResNet50 embeddings, class-agnostic ranks 21-40, uniform partner
sampling, CutMix alpha 1, mixing probability 1, and no warmup.

## Training from Scratch

| k | Crop+Flip | MixUp | CutMix | SimCutMix |
|---:|---:|---:|---:|---:|
| 20 | **17.51 ± 0.45** | 14.13 ± 0.98 | 15.54 ± 0.59 | 16.72 ± 0.14 |
| 50 | 31.15 ± 0.99 | 27.49 ± 0.92 | 30.09 ± 1.12 | **33.26 ± 0.59** |
| 100 | 52.17 ± 0.69 | 50.49 ± 0.74 | 51.94 ± 0.34 | **52.91 ± 0.39** |
| 200 | 66.10 ± 0.14 | 67.11 ± 0.36 | **68.09 ± 0.09** | 67.82 ± 0.47 |
| 300 | 72.32 ± 0.12 | 73.30 ± 0.21 | **74.76 ± 0.15** | 73.95 ± 0.19 |
| 450 | 70.81 ± 1.67 | 69.51 ± 2.13 | 71.99 ± 1.08 | **72.40 ± 1.03** |

## ImageNet-Pretrained Classifier

| k | Crop+Flip | MixUp | CutMix | SimCutMix |
|---:|---:|---:|---:|---:|
| 20 | 66.81 ± 0.33 | 67.17 ± 0.74 | **69.68 ± 0.83** | 67.85 ± 0.04 |
| 50 | 72.71 ± 0.17 | 73.65 ± 0.16 | **75.80 ± 0.15** | 74.58 ± 0.51 |
| 100 | 76.38 ± 0.53 | 77.90 ± 0.16 | **79.65 ± 0.28** | 78.28 ± 0.80 |
| 200 | 80.66 ± 0.23 | 81.21 ± 0.24 | **82.79 ± 0.24** | 81.89 ± 0.54 |
| 300 | 82.45 ± 0.24 | 83.34 ± 0.07 | **84.18 ± 0.34** | 83.53 ± 0.21 |
| 450 | 78.34 ± 0.22 | 77.62 ± 0.19 | **79.82 ± 0.17** | 78.39 ± 0.27 |

## Paired Conclusions

Across the 36 matched initialization/k/seed comparisons:

- SimCutMix versus Crop+Flip: mean delta +1.18 pp; wins 31/36.
- SimCutMix versus CutMix: mean delta -0.23 pp; wins 13/36.
- CutMix versus Crop+Flip: mean delta +1.41 pp; wins 27/36.
- MixUp versus Crop+Flip: mean delta -0.37 pp; wins 21/36.
- MixUp versus CutMix: mean delta -1.78 pp; wins 1/36.
- MixUp versus SimCutMix: mean delta -1.56 pp; wins 1/36.

The expanded baseline confirms that MixUp is relevant but not the strongest
method in any aggregate cell. Guided pairing is most useful for scratch
training at low/moderate budgets: SimCutMix leads at k=50 and k=100 and also
has the highest k=450 mean. CutMix leads at scratch k=200/k=300 and at every
pretrained budget.

The k=450 cells use the full post-validation pool with subset seed 0 and three
training seeds, so they measure optimization variability only. They also use
the predeclared 50-epoch recipe; the drop from k=300 to k=450 is not a pure
data-size effect.

## Provenance Notes

- The reused pretrained seed-0 results at k=20/50/100 had historical test
  exposure. Configuration selection used validation evidence only, but the
  prior exposure remains disclosed.
- Some concurrent evaluation records report `git_dirty=true` because jobs
  observed other tracked result summaries being updated. No source code or
  scientific configuration changed during those evaluations.
- AugMix is excluded because the local implementation omits the canonical JSD
  consistency loss; reporting it as paper-faithful AugMix would be misleading.
