# Third-presentation CIFAR-100 confirmation results

Test accuracy is reported as mean ± sample standard deviation across three paired seeds.
The frozen SimCutMix configuration uses class-agnostic ranks 21–40, uniform sampling,
CutMix alpha 1, mixing probability 1, and no warmup.

## Scratch

| k | Crop+Flip | CutMix | SimCutMix |
|---:|---:|---:|---:|
| 20 | 17.51 ± 0.45 | 15.54 ± 0.59 | 16.72 ± 0.14 |
| 50 | 31.15 ± 0.99 | 30.09 ± 1.12 | 33.26 ± 0.59 |
| 100 | 52.17 ± 0.69 | 51.94 ± 0.34 | 52.91 ± 0.39 |
| 200 | 66.10 ± 0.14 | 68.09 ± 0.09 | 67.82 ± 0.47 |
| 300 | 72.32 ± 0.12 | 74.76 ± 0.15 | 73.95 ± 0.19 |
| 450 | 70.81 ± 1.67 | 71.99 ± 1.08 | 72.40 ± 1.03 |

## ImageNet pretrained

| k | Crop+Flip | CutMix | SimCutMix |
|---:|---:|---:|---:|
| 20 | 66.81 ± 0.33 | 69.68 ± 0.83 | 67.85 ± 0.04 |
| 50 | 72.71 ± 0.17 | 75.80 ± 0.15 | 74.58 ± 0.51 |
| 100 | 76.38 ± 0.53 | 79.65 ± 0.28 | 78.28 ± 0.80 |
| 200 | 80.66 ± 0.23 | 82.79 ± 0.24 | 81.89 ± 0.54 |
| 300 | 82.45 ± 0.24 | 84.18 ± 0.34 | 83.53 ± 0.21 |
| 450 | 78.34 ± 0.22 | 79.82 ± 0.17 | 78.39 ± 0.27 |

## Paired conclusions

Across all 36 initialization/k/seed comparisons:

- SimCutMix versus Crop+Flip: mean delta +1.18 pp; wins 31/36 paired cells.
- SimCutMix versus CutMix: mean delta -0.23 pp; wins 13/36 paired cells.
- CutMix versus Crop+Flip: mean delta +1.41 pp; wins 27/36 paired cells.

The k=450 cells use subset seed 0 with three training seeds and therefore measure
optimization variability only. They also use the predeclared 50-epoch recipe, so the
drop from k=300 to k=450 must not be interpreted as a pure data-size effect.

## Provenance notes

- The reused pretrained seed-0 results at k=20/50/100 had historical test exposure.
  Configuration selection used validation evidence only, but the prior exposure must
  remain disclosed.
- Eighteen concurrent seed-0 evaluation jobs recorded `git_dirty=true` because they
  observed other tracked result summaries being updated. Commit `5fd2e14` changes only
  the 27 evaluated summaries; no source or scientific configuration changed.
