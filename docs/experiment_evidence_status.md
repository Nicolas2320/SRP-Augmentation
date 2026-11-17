# Experiment Evidence Status Before the Final Presentation

Last audited: 2026-11-14.

Accuracy values are percentages. Final cells use one best-validation
checkpoint test evaluation per frozen configuration and report mean ± sample
standard deviation across three paired seeds.

## Status Legend

| Status | Meaning |
|---|---|
| Final | Three paired seeds and one test evaluation per frozen configuration |
| Validation-only | Test set intentionally not evaluated |
| Provisional | At least one required seed is missing |
| Historical | Earlier recipe/cohort; not pooled with the current matrix |

## CIFAR-100 ResNet50

Status: **Final**. The primary matrix contains 144 tested outcomes:
`2 initializations × 6 k values × 4 methods × 3 seeds`.

### Training from Scratch

| k | Crop+Flip | MixUp | CutMix | SimCutMix |
|---:|---:|---:|---:|---:|
| 20 | **17.51 ± 0.45** | 14.13 ± 0.98 | 15.54 ± 0.59 | 16.72 ± 0.14 |
| 50 | 31.15 ± 0.99 | 27.49 ± 0.92 | 30.09 ± 1.12 | **33.26 ± 0.59** |
| 100 | 52.17 ± 0.69 | 50.49 ± 0.74 | 51.94 ± 0.34 | **52.91 ± 0.39** |
| 200 | 66.10 ± 0.14 | 67.11 ± 0.36 | **68.09 ± 0.09** | 67.82 ± 0.47 |
| 300 | 72.32 ± 0.12 | 73.30 ± 0.21 | **74.76 ± 0.15** | 73.95 ± 0.19 |
| 450 | 70.81 ± 1.67 | 69.51 ± 2.13 | 71.99 ± 1.08 | **72.40 ± 1.03** |

### ImageNet-Pretrained Classifier

| k | Crop+Flip | MixUp | CutMix | SimCutMix |
|---:|---:|---:|---:|---:|
| 20 | 66.81 ± 0.33 | 67.17 ± 0.74 | **69.68 ± 0.83** | 67.85 ± 0.04 |
| 50 | 72.71 ± 0.17 | 73.65 ± 0.16 | **75.80 ± 0.15** | 74.58 ± 0.51 |
| 100 | 76.38 ± 0.53 | 77.90 ± 0.16 | **79.65 ± 0.28** | 78.28 ± 0.80 |
| 200 | 80.66 ± 0.23 | 81.21 ± 0.24 | **82.79 ± 0.24** | 81.89 ± 0.54 |
| 300 | 82.45 ± 0.24 | 83.34 ± 0.07 | **84.18 ± 0.34** | 83.53 ± 0.21 |
| 450 | 78.34 ± 0.22 | 77.62 ± 0.19 | **79.82 ± 0.17** | 78.39 ± 0.27 |

The k=450 recipe uses 50 instead of 100 epochs. MixUp is now a complete
fourth baseline but does not lead any aggregate cell. AugMix remains appendix
only because the local implementation omits the canonical JSD consistency
loss. SimMixUp remains exploratory because its frozen configuration was not
confirmed across three seeds.

## Tiny ImageNet ResNet50

| Initialization | k | Crop+Flip | CutMix | SimCutMix | Status |
|---|---:|---:|---:|---:|---|
| Scratch | 20 | 12.71 ± 0.63 | 12.00 ± 0.90 | **14.87 ± 1.07** | Final |
| Scratch | 100 | 42.64 ± 0.79 | 43.14 ± 0.64 | **44.70 ± 0.42** | Final |
| Scratch | 450 | **60.94 ± 0.69** | 60.56 ± 1.04 | 60.49 ± 0.62 | Final |
| Pretrained | 20 | **67.49 ± 0.55** | 66.84 ± 0.86 | 65.65 ± 0.55 | Final |
| Pretrained | 100 | 65.95 ± 0.36 | **68.77 ± 0.25** | 67.69 ± 0.26 | Final |
| Pretrained | 450 | 65.44 ± 0.13 | **65.82 ± 0.14** | 65.54 ± 0.37 | Final |

Status: **Final**. All 54 Tiny ImageNet outcomes are available.

## Pair-Selection Ablation

Validation accuracy, seed 0. All conditions use CutMix alpha 1 and mixing
probability 0.5.

| Setting | Random | Same class | Ranks 1-10 | Ranks 11-20 | Ranks 21-40 | Different label |
|---|---:|---:|---:|---:|---:|---:|
| Scratch k=20 | 15.54 | 16.70 | 17.56 | **17.68** | 16.64 | 16.96 |
| Scratch k=100 | 52.98 | 53.30 | 54.06 | 53.68 | **54.90** | 54.32 |
| Scratch k=300 | **74.86** | 73.32 | 74.48 | 73.78 | 74.42 | 74.06 |
| Pretrained k=100 | 78.74 | 76.92 | 78.80 | 78.96 | **79.18** | 78.98 |

Status: **Validation-only ablation complete**. Partner selection is
regime-dependent rather than universally optimal.

## Easy-to-Hard Curriculum

Validation results:

| k | Random CutMix | Fixed ranks 21-40 | Easy-to-hard | Curriculum − fixed |
|---:|---:|---:|---:|---:|
| 20 | 15.50 ± 1.20 | 16.78 ± 0.94 | **16.97 ± 0.78** | +0.19 ± 0.16 |
| 100 | 52.03 ± 0.82 | 53.61 ± 1.24 | **53.65 ± 1.04** | +0.03 ± 0.20 |

The selected scratch k=20 test evaluation is complete:

| Condition | Test accuracy |
|---|---:|
| Crop+Flip | **17.51 ± 0.45** |
| Random CutMix | 15.54 ± 0.59 |
| Fixed ranks 21-40, mix probability 0.5 | 17.18 ± 0.76 |
| Easy-to-hard curriculum, mix probability 0.5 | 17.10 ± 0.85 |

Curriculum minus fixed is -0.07 ± 0.11 pp and wins 0/3 seeds. The schedule
does not improve on the simpler fixed window and belongs in the ablation or
appendix, not as the proposed final method.

## Remaining Work

| Priority | Task | Status |
|---:|---|---|
| 1 | Generate final figures and presentation deck | In progress |
| 2 | Minimal ViT transfer study | Optional/delegated |
| 3 | Tiny ImageNet k=50/k=200 or canonical AugMix | Future work |

No additional large ResNet50 matrix is required for the final presentation.

## Supported Conclusion

> Similarity-guided CutMix is useful when ResNet50 is trained from scratch
> with scarce or moderate labeled data, but it is not a universal improvement.
> Its advantage disappears with strong classifier pretraining or abundant
> labeled data. Standard CutMix is the strongest and most consistent
> pretrained method, while an easy-to-hard pairing curriculum adds no benefit
> over a fixed rank window.
