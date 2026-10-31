# Experiment Evidence Status Before the Final Presentation

Last audited: 2026-10-31.

This document separates final three-seed evidence from screening, historical,
and validation-only results. Accuracy values are percentages. Final tables use
the best-validation checkpoint and report mean ± sample standard deviation
across three paired seeds.

## Status Legend

| Status | Meaning |
|---|---|
| Final | Three paired seeds and one test evaluation per frozen configuration |
| Validation-only | Test set intentionally not evaluated |
| Screening | One seed; useful for prioritization, not a final claim |
| Historical | Earlier recipe/cohort; cannot be pooled with the current matrix |
| Pending | Required only if the corresponding presentation claim is retained |

## CIFAR-100: Final ResNet50 Evidence

The final matrix contains 108 tested runs: two classifier initializations, six
data budgets, three methods, and three paired seeds. SimCutMix uses frozen
ImageNet-ResNet50 embeddings, class-agnostic ranks 21-40, and uniform partner
sampling.

### Training from Scratch

| k | Crop+Flip | CutMix | SimCutMix | SimCutMix − CutMix |
|---:|---:|---:|---:|---:|
| 20 | 17.51 ± 0.45 | 15.54 ± 0.59 | 16.72 ± 0.14 | +1.18 |
| 50 | 31.15 ± 0.99 | 30.09 ± 1.12 | 33.26 ± 0.59 | +3.17 |
| 100 | 52.17 ± 0.69 | 51.94 ± 0.34 | 52.91 ± 0.39 | +0.97 |
| 200 | 66.10 ± 0.14 | 68.09 ± 0.09 | 67.82 ± 0.47 | −0.27 |
| 300 | 72.32 ± 0.12 | 74.76 ± 0.15 | 73.95 ± 0.19 | −0.81 |
| 450 | 70.81 ± 1.67 | 71.99 ± 1.08 | 72.40 ± 1.03 | +0.41 |

### ImageNet-Pretrained Classifier

| k | Crop+Flip | CutMix | SimCutMix | SimCutMix − CutMix |
|---:|---:|---:|---:|---:|
| 20 | 66.81 ± 0.33 | 69.68 ± 0.83 | 67.85 ± 0.04 | −1.83 |
| 50 | 72.71 ± 0.17 | 75.80 ± 0.15 | 74.58 ± 0.51 | −1.22 |
| 100 | 76.38 ± 0.53 | 79.65 ± 0.28 | 78.28 ± 0.80 | −1.37 |
| 200 | 80.66 ± 0.23 | 82.79 ± 0.24 | 81.89 ± 0.54 | −0.90 |
| 300 | 82.45 ± 0.24 | 84.18 ± 0.34 | 83.53 ± 0.21 | −0.65 |
| 450 | 78.34 ± 0.22 | 79.82 ± 0.17 | 78.39 ± 0.27 | −1.43 |

Status: **Final**. The k=450 recipe uses 50 rather than 100 epochs and must
not be interpreted as a pure continuation of the k curve.

## Why MixUp and AugMix Are Not in the Final Table Yet

They are relevant baselines, but their current evidence is not equivalent to
the final three-seed matrix.

| Method | Available evidence | Current use | Missing for a final claim |
|---|---|---|---|
| MixUp | Seed-0 screening across the CIFAR-100 k grid; historical tested cells at pretrained k=20/50/100 | Secondary baseline/screening | Three paired seeds under the final recipe |
| SimMixUp | Seed-0 screening across the grid; strongly configuration-sensitive | Exploratory proposed variant | Frozen configuration and three-seed confirmation |
| AugMix | Historical single-seed CIFAR-100 results | Appendix only | Paper-faithful JSD implementation and three-seed confirmation |
| Raw/no crop-flip | Historical single-seed results | Diagnostic lower baseline | No expansion needed |

CutMix was prioritized because it is the matched mixing operator: CutMix and
SimCutMix differ primarily in partner selection when their mixing probability
is controlled. MixUp changes the mixing operator. AugMix is also a different
augmentation family, and the local implementation omits the original AugMix
JSD consistency loss, so presenting it as a canonical AugMix baseline would be
misleading.

If the final presentation needs a robust fourth method, standard MixUp is the
best candidate. Completing it over both initializations and all six k values
requires 24 new training runs for seeds 1 and 2, plus a provenance/evaluation
audit of the 12 seed-0 cells. AugMix should not be promoted before its
implementation is made paper-faithful.

## Tiny ImageNet: ResNet50 Evidence

### Final Test Results

| Initialization | k | Crop+Flip | CutMix | SimCutMix | SimCutMix − CutMix | Status |
|---|---:|---:|---:|---:|---:|---|
| Scratch | 20 | 12.71 ± 0.63 | 12.00 ± 0.90 | **14.87 ± 1.07** | **+2.87** | Final |
| Scratch | 450 | **60.94 ± 0.69** | 60.56 ± 1.04 | 60.49 ± 0.62 | −0.07 | Final |
| Pretrained | 20 | **67.49 ± 0.55** | 66.84 ± 0.86 | 65.65 ± 0.55 | −1.19 | Final |
| Pretrained | 450 | 65.44 ± 0.13 | **65.82 ± 0.14** | 65.54 ± 0.37 | −0.28 | Final |

### Existing k=100 Screening

These are validation accuracies from seed 0 only; the test set was not used.

| Initialization | Crop+Flip | CutMix | SimCutMix | SimCutMix − CutMix | Status |
|---|---:|---:|---:|---:|---|
| Scratch | 42.64 | 43.87 | **45.80** | **+1.93** | Screening |
| Pretrained | 66.01 | **69.20** | 68.44 | −0.76 | Screening |

Only k=20, 100, and 450 were predeclared and implemented for Tiny ImageNet.
k=20 and k=450 were repeated because they provide low-data and maximum-data
anchors. k=100 already exists as the middle screening point and is the highest
priority Tiny ImageNet gap: the scratch result is promising and the required
seed-1/2 split files already exist.

Adding k=50 and k=200 would require new committed splits, new neighbor payloads,
and 36 new training runs for two initializations × two k values × three methods
× three seeds. That is lower priority than completing k=100.

## Pair-Selection Ablation

Validation accuracy, seed 0 only. All conditions use CutMix alpha 1 and mixing
probability 0.5.

| Setting | Random | Same class | Ranks 1-10 | Ranks 11-20 | Ranks 21-40 | Different label |
|---|---:|---:|---:|---:|---:|---:|
| Scratch k=20 | 15.54 | 16.70 | 17.56 | **17.68** | 16.64 | 16.96 |
| Scratch k=100 | 52.98 | 53.30 | 54.06 | 53.68 | **54.90** | 54.32 |
| Scratch k=300 | **74.86** | 73.32 | 74.48 | 73.78 | 74.42 | 74.06 |
| Pretrained k=100 | 78.74 | 76.92 | 78.80 | 78.96 | **79.18** | 78.98 |

Status: **Validation-only ablation complete**. The result supports partner
selection as regime-dependent rather than universally optimal.

## Easy-to-Hard Curriculum

Validation accuracy, mean ± sample SD across three paired seeds. Mixing
probability is controlled at 0.5.

| k | Random CutMix | Fixed ranks 21-40 | Easy-to-hard | Curriculum − fixed |
|---:|---:|---:|---:|---:|
| 20 | 15.50 ± 1.20 | 16.78 ± 0.94 | **16.97 ± 0.78** | +0.19 ± 0.16 |
| 100 | 52.03 ± 0.82 | 53.61 ± 1.24 | **53.65 ± 1.04** | +0.03 ± 0.20 |

Status: **Validation-only complete**. Curriculum is only marginally better
than the fixed window. Final test evaluation is pending only for the selected
scratch k=20 curriculum checkpoints.

## Pending Work and Priority

| Priority | Experiment/task | New work | Purpose | Recommendation |
|---:|---|---:|---|---|
| 1 | Curriculum scratch k=20 final evaluation | 3 checkpoint evaluations; no retraining | Close the optional curriculum study | Do now |
| 2 | Tiny ImageNet k=100 confirmation | Verify/build two seed-specific support payloads; 12 new seed-1/2 training runs; evaluate six existing/new configurations as appropriate | Add the missing middle point and test whether the low-data advantage decays | Do next |
| 3 | Standard MixUp three-seed CIFAR-100 matrix | 24 new seed-1/2 training runs plus seed-0 audit/evaluation | Add a robust fourth baseline | Recommended if cluster time permits |
| 4 | Final analysis and presentation figures | No GPU work | Convert evidence into the final narrative | Required |
| 5 | Minimal ViT transfer study | 18 runs delegated separately | Test architecture sensitivity at one positive and one negative ResNet regime | Optional/delegated |
| 6 | Tiny ImageNet k=50 and k=200 | New splits/support plus 36 runs | Denser Tiny ImageNet curve | Not recommended before priorities 1-4 |
| 7 | Canonical AugMix confirmation | Implement JSD loss, validate, then repeat seeds | Fair AugMix comparison | Future work |

## Prior Conclusion

The strongest supported conclusion is not that SimCutMix always wins. It is:

> Similarity-guided CutMix is most useful when the classifier is trained from
> scratch with scarce or moderate labeled data. Its advantage over random
> CutMix diminishes with more labeled data and is absent when the classifier
> already has strong ImageNet initialization.

CIFAR-100 provides the dense k curve. Tiny ImageNet independently supports the
low-data scratch result and currently provides low, middle-screening, and
maximum-data anchors. The curriculum result suggests that changing partner
difficulty over epochs adds little beyond selecting a suitable fixed rank
window.
