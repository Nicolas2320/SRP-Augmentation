# Final Consolidated Results

Last audited: 2026-11-19.

Test accuracy is mean ± sample standard deviation across three seeds. Subset
and training seeds are paired except where the full-data protocol fixes the
subset seed and varies only the training seed.

The consolidated CIFAR-100 evidence contains 234 tested outcomes:

- 144 ResNet50 outcomes: two classifier initializations, six data budgets,
  four methods, and three seeds;
- 90 scratch-ViT outcomes: five data budgets, six methods, and three seeds.

Guided methods use frozen ImageNet-ResNet50 embeddings, class-agnostic ranks
21-40, uniform partner sampling, alpha 1, mixing probability 1, and no
warmup. Tiny ImageNet confirmation and the validation-only ablations remain
in their dedicated result documents and are linked below.

## CIFAR-100 ResNet50

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

### ResNet50 Paired Conclusions

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

## CIFAR-100 Scratch ViT

The ViT study uses the repository's scratch-trained CIFAR ViT with 32×32
inputs and 4×4 patches. Every cell uses 100 epochs and batch size 64. For
k<450, each training seed uses the correspondingly seeded subset; k=450 uses
the complete post-validation pool with subset seed 0 and three training
seeds.

| k | Crop+Flip | MixUp | CutMix | AugMix* | SimMixUp | SimCutMix |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 12.92 ± 0.53 | 12.33 ± 0.22 | 12.45 ± 0.47 | **14.11 ± 0.78** | 13.12 ± 0.52 | 12.55 ± 0.27 |
| 50 | 21.72 ± 0.52 | 19.06 ± 0.78 | 18.97 ± 0.22 | **22.49 ± 0.28** | 20.82 ± 0.14 | 20.84 ± 0.37 |
| 100 | **29.50 ± 0.47** | 24.80 ± 0.14 | 27.55 ± 0.62 | 28.40 ± 0.30 | 27.28 ± 0.21 | 27.44 ± 0.39 |
| 300 | **39.27 ± 0.27** | 32.92 ± 0.11 | 35.63 ± 0.56 | 36.13 ± 0.83 | 36.49 ± 0.30 | 35.82 ± 0.32 |
| 450 | **42.85 ± 0.64** | 35.13 ± 0.44 | 39.36 ± 0.60 | 37.89 ± 0.47 | 39.43 ± 0.71 | 40.25 ± 0.83 |

\*The local AugMix implementation omits JSD consistency and is therefore an
auxiliary augmentation result, not a paper-faithful AugMix comparison. It
also retains its historical learning-rate schedule.

### ViT Paired Effects

The primary comparisons isolate the effect of replacing random pairing with
the frozen similarity-ranked partner rule.

| k | SimMixUp − MixUp | SimCutMix − CutMix |
|---:|---:|---:|
| 20 | +0.80 pp | +0.09 pp |
| 50 | +1.76 pp | +1.87 pp |
| 100 | +2.48 pp | -0.11 pp |
| 300 | +3.58 pp | +0.19 pp |
| 450 | +4.30 pp | +0.89 pp |

Across the 15 paired k/seed comparisons, SimMixUp improves on MixUp in
15/15 cases with a mean gain of +2.58 pp. The gain grows with the data budget.
SimCutMix is less stable: its mean gain over CutMix is +0.59 pp, with 9 wins,
5 losses, and 1 tie.

The overall ViT winner is Crop+Flip at k=100, k=300, and k=450. The auxiliary
AugMix implementation leads at k=20 and k=50. Therefore, the defensible ViT
claim is that similarity guidance consistently repairs a weak MixUp baseline;
it does not establish the guided methods as universally best.

## Cross-Architecture Interpretation

- Scratch ResNet50 provides the clearest support for SimCutMix at scarce and
  moderate budgets: it leads at k=50 and k=100 and has the highest k=450 mean.
- ViT transfers the partner-selection benefit most consistently through
  SimMixUp, not SimCutMix. This shows that the useful guided variant depends on
  classifier architecture and base mixing operator.
- ImageNet-pretrained ResNet50 favors standard CutMix at every tested budget;
  strong classifier pretraining removes the advantage of guided pairing.
- None of the small differences should be described as statistically
  significant from only three seeds. Report individual seeds or uncertainty
  bars alongside the means.

Cross-dataset evidence is reported in [Tiny ImageNet results](tiny_imagenet_results.md).
The controlled rank-window study is reported in
[pair-selection ablation results](pair_selection_ablation_results.md), and the
negative curriculum result is documented in
[easy-to-hard curriculum plan](easy_to_hard_curriculum_plan.md).

## Provenance Notes

- The reused pretrained seed-0 results at k=20/50/100 had historical test
  exposure. Configuration selection used validation evidence only, but the
  prior exposure remains disclosed.
- ViT seed-0 results at k=20/50/100 and the k=450 CutMix/AugMix results predate
  the three-seed completion. Their frozen recipes match the new repeats, but
  their historical provenance remains disclosed.
- ViT k=450 uses subset seed 0 for all runs because it is the complete
  post-validation pool. Its standard deviation therefore reflects training
  variability rather than subset variability.
- Some concurrent evaluation records report `git_dirty=true` because jobs
  observed other tracked result summaries being updated. No source code or
  scientific configuration changed during those evaluations.
- AugMix remains auxiliary because the local implementation omits the
  canonical JSD consistency loss; reporting it as paper-faithful AugMix would
  be misleading.
