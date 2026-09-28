# Presentation Outline: Similarity-Guided Augmentation

Updated: 2026-09-28. Suggested length: 12-15 minutes.

## Main Message

The current pretrained ResNet50 evidence is mixed. CutMix is strongest at
k=20 and k=50. SimMixUp is strongest at k=100, but only 0.24 percentage points
above CutMix in a single run. Similarity guidance is therefore worth
multi-seed evaluation, but the present evidence does not establish a general
advantage.

## Slide 1 - Title

**Similarity-Guided Data Augmentation for Low-Data Image Classification**

- CIFAR-100
- ResNet50 and ViT
- SimMixUp and SimCutMix

Opening:

> We study whether MixUp and CutMix become more effective in low-data
> classification when the partner image is selected by feature similarity
> instead of uniformly at random.

## Slide 2 - Motivation and Question

- Small labeled sets encourage memorization.
- Augmentation can add useful variation but can also distort scarce evidence.
- Standard MixUp/CutMix choose random partners.
- Question: can a moderately similar partner create a more useful mixed
  example?

Do not state the hypothesis as an established mechanism.

## Slide 3 - Methods

| Group | Methods |
|---|---|
| Reference | Raw; crop + flip (`none`) |
| Standard | MixUp, CutMix, AugMix |
| Proposed | SimMixUp, SimCutMix |

Explain that the guided methods change partner selection, then apply the usual
MixUp or CutMix operation.

## Slide 4 - Classifier Versus Guidance Encoder

| Component | Initialization | Role |
|---|---|---|
| ResNet50 classifier | Scratch or ImageNet | Produces reported accuracy |
| ViT classifier | Scratch | Produces reported accuracy |
| ResNet50 guidance encoder | Frozen ImageNet | Computes neighbor similarity |

Key distinction:

> A scratch classifier can still use an external pretrained encoder for
> pairing. Classifier initialization and guidance-encoder initialization are
> separate experimental variables.

## Slide 5 - Guided Pipeline

```text
Committed k-shot split
        |
        v
Frozen ImageNet ResNet50 embeddings
        |
        v
Exact class-agnostic K40 neighbors
        |
        v
Selected rank window + uniform sampling
        |
        v
SimMixUp or SimCutMix
        |
        v
Classifier training and validation selection
```

Important configuration note:

- Pretrained SimCutMix evidence uses ranks 21-40.
- Pretrained SimMixUp evidence uses ranks 1-10.

Do not imply that a single rank window is proven optimal or that the two
guided methods form a shared pairing-window ablation.

## Slide 6 - Experimental Protocol

- Dataset: CIFAR-100.
- Budgets: k=20, 50, 100; scratch-group evidence also includes k=450.
- k is the number of labeled training images per class.
- Fixed 5,000-image validation set and 10,000-image test set.
- Best checkpoint selected by validation accuracy.
- One final test evaluation per completed run.
- Current evidence: subset seed 0, training seed 0.
- Current ResNet50: standard 224x224 torchvision model.

Protocol caveat:

> Baseline CutMix is applied with probability 0.5, while SimCutMix is applied
> with probability 1.0. Their difference is not partner selection alone.

## Slide 7 - Pretrained ResNet50 Overview

Use
`figures/pretrained/available_test_accuracy_vs_k.png`.

| Method | k=20 | k=50 | k=100 |
|---|---:|---:|---:|
| Raw | 65.33% | 73.26% | 77.33% |
| Crop + flip | 66.73% | 72.90% | 76.88% |
| MixUp | 67.66% | 73.47% | 77.72% |
| CutMix | **68.88%** | **75.75%** | 79.89% |
| AugMix | 62.37% | 71.02% | 75.43% |
| SimMixUp | 44.75% | 74.39% | **80.13%** |
| SimCutMix | 67.90% | 75.13% | 78.53% |

Say:

> CutMix leads at k=20 and k=50. SimMixUp leads at k=100, but the margin over
> CutMix is only 0.24 points. The k=20 SimMixUp result is a clear negative
> outcome for that tracked configuration.

## Slide 8 - Pretrained Direct Comparisons

Use `figures/pretrained/matched_test_accuracy.png`.

Key differences:

| Budget | Comparison | Difference |
|---:|---|---:|
| k=20 | SimCutMix - CutMix | -0.98 pp |
| k=50 | SimCutMix - CutMix | -0.62 pp |
| k=100 | SimCutMix - CutMix | -1.36 pp |
| k=20 | SimMixUp - MixUp | -22.91 pp |
| k=50 | SimMixUp - MixUp | +0.92 pp |
| k=100 | SimMixUp - MixUp | +2.41 pp |
| k=100 | SimMixUp - CutMix | +0.24 pp |

Interpretation:

> The outcome depends on the guided operator, rank window, and budget.
> SimCutMix does not beat CutMix in the pretrained matrix. SimMixUp improves
> with budget and leads at k=100, but it is unstable across the three budgets.

## Slide 9 - Validation Evidence

Use `figures/pretrained/matched_validation_curves.png`.

Explain:

- Each star marks the highest-validation checkpoint.
- That checkpoint, not the final epoch, supplies the test result.
- Configuration selection must use validation evidence rather than test
  accuracy.
- A single trajectory does not quantify seed uncertainty.

Avoid narrating small curve differences as statistically meaningful.

## Slide 10 - Train-Validation Gap

Use `figures/pretrained/matched_overfitting_gap.png`.

Define:

```text
gap = training accuracy - validation accuracy
```

Fairness caveat:

- Clean-label methods use ordinary training accuracy.
- MixUp/CutMix variants use partial-credit mixed-label accuracy.
- Absolute gaps are therefore not directly comparable across all methods.
- A smaller gap is supporting evidence, not proof that a method eliminates
  overfitting.

## Slide 11 - Historical Scratch and ViT Evidence

Use `figures/scratch/available_test_accuracy_vs_k.png` only after clearly
labeling it **scratch/historical cohort**.

Useful context:

- Historical scratch ResNet50 SimCutMix reached 46.16% versus CutMix 43.20%
  at k=100 (+2.96 pp).
- Historical ViT does not show a universal guided-method advantage.
- Scratch-group results include imported architecture/recipe cohorts and must
  not be numerically combined with pretrained results.

Purpose of this slide: motivate architecture/initialization sensitivity, not
claim a pooled overall effect.

## Slide 12 - Coverage

Use both coverage figures:

- `figures/pretrained/experiment_coverage.png`
- `figures/scratch/experiment_coverage.png`

State:

- Pretrained ResNet50: seven methods at k=20/50/100; no k=450.
- Scratch ResNet50: seven methods at k=20/50/100/450.
- ViT: six methods at k=20/50/100; four baselines at k=450.
- All cells are single-seed.

## Slide 13 - Limitations

- No multi-seed averages, uncertainty intervals, or significance tests.
- Method-specific guided rank windows in pretrained evidence.
- CutMix/SimCutMix mixing probability is not controlled equally.
- Pretrained k=450 is missing.
- Guided neighbor payloads are absent locally and must be regenerated.
- Only 43 of 71 referenced best checkpoints are available locally; four more
  checkpoints are in incomplete run folders.
- Historical environments are not fully captured.
- Local AugMix omits JSD consistency loss.

## Slide 14 - Next Experiments

1. Select guided configurations using validation evidence.
2. Equalize CutMix and SimCutMix mixing probability for a pairing-only
   ablation.
3. Regenerate and store neighbor payloads.
4. Repeat selected methods across subset seeds 1/2 and multiple training
   seeds.
5. Decide whether the final claim needs pretrained k=450.
6. Report means, standard deviations, run counts, and individual seeds.

## Slide 15 - Conclusion

Put on the slide:

- CutMix leads pretrained ResNet50 at k=20 and k=50.
- SimMixUp leads at k=100 by 0.24 pp over CutMix.
- SimCutMix trails CutMix at all three pretrained budgets.
- Historical scratch evidence is more favorable but belongs to a different
  cohort.
- Multi-seed and controlled ablations are required.

Suggested closing:

> Similarity-guided mixing is not a universal improvement in the current
> evidence. Its behavior changes with the operator, rank window, data budget,
> architecture, and initialization. The k=100 pretrained SimMixUp result and
> the historical scratch results justify further controlled experiments, but
> the final conclusion must wait for multi-seed validation.

## Questions and Answers

### Are the classifiers pretrained?

> The current primary ResNet50 matrix is ImageNet-initialized and fine-tuned.
> A separate scratch group and scratch-trained ViT evidence also exist. The
> guidance encoder is always a separate frozen ImageNet ResNet50.

### Why different rank windows?

> The tracked pretrained SimMixUp uses ranks 1-10 and SimCutMix uses ranks
> 21-40. These are selected method-specific configurations, not proof that
> either window is globally optimal. The selection procedure must be documented
> and confirmed without test-set cherry-picking.

### Does guided mixing beat the standard method?

> Not consistently. SimMixUp beats MixUp at k=50 and k=100 but fails badly at
> k=20. SimCutMix trails CutMix at all three pretrained budgets.

### Is the 0.24-point k=100 lead meaningful?

> It is an observed single-run difference. Without repeated seeds and an
> uncertainty estimate, we cannot say whether it is reliable.

### Why retain scratch results?

> They provide historical and architecture-sensitive evidence, but their
> cohort is labeled and kept separate from the current pretrained comparison.

## Claims to Avoid

- "Similarity guidance always improves MixUp/CutMix."
- "The 0.24-point lead is significant."
- "SimCutMix isolates the effect of partner choice."
- "Ranks 1-10 or 21-40 are optimal."
- "Scratch and pretrained results are directly comparable or poolable."
- "A smaller train-validation gap proves that overfitting is solved."
- "Missing cells have zero accuracy."
