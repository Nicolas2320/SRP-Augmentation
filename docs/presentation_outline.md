# Final Presentation Outline: Similarity-Guided CutMix

Updated: 2026-11-14. Suggested length: 12-15 minutes.

## Main Message

Similarity-guided CutMix is useful in a specific regime: ResNet50 trained from
scratch with scarce or moderate labeled data. It is not a universal
improvement. Standard CutMix remains strongest with ImageNet initialization,
and changing pair difficulty over training does not beat a fixed rank window.

## Slide 1 — Title

**Similarity-Guided Data Augmentation for Low-Data Image Classification**

- CIFAR-100 and Tiny ImageNet
- ResNet50 trained from scratch or initialized from ImageNet
- Three paired seeds per final cell

## Slide 2 — Research Question

- Standard CutMix chooses the partner image uniformly at random.
- With few labels, arbitrary pairs may create unhelpful mixed examples.
- Question: does selecting a moderately similar partner improve CutMix?

State this as a tested hypothesis, not an established mechanism.

## Slide 3 — Proposed Method

```text
Committed k-shot split
        ↓
Frozen ImageNet ResNet50 embeddings
        ↓
Exact class-agnostic K40 neighbors
        ↓
Uniform sample from ranks 21-40
        ↓
Standard CutMix operation
        ↓
Classifier training and validation selection
```

The guidance encoder is separate from the classifier. A scratch classifier
may therefore use a frozen pretrained encoder only for partner selection.

## Slide 4 — Experimental Protocol

- Datasets: CIFAR-100 and Tiny ImageNet.
- Classifier: torchvision ResNet50, scratch and ImageNet-pretrained.
- CIFAR budgets: k=20, 50, 100, 200, 300, 450 images per class.
- Tiny ImageNet budgets: k=20, 100, 450.
- Best checkpoint selected only by validation accuracy.
- One test evaluation per frozen run.
- Three paired subset/training seeds per final cell.
- Methods: Crop+Flip, MixUp, CutMix, and SimCutMix on CIFAR-100.

Caveat: k=450 uses a predeclared 50-epoch recipe instead of 100 epochs.

## Slide 5 — Why These Baselines?

| Method | Role |
|---|---|
| Crop+Flip | Standard geometric augmentation reference |
| MixUp | Alternative random mixing baseline |
| CutMix | Matched operator baseline for SimCutMix |
| SimCutMix | Proposed similarity-guided partner selection |

AugMix is excluded from the main table because the local implementation omits
the canonical JSD consistency loss. SimMixUp remains exploratory and was not
promoted to the frozen three-seed matrix.

## Slide 6 — CIFAR-100 Scratch Results

| k | Crop+Flip | MixUp | CutMix | SimCutMix |
|---:|---:|---:|---:|---:|
| 20 | **17.51** | 14.13 | 15.54 | 16.72 |
| 50 | 31.15 | 27.49 | 30.09 | **33.26** |
| 100 | 52.17 | 50.49 | 51.94 | **52.91** |
| 200 | 66.10 | 67.11 | **68.09** | 67.82 |
| 300 | 72.32 | 73.30 | **74.76** | 73.95 |
| 450 | 70.81 | 69.51 | 71.99 | **72.40** |

Show uncertainty bars in the figure. The central result is the SimCutMix lead
at k=50 and k=100, not a claim that it wins every budget.

## Slide 7 — CIFAR-100 Pretrained Results

| k | Crop+Flip | MixUp | CutMix | SimCutMix |
|---:|---:|---:|---:|---:|
| 20 | 66.81 | 67.17 | **69.68** | 67.85 |
| 50 | 72.71 | 73.65 | **75.80** | 74.58 |
| 100 | 76.38 | 77.90 | **79.65** | 78.28 |
| 200 | 80.66 | 81.21 | **82.79** | 81.89 |
| 300 | 82.45 | 83.34 | **84.18** | 83.53 |
| 450 | 78.34 | 77.62 | **79.82** | 78.39 |

Standard CutMix leads at all six budgets. Similarity guidance is less useful
when the classifier already has strong ImageNet features.

## Slide 8 — Aggregate CIFAR Evidence

Across 36 matched initialization/k/seed comparisons:

- SimCutMix − Crop+Flip: +1.18 pp on average; wins 31/36.
- SimCutMix − CutMix: -0.23 pp on average; wins 13/36.
- MixUp − CutMix: -1.78 pp on average; wins 1/36.

Message: SimCutMix is a useful augmentation relative to Crop+Flip, but it does
not consistently replace standard CutMix.

## Slide 9 — Tiny ImageNet Generalization

| Initialization | k | Crop+Flip | CutMix | SimCutMix |
|---|---:|---:|---:|---:|
| Scratch | 20 | 12.71 | 12.00 | **14.87** |
| Scratch | 100 | 42.64 | 43.14 | **44.70** |
| Scratch | 450 | **60.94** | 60.56 | 60.49 |
| Pretrained | 20 | **67.49** | 66.84 | 65.65 |
| Pretrained | 100 | 65.95 | **68.77** | 67.69 |
| Pretrained | 450 | 65.44 | **65.82** | 65.54 |

Tiny ImageNet reproduces the scratch k=20/k=100 advantage and its disappearance
at full data. Standard CutMix leads pretrained k=100 by 1.08 ± 0.26 pp over
SimCutMix and wins all three paired seeds.

## Slide 10 — Pair-Selection Ablation

- The best rank window changes with budget and initialization.
- Ranks 11-20 lead scratch k=20 validation.
- Ranks 21-40 lead scratch k=100 validation.
- Random pairing leads scratch k=300 validation.

Conclusion: partner difficulty is regime-dependent. Ranks 21-40 are a frozen
primary configuration, not a globally optimal window.

## Slide 11 — Easy-to-Hard Curriculum

At scratch k=20 test accuracy:

| Condition | Accuracy |
|---|---:|
| Crop+Flip | **17.51 ± 0.45** |
| Random CutMix | 15.54 ± 0.59 |
| Fixed ranks 21-40, p=0.5 | 17.18 ± 0.76 |
| Easy-to-hard curriculum, p=0.5 | 17.10 ± 0.85 |

Curriculum − fixed is -0.07 ± 0.11 pp and wins 0/3 seeds. This is a useful
negative result: schedule complexity is not justified by the evidence.

## Slide 12 — Limitations

- The frozen SimCutMix primary matrix uses mixing probability 1, while standard
  CutMix uses 0.5; the separate pairing ablation controls this probability.
- k=450 uses a different epoch/learning-rate recipe.
- No paper-faithful AugMix confirmation.
- ViT transfer is delegated and optional, not part of the current claim.
- Means and standard deviations describe three seeds; do not call small
  differences statistically significant.

## Slide 13 — Conclusion

Put on the slide:

- SimCutMix helps scratch ResNet50 at low/moderate data budgets.
- Tiny ImageNet reproduces this regime-specific advantage.
- Standard CutMix is strongest with ImageNet-pretrained classifiers.
- MixUp is competitive but never leads an aggregate final cell.
- Easy-to-hard pairing does not improve over a fixed window.

Suggested closing:

> Similarity guidance is not a universal replacement for random pairing. It is
> most useful when the classifier must learn from scratch with limited labels,
> while strong pretraining and abundant data reduce or remove its advantage.

## Slide 14 — Remaining Work

- Generate final figures directly from the audited summaries.
- Optionally run the delegated minimal ViT transfer matrix.
- Leave canonical AugMix and denser Tiny ImageNet budgets as future work.

## Questions and Answers

### Are the classifiers pretrained?

Both regimes are reported separately. The guidance encoder is always a
separate frozen ImageNet ResNet50.

### Why ranks 21-40?

They were frozen from validation-based screening and work well at the primary
moderate-data scratch regime. The ablation shows they are not universally
optimal.

### Does guided pairing beat standard CutMix?

In selected scratch low/moderate-data regimes, yes. Across the full matrix,
no: SimCutMix averages 0.23 pp below CutMix and pretrained CutMix wins every k.

### Did the curriculum help?

No. It was essentially tied on validation and slightly below the matched fixed
window on the final k=20 test comparison.

## Claims to Avoid

- “Similarity guidance always improves CutMix.”
- “Ranks 21-40 are globally optimal.”
- “The curriculum improves performance.”
- “The k=450 drop proves that more data hurts.”
- “The incomplete Tiny ImageNet k=100 pretrained CutMix mean is final.”
- “Small three-seed differences are statistically significant.”
