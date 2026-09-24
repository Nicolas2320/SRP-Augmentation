# Presentation: Similarity-Guided Data Augmentation

## Main message

This project studies whether choosing a visually similar partner image can make
MixUp and CutMix more effective when labeled data is limited.

All CIFAR classifiers are trained from scratch. For SimMixUp and SimCutMix only,
a frozen ImageNet-pretrained ResNet50 is used to compute image embeddings and
select partner images. The project is not a comparison between scratch-trained
and pretrained classifiers.

Recommended length: 12-15 minutes.

---

## Slide 1 - Title

### Put on the slide

**Similarity-Guided Data Augmentation for Low-Data Image Classification**

- CIFAR-100
- ResNet50 and ViT
- SimMixUp and SimCutMix

### Say this

> This project investigates whether similarity-guided image mixing can improve
> classification when only a small number of labeled examples are available.
> We compare standard augmentation methods with two proposed methods:
> SimMixUp and SimCutMix.

---

## Slide 2 - Motivation

### Put on the slide

**Why is low-data classification difficult?**

- Few labeled examples encourage memorization.
- Augmentation creates additional training variation.
- MixUp and CutMix usually select partner images randomly.
- Random partners may be uninformative or produce unrealistic combinations.

### Say this

> When the training set is small, a model can memorize the available images
> instead of learning features that generalize. Data augmentation helps, but
> standard MixUp and CutMix do not consider the visual relationship between the
> two images. Our idea is to improve the selection of the second image.

---

## Slide 3 - Research question

### Put on the slide

**Research question**

> Can similarity-guided mixing improve classification under limited data?

**Hypothesis**

> A moderately similar partner may create a more informative training example
> than a completely random or nearly identical partner.

### Say this

> We do not assume that similarity guidance always wins. We test the methods at
> several data budgets to see whether the benefit depends on how much labeled
> data is available.

---

## Slide 4 - Methods

### Put on the slide

| Group | Methods |
|---|---|
| Baseline | None: crop and horizontal flip only |
| Standard augmentation | MixUp, CutMix, AugMix |
| Proposed augmentation | SimMixUp, SimCutMix |

### Say this

> The proposed methods keep the standard MixUp and CutMix operations. The main
> change is how the second image is selected. Standard methods use random
> partners, while the proposed methods use similarity-guided partners.

---

## Slide 5 - Scratch classifier and pretrained encoder

### Put on the slide

| Component | Initialization | Purpose |
|---|---|---|
| CIFAR ResNet50 / ViT | Random initialization | Classifier that is evaluated |
| ImageNet ResNet50 encoder | Pretrained and frozen | Selects similar image partners |

### Say this

> There are two different roles. The CIFAR classifier produces the reported
> accuracy and is trained from scratch for every method. The separate ImageNet
> ResNet50 is frozen and is used only as a feature extractor to measure image
> similarity. Therefore, the results are not scratch-versus-pretrained classifier
> results.

---

## Slide 6 - Similarity-guided pipeline and final configuration

### Put on the slide

```text
CIFAR k-shot subset
        |
        v
ImageNet-pretrained ResNet50
        |
        v
Image embeddings and exact neighbors
        |
        v
Class-agnostic ranks 21-40
        |
        v
Uniform partner selection
        |
        v
SimMixUp or SimCutMix
        |
        v
CIFAR classifier trained from scratch
```

**Final guided configuration**

- Neighbor mode: class-agnostic
- Saved neighbor pool: K=40
- Training window: ranks 21-40
- Sampling: uniform
- Mixing probability: 1.0
- Warm-up: 0 epochs

### Include this figure

Use the architecture diagram from `docs/architecture.md` or a simplified version
of the pipeline above. You may also use a guided-pair example from
`notebooks/02_visualize_guided_pairs.ipynb`.

### Say this

> We chose the class-agnostic mode because the partner should be selected by
> visual similarity rather than by its label. This allows both same-class and
> different-class relationships.
>
> We save the 40 nearest neighbors but use ranks 21 through 40 for training.
> Ranks 1 through 20 may be too close or redundant, while ranks 21 through 40
> provide similarity with more diversity. We sample uniformly, apply mixing with
> probability one, and use no warm-up so the guided method is active from the
> beginning.

---

## Slide 7 - Experimental setup

### Put on the slide

- Dataset: CIFAR-100
- Classifier models: ResNet50 and ViT
- Data budgets: k=20, 50, 100, 450 images per class
- Fixed validation split
- Best checkpoint selected by validation accuracy
- Test evaluation performed using that best checkpoint
- Current results: one subset seed and one training seed

### Say this

> The main variable is the augmentation method. The data split, model, training
> procedure, and checkpoint-selection procedure are controlled within each
> comparison. The reported results are single-run results, so multi-seed
> validation is still required.

---

# ResNet50 results

## Slide 8 - Complete ResNet50 result overview

### Put on the slide

**CIFAR-100 ResNet50 test accuracy**

| Data budget | None | MixUp | CutMix | SimMixUp | SimCutMix |
|---:|---:|---:|---:|---:|---:|
| k=20 | 16.26% | 13.67% | 14.89% | **16.73%** | 16.40% |
| k=50 | 28.49% | 23.65% | 25.80% | 28.30% | **28.65%** |
| k=100 | 41.96% | 41.83% | 43.20% | 44.60% | **46.16%** |
| k=450 | 71.07% | 69.04% | **72.70%** | Not available | 71.60% |

### Include this graph

Use `docs/figures/available_test_accuracy_vs_k.png`.

### Say this exactly

> I will begin with the ResNet50 results. Here, k represents the number of
> labeled training images available per class. Therefore, k=20 is the most
> limited-data setting, while k=450 provides substantially more data.
>
> This table and graph show the available test accuracy for the baseline and
> proposed methods. At k=20, the best result is SimMixUp at 16.73 percent. At
> k=50, the best result is SimCutMix at 28.65 percent. At k=100, SimCutMix is
> again the best method at 46.16 percent. At k=450, standard CutMix is the best
> recorded method at 72.70 percent.
>
> The general pattern is that performance increases as the data budget grows,
> but the best augmentation method depends on the data regime.

---

## Slide 9 - Detailed ResNet50 analysis: k=20

### Include this graph

Keep `available_test_accuracy_vs_k.png` on screen and highlight the k=20 group.

### Key values

- None: 16.26%
- MixUp: 13.67%
- CutMix: 14.89%
- SimMixUp: 16.73%
- SimCutMix: 16.40%

### Important differences

- SimMixUp vs MixUp: **+3.06 percentage points**
- SimMixUp vs CutMix: **+1.84 points**
- SimCutMix vs MixUp: **+2.73 points**
- SimCutMix vs CutMix: **+1.51 points**

### Say this exactly

> At k=20, the data budget is extremely small. SimMixUp reaches 16.73 percent,
> which is the best result in this row. It is 3.06 percentage points above MixUp
> and 1.84 points above CutMix. SimCutMix also outperforms both standard mixing
> methods.
>
> Compared with None, the proposed methods are very close. SimMixUp is 0.47
> points higher and SimCutMix is 0.14 points higher. Therefore, the main result
> at k=20 is that similarity-guided mixing avoids the performance drop observed
> with random MixUp and CutMix, rather than producing a very large gain over no
> mixing.

---

## Slide 10 - Detailed ResNet50 analysis: k=50

### Include this graph

Keep `available_test_accuracy_vs_k.png` on screen and highlight the k=50 group.

### Key values

- None: 28.49%
- MixUp: 23.65%
- CutMix: 25.80%
- SimMixUp: 28.30%
- SimCutMix: 28.65%

### Important differences

- SimMixUp vs MixUp: **+4.65 percentage points**
- SimMixUp vs CutMix: **+2.50 points**
- SimCutMix vs MixUp: **+5.00 points**
- SimCutMix vs CutMix: **+2.85 points**

### Say this exactly

> At k=50, the proposed methods show their clearest advantage over standard
> random mixing. MixUp reaches 23.65 percent and CutMix reaches 25.80 percent.
> SimMixUp reaches 28.30 percent, while SimCutMix reaches 28.65 percent.
>
> SimMixUp is therefore 4.65 points above MixUp and 2.50 points above CutMix.
> SimCutMix is 5.00 points above MixUp and 2.85 points above CutMix.
>
> This supports the idea that partner selection matters in the low-data setting.
> The proposed methods are much stronger than random mixing. At the same time,
> they are close to None: SimMixUp is 0.19 points lower and SimCutMix is 0.16
> points higher. The strongest claim is therefore about improvement over random
> mixing, not that augmentation always beats no augmentation.

---

## Slide 11 - Detailed ResNet50 analysis: k=100

### Include this graph

Use `docs/figures/matched_test_accuracy.png` and highlight the ResNet50 k=100
panel.

### Key values

- None: 41.96%
- MixUp: 41.83%
- CutMix: 43.20%
- SimMixUp: 44.60%
- SimCutMix: **46.16%**

### Important differences

- SimMixUp vs None: **+2.64 percentage points**
- SimMixUp vs MixUp: **+2.77 points**
- SimMixUp vs CutMix: **+1.40 points**
- SimCutMix vs None: **+4.20 points**
- SimCutMix vs MixUp: **+4.33 points**
- SimCutMix vs CutMix: **+2.96 points**

### Say this exactly

> At k=100, both proposed methods outperform all three available baselines.
> SimMixUp reaches 44.60 percent and SimCutMix reaches 46.16 percent.
>
> The strongest direct comparison is SimCutMix against CutMix: 46.16 percent
> compared with 43.20 percent. This is an improvement of 2.96 percentage
> points. SimCutMix is also 4.20 points above None and 4.33 points above MixUp.
>
> This is the clearest evidence that similarity-guided partner selection can
> improve classification accuracy. However, it remains a single-run result and
> must be validated across multiple seeds.

---

## Slide 12 - Detailed ResNet50 analysis: k=450

### Include this graph

Use `available_test_accuracy_vs_k.png` and highlight the k=450 group.

### Key values

- None: 71.07%
- MixUp: 69.04%
- CutMix: **72.70%**
- SimCutMix: 71.60%
- SimMixUp: not available

### Important differences

- SimCutMix vs None: +0.53 points
- SimCutMix vs MixUp: +2.56 points
- SimCutMix vs CutMix: **-1.10 points**

### Say this exactly

> At k=450, SimCutMix reaches 71.60 percent. It is higher than None by 0.53
> points and higher than MixUp by 2.56 points, but it is 1.10 points below
> CutMix, which reaches 72.70 percent.
>
> This is an important negative result. Similarity-guided mixing is not always
> better than standard CutMix. When more labeled data is available, standard
> CutMix may already provide enough useful variation. This is an interpretation,
> not a confirmed causal explanation.

---

## Slide 13 - ResNet50 overall interpretation

### Include this graph

Use `available_test_accuracy_vs_k.png` again, or use a simple table of the
percentage-point differences from Slides 9-12.

### Say this exactly

> Across the data budgets, the proposed methods are consistently stronger than
> random MixUp and usually stronger than standard CutMix. The largest advantage
> over MixUp is at k=50, where SimCutMix is 5.00 percentage points higher. The
> strongest absolute result is SimCutMix at k=100 with 46.16 percent.
>
> The comparison with None is more nuanced. At k=20 and k=50, the proposed
> methods are approximately tied with None. At k=100, SimCutMix clearly exceeds
> None. At k=450, standard CutMix is the best method.
>
> Therefore, our conclusion is not that similarity-guided augmentation always
> wins. The more precise conclusion is that it is especially promising when
> labeled data is limited and can substantially improve over random mixing.

---

# ViT results

## Slide 14 - ViT result overview

### Put on the slide

**CIFAR-100 ViT test accuracy**

| Data budget | None | MixUp | CutMix | SimMixUp | SimCutMix |
|---:|---:|---:|---:|---:|---:|
| k=20 | 12.64% | 12.28% | 12.43% | **12.96%** | 12.42% |
| k=50 | **21.12%** | 19.42% | 18.72% | 20.66% | 20.57% |
| k=100 | **29.83%** | 24.93% | 27.85% | 27.10% | 27.85% |

### Include this graph

Use the ViT panel of `docs/figures/available_test_accuracy_vs_k.png`.

### Say this exactly

> The ViT results show a different pattern from ResNet50. At k=20, SimMixUp is
> the best recorded method at 12.96 percent. At k=50 and k=100, the None
> baseline is strongest at 21.12 and 29.83 percent.
>
> This means the similarity-guided methods are not universally best across model
> architectures. The result depends on both the data budget and the classifier
> architecture. These ViT results are also single-run results and need multi-seed
> validation.

---

# Training behavior and generalization

## Slide 15 - Validation curves

### Include this graph

Use `docs/figures/matched_validation_curves.png`.

### What the graph shows

- Horizontal axis: training epoch.
- Vertical axis: validation accuracy.
- Each panel represents a model and data budget.
- Each curve represents an augmentation method.
- The star marks that run's best validation checkpoint.

### Put on the slide

**Validation accuracy during training**

> The best validation checkpoint, not the final epoch, determines the reported
> test result.

### Say this exactly

> This figure shows how validation accuracy changes during training. The x-axis
> is the epoch and the y-axis is validation accuracy. Each line represents one
> augmentation method under the same model and data budget.
>
> The star on each curve marks the epoch with the highest validation accuracy.
> After training, we reload that best-validation checkpoint and evaluate it once
> on the test set. Therefore, the test accuracy reported in the results table is
> linked to the validation-selected checkpoint rather than automatically to the
> last training epoch.
>
> This is important because a model can continue improving on the training data
> after its validation performance has stopped improving. Selecting the best
> validation checkpoint gives us a consistent model-selection rule without
> using the test set to choose the epoch.

### How to interpret the curves

> When a curve rises and then levels off, the model is no longer gaining much
> validation performance. When training continues but validation accuracy does
> not improve, this is a possible sign that the model is beginning to memorize
> the training subset.
>
> The proposal curves should be interpreted relative to the baselines in the
> same panel. We do not compare a curve from one data budget with a curve from a
> different data budget, because the training difficulty is different.

### What the current curves suggest

Use the same graph and point to the panels one at a time:

> In the k=20 panel, the validation accuracies are low and the methods are
> relatively close together. SimCutMix reaches a validation accuracy of 16.32
> percent and SimMixUp reaches 16.18 percent. This tells us that the proposed
> methods are competitive in the most data-limited setting, but the curves do
> not show a dramatic separation.

> In the k=50 panel, SimCutMix reaches 29.64 percent validation accuracy. Its
> curve reaches a higher validation peak than the standard MixUp and CutMix
> curves in this comparison. This agrees with the test-accuracy result, where
> SimCutMix also performs strongly at k=50.

> In the k=100 panel, the separation is clearest. SimCutMix reaches a best
> validation accuracy of 46.40 percent, while CutMix reaches 44.06 percent.
> The SimCutMix star is therefore higher than the CutMix star. This is the
> validation-side evidence supporting the test-accuracy improvement at k=100.

> In the k=450 panel, standard CutMix reaches the strongest validation result at
> 72.58 percent, while SimCutMix reaches 71.48 percent. This agrees with the
> test results: at the larger data budget, standard CutMix is stronger than the
> similarity-guided method in this run.

### The correct interpretation

> The validation curves show that the positive k=100 result is not caused only
> by a lucky test evaluation at the final epoch. SimCutMix also reaches the
> strongest validation peak in that panel, and that checkpoint is then tested.
>
> At the same time, the curves show that the proposal is not universally best.
> At k=450, CutMix has the stronger validation and test result. The validation
> evidence therefore supports a data-regime-dependent conclusion rather than a
> claim that SimCutMix always wins.

### What this graph does not prove

> This graph does not by itself prove that SimCutMix prevents overfitting. It
> shows validation performance over epochs. To discuss the difference between
> training performance and validation performance, we need the separate
> train-validation-gap graph.

### Transition

> The validation curves tell us which checkpoint generalizes best. The next
> figure asks a different question: how large is the separation between training
> performance and validation performance at that checkpoint?

## Slide 16 - Train-validation gap

### Include this graph

Use `docs/figures/matched_overfitting_gap.png`.

### Define the metric on the slide

```text
Train-validation gap = training accuracy - validation accuracy
```

### What the graph shows

- Horizontal axis: training epoch.
- Vertical axis: train accuracy minus validation accuracy.
- A larger positive gap indicates more separation between fitting the training
        data and performance on validation data.
- The star marks the gap at the best-validation checkpoint.

### Put on the slide

**ResNet50, k=100**

| Method | Gap at best-validation checkpoint |
|---|---:|
| CutMix | 16.88 percentage points |
| SimCutMix | **9.68 percentage points** |

### Say this exactly

> This figure measures the train-validation gap, calculated as training accuracy
> minus validation accuracy. A large positive gap means that the model performs
> much better on the training subset than on validation data, which is
> consistent with stronger memorization.
>
> For ResNet50 with k=100, the CutMix gap at its best-validation checkpoint is
> 16.88 percentage points. The SimCutMix gap is 9.68 points. SimCutMix therefore
> has a gap that is 7.20 points smaller than CutMix in this comparison.
>
> Combined with the test results, SimCutMix reaches 46.16 percent compared with
> 43.20 percent for CutMix, while also showing a smaller train-validation gap.
> This is consistent with better generalization and less memorization in this
> particular experiment.

### Important fairness caveat

> The gap comparison must be interpreted carefully. For None, training accuracy
> is ordinary clean-label accuracy. For MixUp, CutMix, SimMixUp, and SimCutMix,
> training accuracy uses partial credit because each training example has mixed
> labels. Therefore, the absolute gap values are not perfectly identical
> measurements across all methods.

> Even with this caveat, the gap is useful as supporting evidence because the
> difference is large and the comparison between SimCutMix and CutMix uses the
> same training framework. We do not present it as proof by itself.

### What not to say

Do not say:

> SimCutMix completely eliminates overfitting.

Say instead:

> SimCutMix shows a smaller train-validation gap than CutMix in this experiment,
> which is consistent with less memorization.

### Transition to limitations

> These curves and gaps provide evidence about training behavior, but the current
> results are still single runs. The next slide explains what must be completed
> before making a final general conclusion.

---

# Limitations and conclusion

## Slide 17 - Limitations

### Put on the slide

- Current results use one subset seed and one training seed.
- No confidence intervals or multi-seed averages yet.
- Some method-model-budget combinations remain unavailable.
- SimMixUp is not available at ResNet50 k=450.
- The software environment is not fully locked.
- The study does not compare scratch-trained classifiers with pretrained
  classifiers.

### Include this graph

Use `docs/figures/experiment_coverage.png` to show completed and missing cells.

### Say this

> The current results are promising but preliminary. The main limitation is that
> each result currently comes from one subset seed and one training seed. The
> next step is to repeat the selected configurations across multiple seeds and
> report averages and variation.
>
## Slide 18 - Conclusion

### Put on the slide

**Conclusion**

- Similarity-guided mixing is promising in low-data settings.
- SimCutMix reaches 46.16% at ResNet50 k=100.
- At k=50, SimCutMix is 5.00 points above MixUp.
- At k=450, standard CutMix remains stronger.
- Multi-seed validation is required for final claims.

### Say this exactly

> To conclude, similarity-guided mixing shows promising behavior, especially in
> low-data settings. The strongest ResNet50 result is SimCutMix at k=100, where
> it reaches 46.16 percent and improves over CutMix by 2.96 percentage points.
> At k=50, the proposed methods also show a strong advantage over random MixUp.
>
> However, the method does not always win. At k=450, standard CutMix performs
> better. Therefore, our conclusion is that similarity-guided partner selection
> is a promising strategy for low-data classification, but its benefit is not
> universal and must be validated across multiple seeds.

---

## Opening script

> In our previous presentation, we introduced the problem of augmentation for
> low-data image classification. Since then, we implemented the full
> similarity-guided training pipeline and evaluated the proposed methods against
> standard augmentation baselines. Today I will explain the experimental design,
> present the ResNet50 results in detail, compare them with the ViT results, and
> finish with the limitations and next steps.

## Questions and answers

### Are the classifiers pretrained?

> No. The CIFAR ResNet50 and ViT classifiers are trained from scratch. A frozen
> ImageNet-pretrained ResNet50 is used only to compute embeddings and select
> similar partners for SimMixUp and SimCutMix.

### What does k mean?

> k is the number of labeled training images per class. Smaller k means a more
> limited-data setting.

### Why class-agnostic neighbors?

> We want partner selection to depend on visual similarity rather than a class
> restriction. This allows both same-class and different-class relationships.

### Why ranks 21-40?

> Ranks 1-20 may be too similar or redundant. Ranks 21-40 retain visual
> similarity while providing more diversity. This is the selected working
> configuration, not yet a proven globally optimal window.

### Does SimCutMix always outperform CutMix?

> No. It improves over CutMix by 2.96 points at ResNet50 k=100, but it loses by
> 1.10 points at k=450.

### Are the results averaged over several runs?

> Not yet. The current values use one subset seed and one training seed. We need
> multi-seed experiments before making final statistical claims.

### What is the main conclusion?

> Similarity-guided augmentation is most promising when labeled data is limited,
> particularly because it improves substantially over random MixUp and can beat
> CutMix at some budgets. Its advantage is not universal.

## Claims to avoid

- Do not say SimCutMix always wins.
- Do not call the single-run differences statistically significant.
- Do not describe the graphs as a scratch-versus-pretrained classifier study.
- Do not claim that ranks 21-40 are proven optimal.
- Do not interpret a missing bar as zero accuracy.
