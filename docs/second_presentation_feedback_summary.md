# Feedback Summary and Focus for the Third Presentation

## Overall Feedback

The third presentation should provide stronger experimental evidence that the proposed augmentation method works consistently—not only for one dataset, one data regime, or one training setup.

The goal is to support each conclusion through multiple experiments and show that the results are not a coincidence caused by a particular dataset or value of `K`.

## Main Priorities

### 1. Evaluate Both Training Settings

Report results for:

- Training from scratch
- Using a pretrained model

These settings demonstrate different effects. A pretrained model has already benefited from a larger pretraining dataset, so the additional benefit of augmentation may be smaller. Training from scratch provides a cleaner evaluation of the augmentation method itself.

If the proposed method does not outperform the baselines with a pretrained model, this can still be reported as a useful negative result. The presentation should explain that pretraining may reduce the marginal benefit of augmentation.

### 2. Add Tiny ImageNet

Tiny ImageNet was suggested as an additional dataset because it is larger and more challenging than CIFAR-100 while remaining feasible to train on the cluster.

Relevant characteristics mentioned in the feedback:

- 200 classes
- Approximately 100,000 training images
- 500 images per class
- Lower image resolution than full ImageNet

Testing on Tiny ImageNet would help demonstrate that the conclusions are not specific to CIFAR-100.

### 3. Investigate More Data Regimes

For training from scratch, test additional dataset sizes between the current small subsets and the full setting. The feedback referred to values between approximately 100 and 450, although the exact definition of `K` should be confirmed.

The main research question is:

> At what dataset size does the proposed augmentation begin to outperform the standard baselines?

A plot of accuracy versus training-set size would be more informative than results at only a few isolated values such as `K = 20` and `K = 50`.

## Recommended Experiment Matrix

| Dimension | Conditions |
|---|---|
| Datasets | CIFAR-100 and Tiny ImageNet |
| Initialization | From scratch and pretrained |
| Data sizes | Existing small `K` values plus intermediate and larger values |
| Methods | No augmentation, standard baselines, and the proposed method |
| Seeds | Three seeds for final configurations |
| Reporting | Mean accuracy and standard deviation |

### Seed and Hyperparameter Procedure

1. Use one seed while selecting the learning rate, weight decay, and other hyperparameters.
2. Once the best configuration is identified, run it with two additional seeds.
3. Report the mean and standard deviation across the three seeds.

The comparison should use a fair and clearly explained tuning procedure across methods.

## Important Ablation Experiments

Explicitly compare how the second class or image is selected for augmentation:

- Same class
- Most similar class
- Less-similar or distant class
- Random class, if applicable
- The similarity-rank interval used by the proposed method

Some of these experiments may already have been completed, but they were not clearly presented during the second presentation. They should be included and explained in the third presentation.

An example table is:

| Pair-selection strategy | Accuracy | Interpretation |
|---|---:|---|
| Same class | TBD | May generate insufficient diversity |
| Most similar class | TBD | Produces difficult but semantically meaningful examples |
| Distant class | TBD | May be easier to separate but produce less realistic mixtures |
| Random class | TBD | Control experiment |
| Proposed rank range | TBD | Intended balance between similarity and diversity |

## Curriculum Experiment

Lars suggested changing the difficulty of the augmented examples during training:

- Early epochs: mix distant or less-similar classes to create easier examples.
- Later epochs: progressively move toward more similar classes to create harder examples.

Two possible implementations were discussed:

### Stepwise Curriculum

Change the similarity-rank interval at fixed epochs. For example:

- Early training: ranks 21–40
- Middle training: ranks 11–30
- Late training: ranks 1–10

The exact intervals and transition epochs should be adapted to the current ranking system.

### Smooth Curriculum

Gradually decrease the selected similarity rank throughout training instead of changing it at fixed thresholds.

The curriculum should be compared with:

- A fixed rank range
- Random rank selection
- The best current version of the proposed method

This is a useful extension, but it should be treated as a secondary experiment. The core comparisons across datasets, data sizes, initialization settings, and seeds should be completed first.

## Verify the Similarity Computation

The presentation should clearly explain how similarity between classes or images is calculated.

The feedback specifically suggested using representations from the model's latent or feature space instead of final prediction probabilities or logits.

The method description should state:

- Which network layer provides the embeddings
- Whether the embeddings come from a pretrained or currently trained model
- How class representations are calculated
- Which similarity metric is used, such as cosine similarity
- Whether the similarity ranking is fixed or updated during training

If feasible, compare latent-space similarity with the current similarity definition.

## Prove the Small-Data Claim

If the project claims that standard augmentation methods such as Mixup or CutMix perform poorly with small datasets, the presentation should explicitly show the supporting evidence.

Recommended evidence:

- Show their results at small values such as `K = 20` and `K = 50`.
- Compare them with no augmentation and the proposed method.
- Report both absolute accuracy and improvement over the no-augmentation baseline.
- State that the methods underperform in the tested low-data regimes rather than claiming they are universally ineffective.

An accuracy-versus-training-set-size plot would communicate this evidence clearly.

## Priority Order

### Must Have

1. Organize and present the existing same-class, similar-class, and distant-class results.
2. Run experiments with intermediate dataset sizes.
3. Include both training-from-scratch and pretrained settings.
4. Report final results using three seeds and include standard deviations.
5. Add Tiny ImageNet results for at least the most important configurations.
6. Clearly document how similarity is calculated.
7. Compare the proposed method against no augmentation and established baselines.

### Nice to Have

1. Easy-to-hard similarity curriculum.
2. Latent-space versus prediction-based similarity comparison.
3. Additional curriculum schedules.
4. More extensive hyperparameter-sensitivity analysis.

## Suggested Structure for the Third Presentation

1. **Problem:** Existing augmentation methods may be ineffective or unstable when little training data is available.
2. **Hypothesis:** Selecting semantically related classes for mixing creates more useful augmented examples.
3. **Method:** Explain the embeddings, similarity ranking, and class or image selection process.
4. **Ablations:** Compare same-class, similar-class, distant-class, and random-class selection.
5. **Core results:** Show accuracy across several data sizes on CIFAR-100.
6. **Generalization:** Present the Tiny ImageNet results.
7. **Initialization effect:** Compare training from scratch with pretrained models.
8. **Reliability:** Report the mean and standard deviation across three seeds.
9. **Optional extension:** Present the easy-to-hard similarity curriculum.
10. **Conclusion:** State precisely where the method helps, where it does not, and the likely reasons.

## Details to Confirm

Before finalizing the experimental plan, confirm:

- What `K` represents and which values are feasible
- Which baseline augmentation methods are currently implemented
- How similarity is currently calculated
- Which experiments have already been completed
- How much cluster time is available before the third presentation

