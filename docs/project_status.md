# SRP-Augmentation Project Status

Last updated: 2026-06-29

## Current Stage

The project is in the implementation and preliminary evaluation stage. Relative to
the proposal work plan, the repository has moved beyond baseline setup and now
contains the first implementation of the proposed Similarity-Guided Mixing idea.

The core experimental pipeline is in place for CIFAR low-data classification:

- Stratified k-shot splits exist for CIFAR-10 and CIFAR-100 at k = 5, 10, 20,
  50, and 100, with subset seeds 0, 1, and 2.
- A fixed validation split is used so training and validation samples do not
  overlap.
- The unified training script supports ResNet50 and ViT.
- Standard augmentation baselines are implemented: no augmentation, MixUp,
  CutMix, and AugMix.
- Similarity-guided variants are implemented: SimMixUp and SimCutMix.
- Neighbor construction is implemented using frozen ImageNet features.
- Guided pairing supports class-aware and class-agnostic neighbors, rank windows,
  uniform or weighted sampling, mix probability, warmup, and optional anchor
  selection based on uncertainty/rarity scores.
- Unit tests cover the guided dataset, SimMixUp, SimCutMix, neighbor payloads,
  rank-window handling, and anchor-score gating.

In short: the baseline framework is functional, and the proposed method is now
implemented enough to run first ablations.

## Results So Far

The current completed experiment results are mostly for CIFAR-100 with
`subset_seed=0`. This means the numbers are useful as preliminary evidence, but
they are not yet final research results because multi-seed aggregation is still
missing.

### 100-Epoch CIFAR-100 Baselines

CutMix is currently the strongest baseline across both ResNet50 and ViT for the
tested k values.

| Model | k | None | MixUp | CutMix | AugMix | Best |
|---|---:|---:|---:|---:|---:|---|
| ResNet50 | 20 | 11.75 | 14.68 | 15.46 | 13.55 | CutMix |
| ResNet50 | 50 | 22.32 | 25.62 | 27.04 | 26.69 | CutMix |
| ResNet50 | 100 | 31.88 | 36.67 | 38.75 | 37.15 | CutMix |
| ViT | 20 | 9.44 | 13.14 | 13.97 | 12.57 | CutMix |
| ViT | 50 | 16.24 | 20.36 | 20.75 | 19.29 | CutMix |
| ViT | 100 | 22.00 | 26.17 | 27.69 | 23.21 | CutMix |

Values are test accuracy percentages from the best validation checkpoint.

Main observations:

- Low-data CIFAR-100 is difficult, especially at k = 20.
- Plain training overfits heavily: training accuracy becomes high while validation
  and test accuracy remain low.
- MixUp and CutMix reduce overfitting and improve generalization.
- CutMix is consistently strongest in the current baseline grid.
- ViT underperforms ResNet50 in these from-scratch low-data runs, which matches
  the proposal expectation that ViTs are more data/recipe sensitive.

### First Similarity-Guided Results

The first guided experiments focus on CIFAR-100, ResNet50, k = 100,
`subset_seed=0`, mostly for 50 epochs. Because the main baseline table above is
100 epochs, comparisons should be treated carefully.

| Method | Setting | Epochs | Test Acc. |
|---|---|---:|---:|
| None | baseline | 50 | 29.19 |
| MixUp | baseline | 50 | 36.64 |
| SimMixUp | class-aware K20 | 50 | 31.75 |
| SimMixUp | class-agnostic K20 | 50 | 37.41 |
| SimMixUp | class-agnostic ranks 21-40 | 50 | 36.73 |
| SimCutMix | class-agnostic K20/K40 source | 50 | 38.65 |
| SimCutMix | anchor top 20 percent | 50 | 34.12 |
| SimCutMix | anchor random 20 percent | 50 | 32.62 |

Early interpretation:

- Class-agnostic SimMixUp slightly improves over the 50-epoch MixUp baseline.
- Class-aware SimMixUp performs much worse in the current run, suggesting that
  preserving labels too strictly may reduce useful diversity.
- Ungated class-agnostic SimCutMix is the most promising guided result so far:
  38.65 percent after 50 epochs, close to the 100-epoch CutMix baseline at
  38.75 percent.
- Anchor-gated SimCutMix did not help in the tested top-20-percent setup. The
  targeted version is currently worse than applying guided mixing to all anchors.

## Implemented Artifacts

Important outputs already exist:

- `results/experiments/baseline_100e/metrics/`: main 100-epoch baseline summaries and first guided runs.
- `results/experiments/initial_simmixup_v2/metrics/`: 50-epoch ResNet50 k=100 baseline and SimMixUp runs.
- `results/experiments/initial_simcutmix_v3/metrics/`: 50-epoch SimCutMix run.
- `results/experiments/shared/neighbors/cifar100/k100_seed0/`: ResNet50 ImageNet embeddings and
  class-aware/class-agnostic K40 neighbor files for k=100.
- `results/experiments/shared/neighbors/cifar100/k20_seed0/`: ResNet18 ImageNet embeddings and
  class-aware neighbors for k=20, with 19 effective same-class neighbors.
- `results/experiments/shared/anchor_scores/cifar100/k100_seed0/`: uncertainty/rarity anchor scores
  for targeted guided augmentation.

The neighbor payloads record split hashes and are built from the k-shot training
subset. One reproducibility issue should be fixed: the
`results/experiments/shared/neighbors/cifar100/k100_seed0/metadata.json` file appears stale, because
it points to ResNet18/K20 artifacts while the actual k=100 payloads are
ResNet50/K40.

## Open Problems and Risks

- The current results are mostly single-seed, so variance is unknown.
- Guided augmentation has mostly been tested on ResNet50 k = 100 only.
- ViT guided experiments are still missing.
- Some guided results use 50 epochs while the main baselines use 100 epochs, so
  the comparison grid should be normalized before final claims.
- Anchor-score targeting is implemented, but the first tested configuration hurt
  performance; it needs tuning or a stronger justification.
- The current neighbor artifacts use frozen ImageNet ResNet features
  (ResNet50 for k=100, ResNet18 for k=20). It may be worth testing whether a
  stronger or more domain-suitable encoder changes neighbor quality.
- The k=100 neighbor metadata file should be regenerated so metadata paths match
  the actual ResNet50/K40 payloads used by the guided runs.

## Suggested Next Tasks

1. Normalize the comparison protocol: choose 50 or 100 epochs for the main guided
   comparison and rerun all methods under the same budget.
2. Run multi-seed aggregation for CIFAR-100 k = 20, 50, and 100 across seeds 0,
   1, and 2.
3. Extend guided experiments to ViT to test the architecture-interaction
   hypothesis from the proposal.
4. Add CIFAR-10 experiments to check whether the method helps mainly in
   fine-grained CIFAR-100 or also in simpler low-data classification.
5. Run lower-shot settings k = 5 and k = 10, where harmful random mixing may be
   more visible.
6. Tune SimCutMix and SimMixUp ablations: neighbor count K, class-aware vs
   class-agnostic, rank windows, mix probability, warmup, and weighted sampling.
7. Revisit anchor selection with alternatives such as score-probability gating,
   different top percentages, and lower mix probabilities.
8. Produce final tables and plots from the JSON summaries, including mean,
   standard deviation, and best-validation/test accuracy.
9. Add qualitative diagnostics: inspect nearest neighbors, mixed images, and
   confusion matrices for cases where guided mixing helps or hurts.

## Bottom Line

The project has a solid experimental scaffold and has reached the first
prototype-results stage for the proposed method. The strongest established
baseline is CutMix. The most promising new result is class-agnostic SimCutMix on
CIFAR-100 ResNet50 k = 100, which nearly matches the 100-epoch CutMix result
after a 50-epoch run. The next major milestone is to turn these promising
single-run observations into fair, multi-seed, same-budget comparisons.
