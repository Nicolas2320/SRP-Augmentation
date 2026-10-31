# Minimal ViT Architecture-Transfer Experiment

Audience: teammate taking ownership of the optional ViT experiment.

## Goal

Test whether the ResNet50 pattern transfers to a different classifier
architecture. This is an architecture-sensitivity check, not a second full
replication of the ResNet50 matrix.

Do not reuse the existing historical ViT results as repeated seeds. They are
single-seed records from an earlier cohort and should remain appendix context.

## Frozen Minimal Matrix

Use CIFAR-100 and the existing scratch-trained CIFAR ViT implementation:
32×32 inputs, 4×4 patches, six transformer blocks, hidden dimension 256, and
eight attention heads.

Select two ResNet50 regimes:

- `k=50`: strongest positive SimCutMix-versus-CutMix result in the current
  scratch ResNet50 matrix (+3.17 percentage points).
- `k=300`: negative SimCutMix-versus-CutMix result in the current scratch
  ResNet50 matrix (−0.81 percentage points).

For each k, run:

- Crop+Flip (`none`)
- standard random-pair CutMix (`cutmix`)
- frozen SimCutMix (`simcutmix`), using class-agnostic ranks 21-40 and uniform
  sampling

Use paired `(subset_seed, train_seed)` values `(0,0)`, `(1,1)`, and `(2,2)`.
The matrix therefore contains 2 k values × 3 methods × 3 seeds = **18 runs**.

## Scientific Controls

- Keep the ViT architecture, optimizer, learning-rate schedule, epochs, batch
  size, crop/flip preprocessing, and weight decay identical across methods.
- Match CutMix and SimCutMix alpha and application probability. Use alpha 1
  and probability 0.5 so the comparison isolates partner selection.
- Use the same committed CIFAR-100 splits as the ResNet50 study.
- Use only training-subset images when building or loading neighbor payloads.
- The partner-selection encoder may remain the frozen ImageNet-pretrained
  ResNet50; clearly distinguish it from the scratch-trained ViT classifier.
- Select checkpoints by validation accuracy. Do not inspect test accuracy while
  changing the recipe.

## Execution Gates

1. Add or adapt a dedicated ViT Slurm runner; do not silently modify the frozen
   ResNet50 runner.
2. Run the complete CPU test suite.
3. Run one one-epoch smoke job for each of `none`, `cutmix`, and `simcutmix`.
4. Verify that SimCutMix logs ranks 21-40 and uses the intended neighbor file.
5. Run seed 0 for both k values as validation-only.
6. If all six seed-0 runs are valid, submit seeds 1 and 2 without changing the
   recipe.
7. Evaluate the frozen best-validation checkpoints once on the test set.

## Required Deliverables

- Per-run `metrics.csv`, `summary.json`, and best checkpoint.
- Git commit and runtime metadata in every summary.
- A table with mean ± sample SD for all six architecture/k/method cells.
- Paired SimCutMix-minus-CutMix and SimCutMix-minus-Crop+Flip deltas.
- A short conclusion stating whether the positive k=50 and negative k=300
  ResNet50 regimes transfer to ViT.

## Stop Conditions

Stop and report rather than changing the recipe if:

- the ViT job exceeds GPU memory;
- a required split or neighbor payload is missing;
- the three methods do not share the same base preprocessing/training recipe;
- a run attempts to overwrite an existing artifact;
- test results have been viewed before configuration freeze.

Tiny ImageNet ViT is outside this minimal task. Add it only after the CIFAR-100
architecture-transfer matrix is complete and reviewed.
