# Recommended Job Matrix for the Third Presentation

This document turns the supervisor feedback in
[`second_presentation_feedback_summary.md`](../second_presentation_feedback_summary.md)
into a focused, staged cluster plan. It is intended as the handoff document for
continuing the work in a new chat.

## Decision Summary

The earlier idea of running only pretrained CIFAR-100 repeats is not sufficient
for the third presentation. The supervisor expects evidence across:

- training from scratch and ImageNet initialization;
- several data regimes, including intermediate sizes;
- three seeds for final configurations;
- explicit pair-selection ablations;
- a second dataset, preferably Tiny ImageNet;
- a precise description of the similarity computation.

The recommended order is:

1. Build one fair, current-protocol CIFAR-100 size curve at seed 0.
2. Run a focused pair-selection ablation using one mixing operator.
3. Freeze the proposed configuration using validation evidence only.
4. Repeat the frozen comparison for three paired seeds.
5. Implement and validate Tiny ImageNet support.
6. Run a small Tiny ImageNet generalization matrix.
7. Attempt curriculum experiments only if the must-have evidence is complete.

Do not submit a large job array until the corresponding implementation gate and
small smoke test have passed.

## Current Repository Facts

- The current training CLI supports CIFAR-10 and CIFAR-100, not Tiny ImageNet.
- Current CIFAR-100 splits exist for k=20/50/100 with subset seeds 0/1/2 and
  k=450 with seed 0. Additional intermediate splits are required.
- `k` means labeled training images per class after removing the fixed
  validation indices.
- The current ResNet50 protocol uses 224x224 inputs for both scratch and
  pretrained initialization.
- Existing scratch evidence mixes historical and current architecture cohorts.
  It must not be treated as one fair current-protocol matrix.
- Existing pretrained results cover k=20/50/100 at seed 0, but some guided
  settings differ from the current Slurm runner.
- Guided support payloads are local generated artifacts and must be rebuilt for
  every dataset, k, and subset seed used by a guided run.
- The cluster setup, CUDA smoke run, and all 74 unit tests have passed.

## Definitions and Reporting Rules

### Baseline names

- `raw`: deterministic resize and normalization only.
- `none`: random crop and horizontal flip, with no additional augmentation.
- `mixup`: random partner MixUp.
- `cutmix`: random partner CutMix.
- `simmixup`: similarity-guided MixUp.
- `simcutmix`: similarity-guided CutMix.

For the main comparison, `none` is the practical basic-augmentation baseline.
Use `raw` as a diagnostic/reference condition, not as the only meaning of “no
augmentation.” Explain this distinction in the presentation.

### Seed policy

Use paired seeds:

| Replicate | Subset seed | Training seed |
|---:|---:|---:|
| 1 | 0 | 0 |
| 2 | 1 | 1 |
| 3 | 2 | 2 |

This estimates end-to-end variability from both labeled-sample selection and
optimization. It does not separate those two variance sources. A 3x3 factorial
design would do so, but its cost is not justified before the presentation.

Use seed `(0,0)` for configuration screening. Freeze every material setting
before running seeds `(1,1)` and `(2,2)`. Report individual runs, mean, standard
deviation, and `n=3`.

### Selection policy

- Select configurations using validation accuracy and validation curves only.
- Do not use test accuracy to choose a method, rank window, mixing probability,
  learning rate, or stopping epoch.
- Evaluate the test set once for each frozen final run using its best-validation
  checkpoint.
- Keep smoke runs under `results/smoke/`; they are not scientific evidence.

## Phase 1 — Current-Protocol CIFAR-100 Size Curve

### Seed-0 screening matrix

| Dimension | Values |
|---|---|
| Dataset | CIFAR-100 |
| Initialization | `scratch`, `pretrained` |
| k | 20, 50, 100, 200, 300, 450 |
| Methods | `none`, `mixup`, `cutmix`, `simmixup`, `simcutmix` |
| Seeds | `(0,0)` |
| Selection | Validation only |

This matrix contains 60 cells. Do not automatically rerun all 60:

1. Build a manifest from existing summaries.
2. Reuse a cell only when architecture, preprocessing, optimizer recipe,
   method settings, initialization, k, and seeds match exactly.
3. Exclude historical scratch cohorts from the current-protocol curve.
4. Submit only missing or incompatible cells.

Conservative upper bound before reuse: 60 runs. Expected new work is lower for
pretrained k=20/50/100, but most scratch cells may require rerunning because of
the historical 32x32/current 224x224 cohort split.

### Why these k values

`20, 50, 100` preserve the established low-data evidence. `200` and `300` add
the intermediate regimes explicitly requested by the supervisor. `450` is the
largest balanced CIFAR-100 training subset available after the fixed validation
split. Together they support an accuracy-versus-training-size curve.

Create committed split files for k=200 and k=300 with subset seeds 0/1/2. Audit
class balance, fixed-validation exclusion, determinism, and nesting policy
before training.

### Core method scope

The five methods above keep the matrix focused on the research question. `raw`
and `augmix` may remain in supplementary seed-0 plots using compatible existing
records, but they should not consume three-seed budget unless requested later.

## Phase 2 — Pair-Selection Ablation

Use CutMix as the single ablation vehicle because it is the strongest and most
stable standard baseline in the current validation evidence. Holding the mixing
operator fixed makes the partner-selection comparison easier to interpret.

### Primary ablation matrix

| Dimension | Values |
|---|---|
| Dataset | CIFAR-100 |
| Initialization | `scratch` primarily; `pretrained` at one representative k |
| k, scratch | 20, 100, 300 |
| k, pretrained | 100 |
| Seed | `(0,0)` |
| Mixing operator | CutMix |
| Partner strategies | See table below |

| Strategy | Repository interpretation | Implementation status |
|---|---|---|
| Random | Standard `cutmix` | Implemented; reuse Phase 1 |
| Same class | `class_aware` neighbors | Implemented |
| Most similar | Class-agnostic ranks 1–10 | Implemented |
| Middle similarity | Class-agnostic ranks 11–20 | Implemented |
| Less similar proposed range | Class-agnostic ranks 21–40 | Implemented |
| Different label | `different_label` nearest eligible neighbors | Implemented |
| Truly most distant | Lowest cosine-similarity partners | Not implemented |

The implemented “different label” condition is not automatically the same as
“most distant.” Label it accurately. Add a farthest-neighbor mode only if the
supervisor explicitly requires true distance ordering.

The random baseline is already part of Phase 1. The implemented guided
conditions require at most 16 additional seed-0 runs: four new strategies at
three scratch k values plus four at pretrained k=100. Add the truly distant
condition only after implementation and tests.

Freeze one rank/mode/sampling configuration from this phase using validation
evidence. Do not select a different proposed configuration independently for
every seed.

## Phase 3 — Three-Seed Confirmatory CIFAR-100 Matrix

After Phase 2, compare exactly three conditions:

1. `none` — basic crop-and-flip baseline;
2. `cutmix` — matched random-pair established baseline;
3. frozen `simcutmix` — proposed pair-selection configuration.

### Preferred confirmation matrix

| Dimension | Values |
|---|---|
| Dataset | CIFAR-100 |
| Initialization | `scratch`, `pretrained` |
| k | 20, 50, 100, 200, 300, 450 |
| Methods | `none`, `cutmix`, frozen `simcutmix` |
| Seeds | `(0,0)`, `(1,1)`, `(2,2)` |

Total final cells: `2 × 6 × 3 × 3 = 108`. Seed-0 cells come from Phase 1, so
the maximum additional confirmation cost is 72 jobs.

### Reduced matrix if cluster time is insufficient

Use k=20, 100, 300, and 450 while retaining both initialization settings and
all three seeds. This gives 72 total cells and at most 48 additional jobs after
seed-0 screening. Do not reduce to one initialization or one seed; those are
explicit supervisor priorities.

### Optional MixUp-family confirmation

If Phase 1 shows that SimMixUp has a consistent validation advantage not
captured by SimCutMix, add a separate matched pair:

- `mixup` versus a frozen `simmixup` configuration;
- preferably at only the k values where the validation advantage was observed;
- still using all three paired seeds.

Keep this extension separate from the primary CutMix-family claim.

## Phase 4 — Tiny ImageNet Generalization

Tiny ImageNet is a must-have target from the feedback, but it is currently an
engineering task, not a runnable matrix. Complete this gate first:

1. Add and test dataset loading and canonical class mapping.
2. Define the train/validation/test policy without test leakage.
3. Define exactly what k means for this dataset.
4. Generate deterministic per-class splits for seeds 0/1/2.
5. Extend training, embeddings, neighbor generation, and artifact metadata.
6. Cache the dataset and pretrained weights before compute-node execution.
7. Run CPU tests and a one-epoch GPU smoke job.

### Provisional Tiny ImageNet screening matrix

| Dimension | Values |
|---|---|
| Dataset | Tiny ImageNet |
| Initialization | `scratch`, `pretrained` |
| k | 20, 100, `k_max` |
| Methods | `none`, `cutmix`, frozen `simcutmix` |
| Seed | `(0,0)` |

`k_max` must be defined after the validation policy is fixed; do not assume all
500 training images per class remain available.

This screening matrix has 18 cells. If it runs reliably, repeat at least k=20
and `k_max` with seeds `(1,1)` and `(2,2)`, adding at most 24 jobs. If resources
allow, repeat all three k values for a 54-cell final Tiny ImageNet matrix.

## Phase 5 — Curriculum (Only After Must-Haves)

Treat curriculum as optional. Use the frozen SimCutMix configuration and one or
two representative k values. Compare:

- fixed best rank window;
- random partner baseline;
- stepwise easy-to-hard schedule;
- smooth easy-to-hard schedule, only if stepwise is promising.

A proposed stepwise schedule is ranks 21–40 early, 11–30 in the middle, and
1–10 late. Transition epochs must be declared before looking at test results.

## Similarity Method Statement for the Presentation

The current implementation computes similarity as follows:

1. Select only the committed k-shot training indices; validation and test
   samples are excluded.
2. Transform each image using the preprocessing associated with a frozen
   ImageNet-pretrained ResNet encoder.
3. Extract the feature vector before the final classification layer. With the
   current ResNet50 encoder this is the global-average-pooled 2048-dimensional
   representation.
4. L2-normalize the vectors.
5. Compute cosine similarity as the normalized dot product.
6. Build exact nearest-neighbor lists offline within the selected training
   subset, optionally filtering by label relation.
7. Keep the ranking fixed during classifier training.

This is latent-feature similarity, not similarity between final logits or
prediction probabilities. The presentation should state the encoder, feature
layer, dimensionality, filtering mode, rank window, sampling rule, and whether
the ranking is fixed.

## Required Repository Work Before Submission

### Must implement first

- Extend the Slurm runner with explicit subset and training seed arguments.
- Encode separate frozen profiles for SimMixUp and SimCutMix rather than using
  one shared guided configuration accidentally.
- Add k=200 and k=300 CIFAR-100 splits for seeds 0/1/2.
- Add a Slurm support job for embeddings and K40 neighbor payloads.
- Add a manifest with one row per intended job and statuses such as `missing`,
  `queued`, `running`, `complete`, `failed`, and `excluded-incompatible`.
- Record Git commit, Python/PyTorch/CUDA versions, GPU, and Slurm job ID in all
  new summaries.
- Add collision protection so a completed run is never silently overwritten.

### Guided support artifacts

For every guided `(dataset, k, subset_seed)` combination:

1. compute ResNet50 ImageNet embeddings once;
2. build a class-agnostic K40 neighbor payload;
3. build class-aware and different-label payloads only where required by the
   ablation;
4. inspect metadata, split hashes, shapes, and validation overlap;
5. retain the payload on cluster storage and record its checksum.

## Submission Order and Stop/Go Gates

| Order | Submission group | Gate before continuing |
|---:|---|---|
| 1 | New split generation and tests | Balance, determinism, no validation overlap |
| 2 | Seed-0 support payloads | Metadata and neighbor inspection pass |
| 3 | One smoke job per new dataset/path | GPU, outputs, and logs pass |
| 4 | CIFAR-100 Phase 1 seed-0 jobs | Manifest complete; no duplicate configs |
| 5 | Pair-selection ablations | Validation-only selection documented |
| 6 | CIFAR-100 seeds 1/2 support payloads | Checksums and inspection pass |
| 7 | CIFAR-100 confirmation jobs | Frozen settings recorded before submission |
| 8 | Tiny ImageNet implementation/tests | Dataset semantics and leakage checks pass |
| 9 | Tiny ImageNet screening/confirmation | Smoke and seed-0 validation pass |
| 10 | Curriculum jobs | All must-have evidence is complete |

Start with small job batches rather than submitting every cell simultaneously.
After each batch, inspect Slurm exit codes, `.out`/`.err` logs, summaries,
metrics, checkpoints, disk usage, and duplicate configuration IDs.

## Artifact and Git Policy

Commit to GitHub only after review:

- code, split JSON files, job manifests, and documentation;
- final compact `metrics.csv` and `summary.json` evidence;
- aggregate tables and presentation figures.

Keep on cluster storage:

- raw datasets and download caches;
- Conda environments;
- embedding and neighbor `.pt` payloads, with checksums recorded elsewhere;
- checkpoints;
- transient Slurm logs and smoke outputs.

Never select configurations from test results. Never combine historical scratch
records with current 224x224 runs in means or standard deviations.

## Expected Third-Presentation Outputs

1. CIFAR-100 accuracy-versus-k curves for scratch and pretrained models.
2. Mean ± standard deviation and individual points for three seeds.
3. A table comparing random, same-class, rank-window, and different-label
   partner selection.
4. A precise diagram or explanation of the latent-space similarity pipeline.
5. Tiny ImageNet results for the focused baseline/proposed comparison.
6. A clear negative-result statement where guided augmentation does not help.
7. A limitations slide covering n=3, paired-seed variance, current dataset
   scope, and the distinction between efficacy and pairing-only causality.

## Immediate Next-Chat Task

Start by auditing the 71 existing summaries against the Phase 1 matrix and
classifying every candidate cell as reusable, incompatible, or missing. Then
implement explicit seed arguments and new CIFAR-100 k=200/300 splits before
submitting any full training job.
