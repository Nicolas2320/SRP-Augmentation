# Project Status

Last verified: 2026-09-28.

## Current Stage

The implementation and first comparison matrix are substantially complete.
The immediate scientific task is no longer basic pipeline construction; it is
to choose defensible final configurations using validation evidence and repeat
them across multiple subset/training seeds.

The current tracked evidence is exploratory: every result uses subset seed 0
and training seed 0. No single-run difference should be described as
statistically significant.

## Implemented Scope

- CIFAR-10 and CIFAR-100 fixed-validation, k-shot splits.
- Standard 224x224 ResNet50 with scratch or ImageNet initialization.
- Scratch-trained 32x32 CIFAR ViT.
- Raw, crop+flip, MixUp, CutMix, and AugMix baselines.
- SimMixUp and SimCutMix with class-aware, class-agnostic, or different-label
  neighbors.
- Rank windows, uniform/weighted sampling, warm-up, anchor gating, and dynamic
  neighbor pools.
- Validation-only runs, best-checkpoint test evaluation, recipe-aware plots,
  artifact audits, and 73 passing tests.

## Evidence Inventory

The active evidence root is `results/comparison_v1/`.

| Group | Models | Budgets | Summary/metrics pairs |
|---|---|---|---:|
| `pretrained` | ResNet50 | k=20/50/100 | 21 |
| `scratch` | ResNet50 | k=20/50/100/450 | 28 |
| `scratch` | ViT | k=20/50/100/450 | 22 |
| **Total** |  |  | **71** |

The pretrained ResNet50 matrix is complete for seven methods at k=20/50/100:
raw, none, MixUp, CutMix, AugMix, SimMixUp, and SimCutMix. It has no k=450
runs. The scratch ResNet50 matrix has seven methods at all four budgets. ViT
has six methods at k=20/50/100 and four baselines at k=450; it has no tracked
raw runs and no guided k=450 runs.

There are 43 referenced local best checkpoints for 71 run records. All 71
summaries have matching metrics. Four additional checkpoint files are in
incomplete run folders and require a retention decision.

## Current Pretrained ResNet50 Results

Test accuracy on CIFAR-100, one run per cell:

| Method | k=20 | k=50 | k=100 |
|---|---:|---:|---:|
| Raw | 65.33% | 73.26% | 77.33% |
| Crop + flip (`none`) | 66.73% | 72.90% | 76.88% |
| MixUp | 67.66% | 73.47% | 77.72% |
| CutMix | **68.88%** | **75.75%** | 79.89% |
| AugMix | 62.37% | 71.02% | 75.43% |
| SimMixUp | 44.75% | 74.39% | **80.13%** |
| SimCutMix | 67.90% | 75.13% | 78.53% |

Interpretation:

- At k=20, CutMix is strongest. SimCutMix is 0.98 percentage points lower;
  the tracked SimMixUp configuration is substantially worse.
- At k=50, CutMix is strongest. SimCutMix is 0.62 points lower, while SimMixUp
  is 0.92 points above MixUp but 1.36 points below CutMix.
- At k=100, SimMixUp is strongest at 80.13%, 0.24 points above CutMix and 2.41
  points above MixUp. SimCutMix is 1.36 points below CutMix.

These results do not establish that guided pairing is generally superior. The
only best-in-column guided result is SimMixUp at k=100, and its advantage over
CutMix is small relative to the missing seed-level uncertainty.

Configuration caveat: the tracked pretrained SimMixUp runs use
class-agnostic ranks 1-10, whereas pretrained SimCutMix uses ranks 21-40. This
means they are method-specific selected configurations, not a shared
pairing-only ablation.

## Scratch-Group Evidence

The scratch figure group contains useful exploratory results for ResNet50 and
ViT, including imported historical cohorts. It must not be presented as one
uniform modern training recipe.

- Most scratch ResNet50 and all ViT records are tagged
  `historical_from_scratch` and reflect the architecture/recipe in use when
  they were produced.
- Some current scratch ResNet50 records use the standard 224x224 architecture.
- The plotting code includes architecture/recipe cohort in matching and does
  not pool incompatible records as repeated seeds.

The older headline result, scratch ResNet50 SimCutMix 46.16% versus CutMix
43.20% at k=100, remains a valid record within its matched historical cohort.
It is not the current pretrained result and should not be used to describe the
ImageNet-initialized classifier.

## Reproducibility and Artifact Gaps

- Multi-seed repeats and confidence intervals are absent.
- Pretrained k=450 is absent.
- Guided neighbor `.pt` payloads referenced by summaries are absent locally.
- Twenty-eight tracked runs lack their referenced best checkpoint locally.
- Four local checkpoints belong to incomplete run folders.
- Historical summaries do not capture Git commit, Python, PyTorch/CUDA, GPU,
  or a fully resolved environment.
- `requirements.txt` is bounded but not a lock file.
- Baseline CutMix applies with probability 0.5, while SimCutMix applies with
  probability 1.0 in the fixed protocol; their difference is not partner
  selection alone.
- The local AugMix implementation uses ordinary cross-entropy and omits the
  AugMix paper's JSD consistency loss.

## Current Documentation and Evidence Sources

| Source | Role |
|---|---|
| Root `README.md` | Setup and common entry points |
| `docs/architecture.md` | System and artifact flow |
| `docs/reproducibility.md` | Reproduction requirements and limitations |
| `docs/resnet_comparison_v1.md` | Fixed comparison recipe and deviations |
| `docs/project_status.md` | Date-sensitive status and interpretation |
| `docs/figures/<initialization>/runs.csv` | Generated index of plotted summaries |
| `results/comparison_v1/**/summary.json` | Authoritative per-run configuration/result |

The March proposal and dated meeting brief are historical records, not current
sources of truth.

## Next Research Tasks

1. Decide the validation-based rule for choosing one SimMixUp and one
   SimCutMix configuration; do not choose solely from test accuracy.
2. Regenerate and durably store the required embedding/neighbor payloads.
3. Run selected baseline and guided configurations for subset seeds 1/2 and
   multiple training seeds.
4. Decide whether pretrained k=450 is necessary for the final research claim.
5. Report mean, standard deviation, run count, and individual seeds.
6. Add pairing-only ablations with equal mixing probability if attributing an
   effect specifically to partner selection.

## Next Repository Tasks

1. Review the four checkpoints in incomplete run folders.
2. Decide which best checkpoints and guided support payloads need durable
   external storage.
3. Capture Git/environment metadata automatically in future summaries.
4. Add a comparison-specific manifest command or make the generic manifest
   builder accept explicit input/output roots.

## Bottom Line

The code and single-seed comparison evidence are in place. For the current
pretrained ResNet50 study, CutMix leads at k=20 and k=50, while SimMixUp leads
at k=100 by 0.24 percentage points. That pattern is interesting but not yet a
final claim. Multi-seed experiments, support-artifact recovery, and stricter
pairing-only controls are the critical next steps.
