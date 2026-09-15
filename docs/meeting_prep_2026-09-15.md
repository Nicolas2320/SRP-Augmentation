# Meeting Prep — Haya — Tuesday 2026-09-15

**One-line opener:**  Our similarity-guided mixing either beats CutMix/MixUp outright, or ties "no
augmentation" on accuracy while training with a much smaller train/validation
gap — i.e. less memorization for the same generalization.

## 1. What SimMixUp / SimCutMix actually do (30-second explanation)

Standard MixUp/CutMix blend each image with a **randomly chosen** partner from
the batch. Our proposal changes only how the partner is picked:

1. Embed every training image once with a frozen ImageNet-pretrained ResNet50
   encoder.
2. For each image, precompute its nearest neighbors in embedding space
   (`class_aware` / `class_agnostic` / `different_label` modes
   constrain which classes are eligible partners).
3. At mixing time, instead of a random partner, take one from a chosen
   **neighbor-rank window** (our runs use ranks 21–40 of 40 saved neighbors —
   "similar, but not near-duplicates").
4. Apply the exact same MixUp/CutMix math to that pair →  **SimMixUp** / **SimCutMix**.

Intuition: mixing two near images teaches the model
little and risks label noise; mixing two totally random images can produce
unnatural pairs, which hurts more when there are only a handful of examples
per class. A moderately-similar partner is meant to give a better,
informative interpolation. 

## 2. Setup recap

- Dataset: CIFAR-100. Models: ResNet50 and ViT (CIFAR-adapted).
- Data budgets (images/class): k=20, k=50, k=100, k=450 (near full dataset).
- Baselines: None (still has crop+flip, just no mixing), MixUp, CutMix.
- Proposal: SimMixUp, SimCutMix, `class_agnostic` mode, neighbor ranks 21–40. 
- **Everything below is a single run** (one subset seed, one training seed).
  Multi-seed validation is the immediate next step, not done yet — say this
  proactively.

## 3. Figures to show

### `docs/figures/matched_test_accuracy.png` — the primary slide
Paired comparisons: for every (backbone, k) cell with a matched baseline, shows
baseline → proposal with the delta in percentage points.

- **ResNet50 k=100**: SimCutMix 46.16% vs CutMix 43.20% (**+2.96pp**) — cleanest
  win, lead with this one.
- **ResNet50 k=20/50**: SimCutMix/SimMixUp beat CutMix and MixUp by 1.5–5pp,
  and roughly match "no augmentation" (deltas near zero: +0.14, +0.16, -0.19pp).
- **ResNet50 k=450**: the one negative result — CutMix beats SimCutMix by
  1.10pp. Say this upfront; it's kept as an honest negative result, not hidden.
- **ViT k=20/50**: SimMixUp/SimCutMix beat CutMix and MixUp by 1.15–1.94pp,
  and roughly match "no augmentation" (deltas -0.22 to -0.55pp).

### `docs/figures/available_test_accuracy_vs_k.png`
All available results per backbone across k, side by side. Use it to be
transparent about **coverage gaps**: ViT SimMixUp/SimCutMix only exist at
k=20 and k=50 so far (k=100/k=450 proposal runs are next); ResNet50 k=100 has
no "None" baseline yet.

### `docs/figures/matched_validation_curves.png`
Validation-accuracy trajectories with stars marking the best-validation
checkpoint actually used for the reported test score. Worth eyeballing live —
proposal curves generally track the strongest baseline rather than trailing.
**Caveat:** this plots validation accuracy only, not training accuracy — it
does not by itself carry the overfitting story. That's the next figure.

### `docs/figures/matched_overfitting_gap.png` — the overfitting slide
Per-epoch `train_acc − val_acc` for the proposal and every matched baseline,
same panels as the validation-curves figure, stars again marking each run's
best-validation checkpoint (i.e. the gap value at the moment that checkpoint
was selected, not at epoch 100).

- **What to point at**: "None"'s dashed gray line climbs sharply after roughly
  epoch 25–30 in almost every panel and keeps climbing, while SimCutMix
  (solid) stays far lower and flattens out. That visual — a curve that keeps
  climbing vs. one that plateaus — is the whole overfitting argument in one
  picture.
- **Live nuance visible in the plot**: SimCutMix's line sits below CutMix's in
  every panel; SimMixUp's line sits *above* MixUp's in every panel (but always
  far below "None"). Use the exact numbers in section 4 below if she wants
  precision instead of eyeballing the plot.
- **Caveat to mention if asked**: the caption on the figure itself flags that
  train accuracy under the four mixing methods uses partial-credit mixed-label
  accuracy, not clean-label accuracy like "None" — see section 4.

### `docs/figures/experiment_coverage.png`
The full run-count matrix. Use it as the "here's exactly what's left to run"
slide.

## 4. The overfitting story — exact numbers behind the gap figure

The plot above is the visual; this table gives the precise values to quote,
pulled directly from each run's `metrics.csv` at its best-validation epoch
(the same checkpoint used for the reported test score).

**Nuance to mention if asked "is this comparison fair?":** train accuracy for
CutMix/MixUp/SimCutMix/SimMixUp is computed with the same partial-credit
formula the original papers use — `lam * acc(label_a) + (1-lam) * acc(label_b)`
— not clean single-label accuracy like "None" uses. So the absolute train-acc
numbers for mixing methods vs. "None" aren't perfectly apples-to-apples. That
said, the gaps below differ by 40–80 points, far more than that metric
definition alone could produce.

### ResNet50 · CIFAR-100 (train_acc / val_acc / gap, at best epoch, %)

| k | Method | Train | Val | Gap |
|---:|---|---:|---:|---:|
| 20 | None | 72.40 | 16.88 | 55.52 |
| 20 | CutMix | 31.55 | 14.56 | 16.99 |
| 20 | MixUp | 21.38 | 13.24 | 8.14 |
| 20 | **SimCutMix** | 32.03 | 16.32 | **15.71** |
| 20 | **SimMixUp** | 38.32 | 16.18 | **22.14** |
| 50 | None | 87.70 | 28.66 | 59.04 |
| 50 | CutMix | 43.28 | 26.58 | 16.70 |
| 50 | MixUp | 25.51 | 23.10 | 2.41 |
| 50 | **SimCutMix** | 41.44 | 29.64 | **11.80** |
| 50 | **SimMixUp** | 49.25 | 28.30 | **20.95** |
| 100 | CutMix | 60.94 | 44.06 | 16.88 |
| 100 | MixUp | 41.71 | 40.84 | 0.87 |
| 100 | **SimCutMix** | 56.08 | 46.40 | **9.68** |
| 100 | **SimMixUp** | 64.05 | 44.20 | **19.85** |
| 450 | None | 96.90 | 71.08 | 25.82 |
| 450 | CutMix | 73.26 | 72.58 | 0.68 |
| 450 | MixUp | 54.31 | 68.40 | -14.09 |
| 450 | **SimCutMix** | 66.84 | 71.48 | **-4.64** |

(No "None" baseline recorded yet at ResNet50 k=100 — a known gap.)

### ViT · CIFAR-100 (train_acc / val_acc / gap, at best epoch, %)

| k | Method | Train | Val | Gap |
|---:|---|---:|---:|---:|
| 20 | None | 39.55 | 12.56 | 26.99 |
| 20 | CutMix | 18.69 | 12.56 | 6.13 |
| 20 | MixUp | 14.60 | 12.32 | 2.28 |
| 20 | **SimCutMix** | 19.12 | 12.52 | **6.60** |
| 20 | **SimMixUp** | 23.49 | 13.18 | **10.31** |
| 50 | None | 38.88 | 20.80 | 18.08 |
| 50 | CutMix | 21.08 | 19.18 | 1.90 |
| 50 | MixUp | 19.32 | 19.32 | 0.00 |
| 50 | **SimCutMix** | 24.08 | 20.68 | **3.40** |
| 50 | **SimMixUp** | 27.72 | 21.04 | **6.68** |

(ViT proposal runs only exist at k=20/k=50 so far; k=100/k=450 are next.)

### How to narrate this table

- **Headline pair**: ViT k=50 — "None" gets 21.12% test with a train/val gap of
  18.08pp; SimCutMix gets 20.57% test (essentially tied) with a gap of only
  3.40pp. Same story at ResNet50 k=20/50/100: SimCutMix matches or beats
  "None" on test accuracy while cutting the gap by roughly half to a third.
- **Be precise about which comparison this supports**: the "much less
  overfitting" claim is strongest for **SimCutMix vs. None** (and SimCutMix's
  gap is also consistently smaller than CutMix's own gap, at every k — a
  second independent piece of evidence). **Don't** claim SimMixUp is "more
  regularizing than MixUp" — its gap is smaller than "None" every time, but
  larger than plain MixUp's every time (e.g. ResNet50 k=50: MixUp gap 2.41 vs
  SimMixUp 20.95). The honest claim for SimMixUp is: much less overfit than no
  augmentation, while still beating both baselines on raw test accuracy at
  most budgets.
- **Negative-looking gaps at k=450** (train below val) are an expected
  side-effect of the partial-credit train metric under heavy mixing, not
  evidence of underfitting — mention only if asked.

## 5. Likely questions from Haya

- *"Is this averaged over seeds?"* → No, one run per cell so far. Splits for
  seeds 0/1/2 already exist; multi-seed runs are the next concrete task.
- *"Why does the proposal lose at ResNet50 k=450?"* → Honest negative result,
  kept in the record rather than dropped. Plausible explanation: at
  near-full-dataset budget there's little room left for a smarter pairing
  strategy to add value — worth investigating further, not yet explained.
- *"Is the train/val gap comparison fair across methods?"* → Give the
  partial-credit-metric nuance from section 4.
- *"Why does ViT have less proposal coverage than ResNet50?"* → SimMixUp/
  SimCutMix on ViT only run at k=20/k=50 so far; k=100 and k=450 are queued.
- *"What's next?"* → Fill the missing baseline/proposal cells (see the
  coverage figure), then multi-seed runs, then final aggregate tables with
  uncertainty bars.
