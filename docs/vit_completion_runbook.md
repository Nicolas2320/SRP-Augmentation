# Single-Seed ViT Completion Runbook

Status: frozen for execution. This extends the historical CIFAR-100 scratch
ViT cohort without creating a new full architecture study.

## Existing Coverage Audit

Twenty-one existing ViT runs use 100 epochs. The only 50-epoch record is
Crop+Flip at k=450. The existing k=450 MixUp also uses alpha 0.1, while the
k=20/50/100 MixUp records use alpha 1. Both cells are therefore retrained.

## New Final Matrix

| k | Methods to run | Count |
|---:|---|---:|
| 300 | Crop+Flip, MixUp, CutMix, AugMix, SimMixUp, SimCutMix | 6 |
| 450 | Crop+Flip, MixUp, SimMixUp, SimCutMix | 4 |
| **Total** |  | **10** |

All jobs use `(subset_seed, train_seed)=(0,0)`, batch size 64, and 100
epochs. Existing k=450 CutMix and AugMix are reusable. AugMix retains its
historical learning-rate schedule and remains an auxiliary, non-canonical
baseline because the local implementation omits JSD consistency.

Guided methods use frozen ImageNet-ResNet50 class-agnostic K40 neighbors,
ranks 21-40, uniform sampling, alpha 1, mixing probability 1, and no warmup.

## Preflight

```bash
for k in 300 450; do
  split="data/splits/cifar100/k${k}_seed0.json"
  neighbors="results/experiments/shared/neighbors/cifar100/k${k}_seed0/neighbors_class_agnostic_K40.pt"
  [[ -f "$split" ]] && echo "READY $split" || echo "MISSING $split"
  [[ -f "$neighbors" ]] && echo "READY $neighbors" || echo "MISSING $neighbors"
done
```

## Smoke Gate

The three distinct execution paths are sufficient for the smoke gate:

```bash
for method in none augmix simcutmix; do
  sbatch scripts/run_vit_completion.slurm 300 "$method" smoke 0 0
done
```

Require all three jobs to complete. Confirm the SimCutMix log reports
class-agnostic ranks 21-40.

## Submit Final Jobs

```bash
VIT_JOBS="logs/vit-completion-jobs.txt"
: > "$VIT_JOBS"

for method in none mixup cutmix augmix simmixup simcutmix; do
  jid=$(sbatch --parsable \
    scripts/run_vit_completion.slurm 300 "$method" final 0 0)
  printf 'k300_%s=%s\n' "$method" "$jid" | tee -a "$VIT_JOBS"
done

for method in none mixup simmixup simcutmix; do
  jid=$(sbatch --parsable \
    scripts/run_vit_completion.slurm 450 "$method" final 0 0)
  printf 'k450_%s=%s\n' "$method" "$jid" | tee -a "$VIT_JOBS"
done
```

## Monitor

```bash
IDS=$(cut -d= -f2 logs/vit-completion-jobs.txt | paste -sd, -)
squeue -j "$IDS" -o "%.18i %.24j %.10T %.12M %.12l %R"
sacct -j "$IDS" -X --format=JobID,JobName,State,Elapsed,ExitCode,NodeList
```

Gate: all ten top-level jobs must be `COMPLETED` with exit code `0:0`.
Commit metrics and summaries only; do not commit checkpoints or Slurm logs.

## Reporting Boundary

Report this as exploratory, single-seed architecture-transfer evidence. The
cleanest comparisons are SimMixUp versus MixUp and SimCutMix versus CutMix.
Do not report uncertainty or statistical significance.
