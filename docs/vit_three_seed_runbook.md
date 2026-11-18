# Three-Seed ViT Completion Runbook

Status: frozen for execution. Seed 0 is complete. This runbook adds paired
seeds 1 and 2 across the full exploratory ViT curve.

## Frozen Matrix

| Dimension | Values |
|---|---|
| Dataset | CIFAR-100 |
| Classifier | Scratch CIFAR ViT, 32×32 inputs, 4×4 patches |
| k | 20, 50, 100, 300, 450 |
| Methods | Crop+Flip, MixUp, CutMix, AugMix, SimMixUp, SimCutMix |
| New training seeds | 1, 2 |
| New jobs | `5 × 6 × 2 = 60` |

For k<450, use paired `(subset_seed, train_seed)` values `(1,1)` and `(2,2)`.
For k=450, the full post-validation pool has only subset seed 0, so use
`(0,1)` and `(0,2)`.

All runs use 100 epochs and batch size 64. Crop+Flip, MixUp, CutMix, SimMixUp,
and SimCutMix use SGD with learning rate 0.01, milestones 30/55/75, and gamma
0.1. AugMix retains the historical cohort schedule: learning rate 0.1,
milestones 30/60/80, and gamma 0.2. The local AugMix implementation is not
paper-faithful because it omits JSD consistency; keep it auxiliary.

Guided methods use frozen ImageNet-ResNet50 class-agnostic K40 neighbors,
ranks 21-40, uniform sampling, alpha 1, mixing probability 1, and no warmup.

## Cluster Preflight

```bash
set -euo pipefail

for k in 20 50 100 300; do
  for seed in 1 2; do
    split="data/splits/cifar100/k${k}_seed${seed}.json"
    neighbors="results/experiments/shared/neighbors/cifar100/k${k}_seed${seed}/neighbors_class_agnostic_K40.pt"
    [[ -f "$split" ]] && echo "READY $split" || { echo "MISSING $split"; exit 1; }
    [[ -f "$neighbors" ]] && echo "READY $neighbors" || { echo "MISSING $neighbors"; exit 1; }
  done
done

split="data/splits/cifar100/k450_seed0.json"
neighbors="results/experiments/shared/neighbors/cifar100/k450_seed0/neighbors_class_agnostic_K40.pt"
[[ -f "$split" ]] && echo "READY $split" || { echo "MISSING $split"; exit 1; }
[[ -f "$neighbors" ]] && echo "READY $neighbors" || { echo "MISSING $neighbors"; exit 1; }
```

Expected: 18 `READY` messages and no `MISSING` messages.

## Submit 60 Jobs

```bash
VIT_3SEED_JOBS="logs/vit-three-seed-jobs.txt"
: > "$VIT_3SEED_JOBS"

methods=(none mixup cutmix augmix simmixup simcutmix)

for k in 20 50 100 300; do
  for seed in 1 2; do
    for method in "${methods[@]}"; do
      jid=$(sbatch --parsable \
        scripts/run_vit_completion.slurm \
        "$k" "$method" final "$seed" "$seed")

      printf 'k%s_%s_s%s_t%s=%s\n' \
        "$k" "$method" "$seed" "$seed" "$jid" \
        | tee -a "$VIT_3SEED_JOBS"
    done
  done
done

for seed in 1 2; do
  for method in "${methods[@]}"; do
    jid=$(sbatch --parsable \
      scripts/run_vit_completion.slurm \
      450 "$method" final 0 "$seed")

    printf 'k450_%s_s0_t%s=%s\n' \
      "$method" "$seed" "$jid" \
      | tee -a "$VIT_3SEED_JOBS"
  done
done

echo "Submitted $(wc -l < "$VIT_3SEED_JOBS") jobs"
```

The final line must report `Submitted 60 jobs`.

## Monitor

```bash
VIT_3SEED_JOBS="logs/vit-three-seed-jobs.txt"
VIT_IDS=$(cut -d= -f2 "$VIT_3SEED_JOBS" | paste -sd, -)

squeue -j "$VIT_IDS" \
  -o "%.18i %.24j %.10T %.12M %.12l %R"

sacct -j "$VIT_IDS" -X \
  --format=JobID,JobName,State,Elapsed,ExitCode,NodeList
```

Gate: all 60 top-level jobs must be `COMPLETED` with exit code `0:0`.

## Artifact Policy

Commit and push only `metrics.csv` and `summary.json`. Do not commit
checkpoints, neighbor payloads, raw data, or Slurm logs.

## Reporting

After completion, report mean ± sample SD across seeds 0/1/2 for each of the
30 k/method cells. The primary paired comparisons are SimMixUp−MixUp and
SimCutMix−CutMix. The older seed-0 records at k=20/50/100 remain historical
provenance and must be disclosed even though their frozen recipes match the
new repeats.
