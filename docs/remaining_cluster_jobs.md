# Remaining Cluster Jobs

Last updated: 2026-10-31.

Run these stages in order. Every submission writes its job IDs below `logs/`.
Do not resubmit a stage whose summaries have already been reviewed and merged.

## Stage A — Evaluate the Frozen Curriculum Candidate

These jobs load the existing scratch k=20 best-validation checkpoints and
evaluate them once on the test set. They do not retrain.

```bash
CURRICULUM_JOBS="logs/curriculum-k20-evaluation-jobs.txt"
: > "$CURRICULUM_JOBS"

for seed in 0 1 2; do
  jid=$(sbatch --parsable \
    scripts/run_easy_to_hard_curriculum.slurm \
    scratch 20 evaluate "$seed" "$seed")
  printf 'curriculum_k20_seed%s=%s\n' "$seed" "$jid" | tee -a "$CURRICULUM_JOBS"
done
```

Gate: all three top-level jobs must be `COMPLETED`, and every updated summary
must retain `best_val_acc` while adding `test_evaluated=true` and a test
accuracy.

## Stage B — Complete Tiny ImageNet k=100

The seed-0 runs already exist as validation-only checkpoints. Splits for seeds
1 and 2 are committed. First ensure that the two class-agnostic K40 support
payloads exist on the cluster.

```bash
for seed in 1 2; do
  support="results/experiments/shared/neighbors/tinyimagenet/k100_seed${seed}/neighbors_class_agnostic_K40.pt"
  if [[ -f "$support" ]]; then
    echo "READY $support"
  else
    sbatch scripts/build_guided_support.slurm tinyimagenet 100 "$seed" class_agnostic
  fi
done
```

Wait for any submitted support jobs to complete and pass validation. Then
submit the six seed-0 checkpoint evaluations and twelve seed-1/2 final runs:

```bash
TINY_K100_JOBS="logs/tiny-k100-confirmation-jobs.txt"
: > "$TINY_K100_JOBS"

for initialization in scratch pretrained; do
  for method in none cutmix simcutmix; do
    jid=$(sbatch --parsable \
      scripts/run_tiny_imagenet.slurm \
      "$initialization" 100 "$method" evaluate 0 0)
    printf 'evaluate_%s_%s_seed0=%s\n' "$initialization" "$method" "$jid" \
      | tee -a "$TINY_K100_JOBS"
  done

  for seed in 1 2; do
    for method in none cutmix simcutmix; do
      jid=$(sbatch --parsable \
        scripts/run_tiny_imagenet.slurm \
        "$initialization" 100 "$method" final "$seed" "$seed")
      printf 'final_%s_%s_seed%s=%s\n' "$initialization" "$method" "$seed" "$jid" \
        | tee -a "$TINY_K100_JOBS"
    done
  done
done
```

Gate: 18 top-level jobs must be `COMPLETED`. The final evidence must contain
three paired seeds for each of the six initialization/method cells.

## Stage C — Promote Standard MixUp to the CIFAR-100 Final Matrix

The final recipe is already frozen. Seed-0 screening checkpoints exist for
scratch at every k and for pretrained k=200/300/450. Compatible, already-tested
pretrained seed-0 results at k=20/50/100 are reused from `comparison_v1`.

First evaluate the nine compatible seed-0 screening checkpoints:

```bash
MIXUP_JOBS="logs/cifar100-mixup-confirmation-jobs.txt"
: > "$MIXUP_JOBS"

for k in 20 50 100 200 300 450; do
  jid=$(sbatch --parsable \
    scripts/run_resnet_comparison.slurm scratch "$k" mixup evaluate 0 0)
  printf 'evaluate_scratch_k%s_seed0=%s\n' "$k" "$jid" | tee -a "$MIXUP_JOBS"
done

for k in 200 300 450; do
  jid=$(sbatch --parsable \
    scripts/run_resnet_comparison.slurm pretrained "$k" mixup evaluate 0 0)
  printf 'evaluate_pretrained_k%s_seed0=%s\n' "$k" "$jid" | tee -a "$MIXUP_JOBS"
done
```

Then submit seeds 1 and 2 for both initializations and all six k values:

```bash
for initialization in scratch pretrained; do
  for k in 20 50 100 200 300 450; do
    for train_seed in 1 2; do
      subset_seed="$train_seed"
      [[ "$k" == "450" ]] && subset_seed=0

      jid=$(sbatch --parsable \
        scripts/run_resnet_comparison.slurm \
        "$initialization" "$k" mixup final "$subset_seed" "$train_seed")
      printf 'final_%s_k%s_s%s_t%s=%s\n' \
        "$initialization" "$k" "$subset_seed" "$train_seed" "$jid" \
        | tee -a "$MIXUP_JOBS"
    done
  done
done
```

Gate: 33 submitted jobs must be `COMPLETED`. Together with the three compatible
pretrained seed-0 k=20/50/100 records, this yields three tested seeds for all
12 initialization/k cells.

## Monitoring

For any job file above:

```bash
JOB_FILE="logs/curriculum-k20-evaluation-jobs.txt"  # change as needed
IDS=$(cut -d= -f2 "$JOB_FILE" | paste -sd, -)

squeue -j "$IDS" -o "%.18i %.24j %.10T %.10M %.6D %R"
sacct -j "$IDS" --format=JobID,JobName,State,Elapsed,ExitCode,NodeList
```

Commit and push only metrics and summaries. Do not commit checkpoints,
embeddings, neighbor payloads, raw data, or Slurm logs.
