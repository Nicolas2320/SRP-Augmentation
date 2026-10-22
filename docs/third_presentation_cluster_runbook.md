# Third Presentation Cluster Runbook

This runbook is the operational companion to
[`third_presentation_job_matrix.md`](third_presentation_job_matrix.md). Run all
commands from the repository root on the ISMLL cluster. The STUD partition has
ten shared RTX 2070 GPUs; submit small batches and expect other users to share
the node.

## 1. Preflight on the Login Node

```bash
cd "$HOME/SRP-Augmentation"
conda activate srp-augmentation
git status --short --branch
git pull --ff-only
mkdir -p logs
python src/data/audit_splits.py
python src/experiments/build_third_presentation_manifest.py
```

Review `experiments/manifests/third_presentation_phase1.csv`. Do not submit a
row already classified as `reusable` or already marked `queued`, `running`, or
`complete`.

Submit the CPU test suite after every code change:

```bash
sbatch \
  --partition=CPU \
  --time=00:15:00 \
  --cpus-per-task=2 \
  --mem=4G \
  --job-name=srp-tests \
  --chdir="$PWD" \
  --output=logs/srp-tests-%j.log \
  --wrap="$CONDA_PREFIX/bin/python -m unittest discover -s tests -v"
```

## 2. Seed-0 Guided Support

Generate class-agnostic K40 payloads for every Phase-1 k value:

```bash
for k in 20 50 100 200 300 450; do
  sbatch scripts/build_guided_support.slurm cifar100 "$k" 0 class_agnostic
done
```

The support job computes embeddings when absent, refuses to overwrite an
existing neighbor payload, validates split metadata and leakage, and writes a
checksum file. Check every exit code and log before guided training.

The pairing ablation additionally needs class-aware and different-label
payloads at scratch k=20/100/300 and pretrained k=100:

```bash
for k in 20 100 300; do
  sbatch scripts/build_guided_support.slurm cifar100 "$k" 0 class_aware
  sbatch scripts/build_guided_support.slurm cifar100 "$k" 0 different_label
done
```

The k=100 files are shared by scratch and pretrained classifiers because the
frozen pairing encoder and selected data subset are identical.

## 3. Smoke Tests

```bash
sbatch scripts/run_resnet_comparison.slurm pretrained 200 none smoke 0 0
sbatch scripts/run_resnet_comparison.slurm scratch 200 simcutmix smoke 0 0
sbatch scripts/run_pair_ablation.slurm scratch 100 most_similar smoke 0 0
```

Smoke outputs stay under `results/smoke/` and are never presentation evidence.

## 4. Phase-1 Validation Screening

The runner syntax is:

```text
sbatch scripts/run_resnet_comparison.slurm INITIALIZATION K METHOD screen SUBSET_SEED TRAIN_SEED
```

Submit only missing or incompatible rows from the manifest, preferably three
to five jobs at a time. Screening mode always passes `--skip-test`.

Example batch:

```bash
sbatch scripts/run_resnet_comparison.slurm scratch 20 none screen 0 0
sbatch scripts/run_resnet_comparison.slurm scratch 20 mixup screen 0 0
sbatch scripts/run_resnet_comparison.slurm scratch 20 cutmix screen 0 0
```

After each batch:

```bash
squeue -u "$USER" -p STUD
sacct -j JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed
cat logs/srp-resnet-JOB_ID.err
```

Update the manifest status and Slurm job ID. Rebuilding the manifest preserves
operational statuses for non-reusable cells.

## 5. Pair-Selection Ablation

Compare six strategies at scratch k=20/100/300 and pretrained k=100:

```text
random
same_class
most_similar
middle_similarity
less_similar
different_label
```

For example:

```bash
sbatch scripts/run_pair_ablation.slurm scratch 100 same_class screen 0 0
sbatch scripts/run_pair_ablation.slurm scratch 100 most_similar screen 0 0
sbatch scripts/run_pair_ablation.slurm scratch 100 middle_similarity screen 0 0
sbatch scripts/run_pair_ablation.slurm scratch 100 less_similar screen 0 0
sbatch scripts/run_pair_ablation.slurm scratch 100 different_label screen 0 0
```

Reuse the Phase-1 CutMix cell for the random condition; submit the five guided
strategies through the ablation runner. All six use CutMix alpha 1 and mixing
probability 0.5. Select the final guided strategy from validation evidence only
and record the frozen settings before the confirmation jobs.

The Phase-2 decision was frozen on 2026-10-20 from the validation results in
commit `205790e`:

```text
augmentation: simcutmix
pairing encoder: frozen ImageNet ResNet50
guided mode: class_agnostic
neighbor rank window: 21-40
neighbor rank start: 21
neighbor count: 20
pair sampling: uniform
CutMix alpha: 1.0
confirmation mix probability: 1.0
mix warmup: 0 epochs
```

The 0.5 probability used in the ablation isolates partner selection. The
confirmation profile remains the predeclared Phase-1 profile at probability
1.0. See `docs/third_presentation_job_matrix.md` for the validation table,
selection rationale, and disclosure about the historical pretrained k=100
random condition.

## 6. Confirmation Seeds

For k below 450, use `(subset_seed, train_seed)` pairs `(0,0)`, `(1,1)`, and
`(2,2)`. Generate the class-agnostic support payload for each guided subset
seed before SimCutMix training.

At k=450, the selected subset is the entire post-validation pool. Use subset
seed 0 with training seeds 0, 1, and 2, and disclose that these runs measure
optimization variability only.

Use `final` only after the method is frozen:

```bash
sbatch scripts/run_resnet_comparison.slurm scratch 100 none final 1 1
sbatch scripts/run_resnet_comparison.slurm scratch 100 cutmix final 1 1
sbatch scripts/run_resnet_comparison.slurm scratch 100 simcutmix final 1 1
```

To evaluate the test set for an existing seed-0 screening checkpoint without
retraining it, use `evaluate` with the identical configuration:

```bash
sbatch scripts/run_resnet_comparison.slurm scratch 100 simcutmix evaluate 0 0
```

Do not use `evaluate` until the configuration is frozen. Tiny ImageNet and
curriculum work begin only after the CIFAR-100 confirmation gate passes.
