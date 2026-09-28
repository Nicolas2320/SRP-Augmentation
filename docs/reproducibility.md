# Reproducibility

Last verified: 2026-09-28.

Reproduction depends on four layers: committed data selections, controlled
randomness, an adequately recorded software/hardware environment, and the
availability of local support artifacts. The first two are implemented; the
last two remain incomplete for historical runs.

## Data Selection

The JSON files under `data/splits/` are authoritative. They store original
CIFAR training indices, not image data.

- Validation seed is fixed at 123.
- CIFAR-10 reserves 500 validation images per class.
- CIFAR-100 reserves 50 validation images per class.
- k=5/10/20/50/100 subsets exist for seeds 0/1/2.
- The maximum balanced post-validation subset is generated only for seed 0
  (k=450 for CIFAR-100).

Existing experiments should reuse committed splits. Run
`src/data/make_splits.py` only when deliberately rebuilding or validating the
collection.

## Randomness

`subset_seed` selects labeled examples. `train_seed` controls optimization,
shuffling, augmentation RNGs, workers, transforms, PyTorch CPU/CUDA RNGs, and
deterministic cuDNN settings where available.

Deterministic seeding does not guarantee bit-for-bit equality across PyTorch,
CUDA, drivers, GPUs, or operating systems. Scientific conclusions require
multiple independent runs; the current tracked evidence uses only subset seed
0 and training seed 0.

## Environment Setup

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements select PyTorch 2.7.0 and TorchVision 0.22.0 CUDA 12.8 wheels,
plus bounded versions of NumPy, Matplotlib, pandas, Pillow, and Jupyter. This
is not a lock file: Python, drivers, platform libraries, and transitive
dependencies are not pinned.

For every new reportable run, record:

```powershell
python --version
python -c "import torch, torchvision, numpy; print(torch.__version__); print(torchvision.__version__); print(numpy.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
git rev-parse HEAD
python -m pip freeze
```

Store this information with the experiment notes until summary-level
environment capture is implemented.

## Preflight Verification

Run the test suite:

```powershell
python -m unittest discover -s tests -v
```

As of 2026-09-28, all 73 tests pass. They cover split logic, indexed datasets,
neighbor validation, guided mixing, recipe/config IDs, artifact auditing, and
plot grouping. They do not download CIFAR or run full training.

Check the required split files, for example:

```powershell
Test-Path data\splits\cifar100\fixed_validation_split.json
Test-Path data\splits\cifar100\k20_seed0.json
```

Raw data downloads to `data/raw/` when missing.

## Fixed ResNet50 Protocol

Use the runner to avoid transcription errors. Without `-Execute`, it performs
a dry run and prints the exact command.

```powershell
.\scripts\run_resnet_comparison.ps1 -Initialization pretrained -K 20 -Method cutmix
.\scripts\run_resnet_comparison.ps1 -Initialization pretrained -K 20 -Method cutmix -Execute
```

The runner uses CIFAR-100, seeds 0/0, batch size 32, SGD with momentum 0.9,
Nesterov, and weight decay 0.0005. For k=20/50/100 it uses 100 epochs, LR
0.01, milestones 30/55/75, and gamma 0.1. For k=450 it uses 50 epochs, LR 0.1,
milestones 15/30/40, and gamma 0.2.

Read [ResNet50 comparison v1](resnet_comparison_v1.md) for method parameters,
historical deviations, and the current run matrix.

## Custom Standard Run

Specify all material hyperparameters. Use `--skip-test` during configuration
selection to avoid repeated test-set feedback:

```powershell
python -u src\train.py --dataset cifar100 --model resnet50 --pretrained --k 20 --subset-seed 0 --train-seed 0 --augmentation cutmix --cutmix-alpha 1 --cutmix-prob 0.5 --epochs 100 --batch-size 32 --optimizer sgd --lr 0.01 --momentum 0.9 --nesterov --weight-decay 0.0005 --lr-milestones 30 55 75 --lr-gamma 0.1 --num-workers 2 --skip-test
```

Remove `--skip-test` only after fixing the configuration. The test evaluation
then uses the best-validation checkpoint.

Initialization matters:

- `--no-pretrained` (default): random ResNet50 weights.
- `--pretrained`: ImageNet weights, with all parameters fine-tuned.

Both use the same standard 224x224 architecture and ImageNet normalization.

## Guided Run

All guided stages must agree on dataset, k, subset seed, encoder, and neighbor
mode.

### 1. Compute embeddings

```powershell
python -u src\proposal\compute_embeddings.py --dataset cifar100 --k 20 --subset-seed 0 --encoder resnet50_imagenet --batch-size 64 --num-workers 2 --device auto
```

### 2. Build and inspect neighbors

```powershell
python -u src\proposal\build_neighbors.py --dataset cifar100 --k 20 --subset-seed 0 --encoder resnet50_imagenet --mode class_agnostic --max-neighbors 40 --query-batch-size 512 --device auto
python -u src\proposal\inspect_neighbors.py --dataset cifar100 --k 20 --subset-seed 0 --encoder resnet50_imagenet --mode class_agnostic --max-neighbors 40
```

Neighbor search is exact and blockwise. Lowering `--query-batch-size` reduces
memory use without changing the result.

### 3. Train

```powershell
python -u src\train.py --dataset cifar100 --model resnet50 --pretrained --k 20 --subset-seed 0 --train-seed 0 --augmentation simcutmix --cutmix-alpha 1 --epochs 100 --batch-size 32 --optimizer sgd --lr 0.01 --momentum 0.9 --nesterov --weight-decay 0.0005 --lr-milestones 30 55 75 --lr-gamma 0.1 --num-workers 2 --output-root results\comparison_v1\pretrained --neighbor-path results\experiments\shared\neighbors\cifar100\k20_seed0\neighbors_class_agnostic_K40.pt --guided-mode class_agnostic --neighbor-k 20 --neighbor-rank-start 21 --pair-sampling uniform --mix-prob 1 --mix-warmup-epochs 0
```

The example selects ranks 21-40 from a saved K40 set. Anchor-gated runs first
create a score payload with `src/proposal/score_anchors.py` and pass it through
`--anchor-score-path`.

## Output and Configuration Identity

Each run contains:

- `metrics.csv`: one row per epoch;
- `summary.json`: full configuration, best epoch, and test evaluation;
- `checkpoint_best.pt`: local best-validation checkpoint.

The output directory includes an eight-character configuration ID. Machine
paths and worker count are excluded from that hash; scientific settings are
included. Never infer a full configuration from the folder name alone - read
`summary.json`.

## Artifact Availability

Verified local snapshot on 2026-09-28:

| Item | Count |
|---|---:|
| `summary.json` records | 71 |
| Matching `metrics.csv` records | 71 |
| Referenced local `checkpoint_best.pt` files | 43 |
| Checkpoints in incomplete run folders | 4 |
| Generated scratch-group run rows | 50 |
| Generated pretrained-group run rows | 21 |

All guided summaries reference neighbor payloads below
`results/experiments/shared/neighbors/`, but those payloads are not present in
this checkout. They must be regenerated before rerunning guided training.
Missing checkpoints do not invalidate the tracked CSV/JSON result record, but
they prevent resume or reevaluation from that saved state.

The artifact audit reports four local checkpoints in incomplete run
folders that require a retention decision. Because the audit treats the other
initialization directory as a sibling when scoped to one group, its
"noncanonical location" count should not be interpreted as an error for the
two-group `comparison_v1` layout.

## Result Maintenance

Regenerate figures and their `runs.csv` indexes:

```powershell
python src\graphs\plot_graphs.py --experiments-dir results\comparison_v1 --output-dir docs\figures
```

Audit each initialization group:

```powershell
python src\experiments\audit_artifacts.py --results-root results\comparison_v1\scratch --details
python src\experiments\audit_artifacts.py --results-root results\comparison_v1\pretrained --details
```

The audit is read-only unless `--json-output` is supplied. The generic
`build_manifest.py` still defaults to `results/experiments`; do not run it and
assume it updates the comparison-specific `runs.csv` files.

Before reporting a result:

1. Verify the summary/metrics pair and the best epoch.
2. Confirm initialization, architecture cohort, seeds, optimizer, schedule,
   batch size, and augmentation parameters.
3. Confirm whether the referenced checkpoint and guided support payload exist.
4. Record the Git commit and environment.
5. Use validation evidence for configuration selection.
6. Report multi-seed mean and variation when available; otherwise label the
   value as a single run.
