# Reproducibility

This project separates reproducibility into four layers: data selection,
training randomness, software environment, and artifact availability. The
first two are controlled by the code. The latter two require care when
reproducing historical runs.

For the system overview, see [architecture.md](architecture.md). For current
research status and known result discrepancies, see
[project_status.md](project_status.md).

## What Is Reproducible

### Data selection

The JSON files under `data/splits/` are committed and are the authoritative
sample selections. Each split records original CIFAR training-set indices.

- The fixed validation split is created once per dataset.
- k-shot subsets are drawn only from the remaining training pool.
- `subset_seed` selects a committed k-shot subset.
- The maximum post-validation split uses seed 0 because it contains the entire
  remaining pool.

Existing experiments should use the committed files rather than regenerate
them. Run `src/data/make_splits.py` only when intentionally validating or
rebuilding the split collection.

### Training randomness

`train_seed` controls the main sources of run randomness:

- Python;
- NumPy;
- PyTorch CPU and CUDA;
- data-loader shuffling and workers;
- augmentation random-number generators;
- transform seeding;
- CUDA/cuDNN deterministic behavior where available.

`subset_seed` and `train_seed` answer different questions. The former changes
which labeled examples are available; the latter changes optimization,
shuffling, and augmentation randomness for the same examples.

Exact GPU equality is not guaranteed across PyTorch, CUDA, driver, or hardware
versions. Final claims should therefore use multiple subset and training seeds,
not a single deterministic-looking run.

## Environment Setup

Create an isolated environment from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The current `requirements.txt` selects CUDA 12.8 builds of PyTorch and
TorchVision. It is an installation specification, not a complete lock file.
It does not currently capture the Python version, CUDA driver, operating
system, or every resolved transitive dependency.

Before running a final experiment, record at least:

```powershell
python --version
python -c "import torch, torchvision, numpy; print(torch.__version__); print(torchvision.__version__); print(numpy.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
git rev-parse HEAD
```

Historical `summary.json` files do not yet contain this environment metadata.
They preserve the training configuration and result paths, but exact software
and hardware reconstruction may require external notes.

## Verification Before Training

Run the test suite:

```powershell
python -m unittest discover -s tests -v
```

The tests do not download CIFAR or run a full experiment. They cover split
logic, indexed datasets, neighbor construction, SimMixUp, SimCutMix, and the
standard training recipe.

Confirm that the required inputs exist:

```powershell
Test-Path data\splits\cifar100\k20_seed0.json
Test-Path data\splits\cifar100\fixed_validation_split.json
```

Raw CIFAR data is downloaded into `data/raw/` when missing. That directory is
ignored by Git.

## Reproducing a Standard Run

Use all material hyperparameters explicitly in commands intended for a report.
For example:

```powershell
python -u src\train.py --dataset cifar100 --model resnet50 --k 20 --subset-seed 0 --train-seed 0 --augmentation cutmix --mixup-alpha 1 --cutmix-prob 0.5 --epochs 100 --batch-size 128 --optimizer sgd --lr 0.1 --momentum 0.9 --nesterov --weight-decay 0.0005 --lr-milestones 30 60 80 --lr-gamma 0.2 --num-workers 2
```

Although `--mixup-alpha` is included in some shared command templates, it does
not affect baseline CutMix; `--cutmix-prob` is the relevant CutMix-specific
parameter.

The run writes:

- `metrics.csv` for per-epoch values;
- `summary.json` for configuration and final evaluation;
- `checkpoint_best.pt` for the best validation model.

The test set is evaluated after training using the best validation checkpoint.

## Reproducing a Similarity-Guided Run

Guided training has a dependency chain. All stages must use the same dataset,
k, subset seed, encoder, and compatible neighbor settings.

### 1. Compute embeddings

```powershell
python -u src\proposal\compute_embeddings.py --dataset cifar100 --k 20 --subset-seed 0 --encoder resnet50_imagenet --batch-size 64 --num-workers 2 --device auto
```

### 2. Build and inspect neighbors

```powershell
python -u src\proposal\build_neighbors.py --dataset cifar100 --k 20 --subset-seed 0 --encoder resnet50_imagenet --mode class_agnostic --max-neighbors 40 --query-batch-size 512 --device auto
python -u src\proposal\inspect_neighbors.py --dataset cifar100 --k 20 --subset-seed 0 --encoder resnet50_imagenet --mode class_agnostic --max-neighbors 40
```

Reducing `--query-batch-size` changes memory use but not the exact neighbor
result.

### 3. Train with the saved neighbor payload

```powershell
python -u src\train.py --dataset cifar100 --model resnet50 --k 20 --subset-seed 0 --train-seed 0 --augmentation simcutmix --mixup-alpha 1 --epochs 50 --batch-size 64 --optimizer sgd --lr 0.1 --momentum 0.9 --nesterov --weight-decay 0.0005 --lr-milestones 15 30 40 --lr-gamma 0.2 --num-workers 2 --neighbor-path "results\experiments\shared\neighbors\cifar100\k20_seed0\neighbors_class_agnostic_K40.pt" --guided-mode class_agnostic --neighbor-k 20 --neighbor-rank-start 21 --pair-sampling uniform --mix-prob 1 --mix-warmup-epochs 0
```

The neighbor file contains more neighbors than a run necessarily uses. The
rank window above selects ranks 21–40 from a saved K40 payload.

Anchor-gated runs add a score-generation stage with
`src/proposal/score_anchors.py` and pass the resulting CSV through
`--anchor-score-path`.

## Artifact Availability

Git tracks the small records needed to inspect results:

- split JSON files;
- epoch metric CSV files;
- run summary JSON files;
- the experiment manifest;
- documentation and tests.

Git ignores large or generated artifacts:

- raw datasets;
- model checkpoints;
- embedding and neighbor `.pt` payloads;
- generated plot images.

As a result, a fresh clone can inspect reported results and run standard
experiments, but it must regenerate or obtain the relevant `.pt` payloads to
resume a checkpoint or run a guided method.

Two active ViT AugMix summaries reference local checkpoints that are no longer
present. These cases are recorded in
[project_status.md](project_status.md); their CSV and JSON result records remain
available.

## Result Maintenance

After adding or moving canonical summaries, rebuild the manifest:

```powershell
python src\experiments\build_manifest.py
```

Audit the availability and classification of local artifacts with:

```powershell
python src\experiments\audit_artifacts.py --details
```

This command is read-only. Use `--json-output <path>` only when a
machine-readable local snapshot is needed.

Generate the current comparison figures with:

```powershell
python src\graphs\plot_graphs.py
```

By default, the command writes the four curated, versioned figures to
`docs/figures/`. Review the regenerated images and commit them with the
underlying result changes so the GitHub README remains synchronized with the
experiment records. Use `--output-dir` for temporary local figures.

The direct-comparison figures pair every available SimMixUp or SimCutMix run
with all standard baseline augmentations whose dataset, model, k, seeds, epoch
budget, optimizer, and learning-rate recipe match. Augmentation-specific
settings remain properties of the compared method configurations. A single run
is plotted without an uncertainty interval.

Before reporting a result:

1. Verify the summary and metrics pair.
2. Confirm the subset and training seeds.
3. Confirm the optimizer, schedule, batch size, and augmentation parameters.
4. Record the source commit and environment.
5. Prefer multi-seed mean and variation over a single run.
6. Document missing checkpoints or support payloads instead of implying that
   they are available.
