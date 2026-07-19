# SRP-Augmentation

Image augmentation experiments for small-data image classification.

This repository contains the implementation for our Student Research Project
(SRP) at the University of Hildesheim. The project studies how different image
augmentation methods affect classification performance when only a limited
number of labeled training samples are available.

The current focus is on controlled low-data experiments using CIFAR-10,
CIFAR-100, ResNet50, ViT, and reproducible k-shot training subsets.

---

## Project Motivation

Deep learning models usually require large labeled datasets. However, in many
real-world domains, labeled data can be expensive, limited, or difficult to
collect. In these low-data settings, models can overfit easily and fail to
generalize well.

Data augmentation is a common strategy to improve generalization by increasing
training diversity without collecting new labels. This project compares standard
augmentation baselines and later aims to develop a small-data-oriented
augmentation method.

---

## Research Question

Which image augmentation strategies improve classification performance most
reliably in low-data image classification settings, and can a
small-data-targeted augmentation variant outperform or complement standard
augmentation baselines?

---

## Current Project Status

### Implemented

- Reproducible CIFAR-10 and CIFAR-100 k-shot split generation
- Fixed validation split
- ResNet50 adapted for CIFAR-size images
- ViT adapted for CIFAR-size images
- Unified training entry point in `src/train.py`
- No-augmentation baseline
- MixUp augmentation
- CutMix augmentation
- AugMix augmentation
- Seeded augmentation, DataLoader, PyTorch, and CUDA/cuDNN behavior where possible
- Metrics saved as CSV files
- Experiment summaries saved as JSON files
- Best validation checkpoint saving
- Final test evaluation using the best validation checkpoint
- Plot generation from saved metrics

### In Progress

- Final comparison tables and plots
- Multi-seed result aggregation
- First presentation slides

---

## Repository Structure

```text
SRP-Augmentation/
|
|-- data/
|   |-- raw/                    # Local CIFAR downloads, ignored by Git
|   `-- splits/                 # Reproducible k-shot split files
|
|-- notebooks/                  # Exploratory notebooks
|-- notes/                      # Project notes and documentation
|
|-- results/
|   `-- experiments/            # Central experiment outputs and manifest
|       |-- manifest.csv         # Index of local experiment summaries
|       |-- cifar100/            # dataset/model/k/method experiment runs
|       |   |-- resnet50/
|       |   `-- vit/
|       `-- shared/              # Shared neighbors, anchor scores, figures
|
|-- src/
|   |-- augmentations/
|   |   |-- mixup.py
|   |   |-- cutmix.py
|   |   `-- augmix.py
|   |
|   |-- data/
|   |   `-- make_splits.py
|   |
|   |-- graphs/
|   |   `-- plot_graphs.py
|   |
|   |-- models/
|   |   |-- resnet.py
|   |   `-- vit.py
|   |
|   `-- train.py
|
|-- requirements.txt
`-- README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

The requirements use compatible version ranges instead of exact pins. This keeps
the project reproducible within the tested dependency family while still
allowing patch updates.

---

## Data Splits

Generate CIFAR-10 and CIFAR-100 splits with:

```bash
python src/data/make_splits.py
```

The split generator first creates a fixed validation split, then samples k-shot
training subsets from the remaining training pool. This prevents overlap between
training and validation indices.

The generated CIFAR-100 maximum split is `k=450`: 450 images per class for
training and 50 images per class for validation. The maximum split is generated
only for `subset_seed=0` because it contains the complete post-validation pool.

---

## Training

Example run:

```bash
python src/train.py \
  --dataset cifar100 \
  --model resnet50 \
  --k 20 \
  --subset-seed 0 \
  --augmentation cutmix \
  --epochs 100
```

The default training configuration follows a 100-epoch CIFAR-style recipe:

- 32x32 CIFAR-style model input
- random crop with 4 pixels of padding and random horizontal flip
- SGD with Nesterov momentum 0.9
- initial learning rate 0.1
- learning-rate multiplier 0.2 after epochs 30, 60, and 80
- weight decay 0.0005
- batch size 128
- CutMix probability 0.5

The spatial crop and flip are applied to every training method. Therefore,
`--augmentation none` means no additional mixing method; it still uses the
standard CIFAR spatial augmentation.

Full post-validation CIFAR-100 sanity check with CutMix:

```powershell
python -u src\train.py --dataset cifar100 --model resnet50 --k 450 --subset-seed 0 --train-seed 0 --augmentation cutmix --cutmix-prob 0.5 --epochs 100 --batch-size 128 --optimizer sgd --lr 0.1 --momentum 0.9 --nesterov --weight-decay 0.0005 --lr-milestones 30 60 80 --lr-gamma 0.2 --num-workers 2
```

To transfer the best k=100 SimCutMix configuration to the successful 50-epoch
k=450 recipe, first compute ImageNet ResNet50 embeddings and build the 40 exact
nearest neighbors. Neighbor search is blockwise so it does not allocate the
full 45,000-by-45,000 similarity matrix:

```powershell
python -u src\proposal\compute_embeddings.py --dataset cifar100 --k 450 --subset-seed 0 --encoder resnet50_imagenet --batch-size 64 --num-workers 2 --device auto
python -u src\proposal\build_neighbors.py --dataset cifar100 --k 450 --subset-seed 0 --encoder resnet50_imagenet --mode class_agnostic --max-neighbors 40 --query-batch-size 512 --device auto
python -u src\proposal\inspect_neighbors.py --dataset cifar100 --k 450 --subset-seed 0 --encoder resnet50_imagenet --mode class_agnostic --max-neighbors 40
```

Then run SimCutMix using the original best rank window (ranks 21-40) with the
same optimizer, spatial augmentation, and schedule as the k=450 CutMix run:

```powershell
python -u src\train.py --dataset cifar100 --model resnet50 --k 450 --subset-seed 0 --train-seed 0 --augmentation simcutmix --mixup-alpha 1 --epochs 50 --batch-size 64 --optimizer sgd --lr 0.1 --momentum 0.9 --nesterov --weight-decay 0.0005 --lr-milestones 15 30 40 --lr-gamma 0.2 --num-workers 2 --neighbor-path "results\experiments\shared\neighbors\cifar100\k450_seed0\neighbors_class_agnostic_K40.pt" --guided-mode class_agnostic --neighbor-k 20 --neighbor-rank-start 21 --pair-sampling uniform --mix-prob 1 --mix-warmup-epochs 0
```

If neighbor construction runs out of GPU memory, reduce
`--query-batch-size 512` to `256` or `128`. This does not change the exact
neighbors; it only uses smaller search blocks.

If batch size 128 does not fit in GPU memory, use batch size 64 and learning
rate 0.05 as a proportional starting point.

Supported datasets:

- `cifar10`
- `cifar100`

Supported models:

- `resnet50`
- `vit`

Supported augmentations:

- `none`
- `mixup`
- `cutmix`
- `augmix`

---

## Reproducibility

Training uses `--train-seed` to seed Python, NumPy, PyTorch, DataLoader shuffling,
DataLoader workers, augmentation RNGs, and CUDA/cuDNN deterministic behavior
where possible.

Some GPU operations can still vary slightly depending on hardware, driver,
PyTorch version, and CUDA backend behavior. For reporting results, compare
multiple `subset-seed` and `train-seed` values rather than relying on a single
run.

---

## Dataset Downloading

The training and split scripts currently use `download=True` for torchvision
CIFAR datasets. This means:

- If the dataset is missing in `data/raw`, torchvision downloads it.
- If the dataset already exists, torchvision reuses it.

Changing this to `download=False` means:

- Runs become stricter and cluster-friendly.
- The dataset must already exist in `data/raw`.
- If the dataset is missing, the script fails immediately instead of downloading.

For local development, `download=True` is convenient. For offline or cluster of the University.
runs, `download=False` is usually better after the data has been staged.

---

## Plotting

Generate accuracy and loss plots from saved metric CSV files:

```bash
python src/graphs/plot_graphs.py
```
