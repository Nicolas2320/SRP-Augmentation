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
|   |-- checkpoints/            # Best validation checkpoints, ignored by Git
|   |-- figures/                # Generated plots, ignored by Git
|   `-- metrics/                # CSV metrics and JSON summaries
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
