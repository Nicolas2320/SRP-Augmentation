# SRP-Augmentation

Image augmentation experiments for small-data image classification.

This repository contains the implementation for our Student Research Project (SRP) at the University of Hildesheim. The project studies how different image augmentation methods affect classification performance when only a limited number of labeled training samples are available.

The current focus is on controlled low-data experiments using CIFAR datasets, ResNet50 and ViT, and reproducible k-shot training subsets.

---

## Project Motivation

Deep learning models usually require large labeled datasets. However, in many real-world domains, labeled data can be expensive, limited, or difficult to collect. In these low-data settings, models can overfit easily and fail to generalize well.

Data augmentation is a common strategy to improve generalization by increasing training diversity without collecting new labels. This project compares standard augmentation baselines and later aims to develop a small-data-oriented augmentation method.

---

## Research Question

Which image augmentation strategies improve classification performance most reliably in low-data image classification settings, and can a small-data-targeted augmentation variant outperform or complement standard augmentation baselines?

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
- Metrics saved as CSV files
- Experiment summaries saved as JSON files
- Best validation checkpoint saving
- Final test evaluation using the best validation checkpoint

### In Progress (Before First-Presentation)

- Final comparison tables and plots
- First Presentation Slides

---

## Repository Structure

```text
SRP-Augmentation/
│
├── data/
│   └── splits/                  # Reproducible k-shot split files
│
├── notebooks/                   # Exploratory notebooks
│
├── notes/                       # Project notes and documentation
│
├── results/
│   └── metrics/                 # CSV metrics and JSON summaries
│
├── src/
│   ├── augmentations/
│   │   ├── mixup.py
│   │   ├── cutmix.py
│   │   └── augmix.py
│   │
│   ├── data/
│   │   └── make_splits.py
│   │
│   ├── graphs/
│   │   └── plot_graphs.py       # Plot Accuracy and Loss functions
│   │
│   ├── models/
│   │   ├── resnet.py
│   │   └── vit.py
│   │
│   └── train.py
│
├── requirements.txt
└── README.md
