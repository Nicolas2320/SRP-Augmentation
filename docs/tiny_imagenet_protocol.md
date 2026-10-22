# Tiny ImageNet Protocol

This document freezes the dataset semantics before any Tiny ImageNet result is
used as scientific evidence.

## Dataset policy

Tiny ImageNet contains 200 classes, 500 official training images per class, 50
labeled official-validation images per class, and an unlabeled official test
partition. The project uses these partitions as follows:

| Project role | Tiny ImageNet source | Size |
|---|---|---:|
| Training/selection source | Official training partition | 100,000 |
| Fixed validation | 50/class reserved from official training | 10,000 |
| Maximum balanced training pool | Remaining 450/class | 90,000 |
| Final test | Labeled official validation partition | 10,000 |
| Unused | Unlabeled official test partition | 10,000 |

Therefore `k` means labeled training images per class after the fixed validation
reservation, and `k_max=450`. The fixed validation indices use seed 123. For
k=20 and k=100, subset seeds 0/1/2 produce deterministic nested balanced
subsets. The full k=450 pool has subset seed 0 only; training seeds measure
optimization variability.

Class IDs and numeric labels always follow `wnids.txt`. Both the official train
folders and `val_annotations.txt` are mapped through that same ordering.

## Local or login-node preparation

Download `tiny-imagenet-200.zip` through an authorized source, then run:

```bash
python src/data/prepare_tiny_imagenet.py \
  --archive /path/to/tiny-imagenet-200.zip \
  --data-root data/raw \
  --cache-resnet50

python src/data/make_splits.py --dataset tinyimagenet
python src/data/audit_splits.py --dataset tinyimagenet
```

The preparation command rejects unsafe archive paths, verifies 100,000
training images and 10,000 labeled test images, and optionally caches the
ImageNet ResNet50 weights before compute-node use. The audit must report seven
split files: k=20 and k=100 for seeds 0/1/2, plus k=450 seed 0.

Do not commit or push the image dataset, pretrained weight cache, embeddings,
neighbor tensors, or checkpoints. Commit the generated split JSON files,
metadata, checksums, metrics, and summaries.

## Frozen model and training policy

- Classifier: unmodified torchvision ResNet50 at 224×224.
- Initializations: random and ImageNet pretrained.
- Native spatial augmentation: random 64×64 crop with padding 8, then
  horizontal flip; resize to 224 follows the native-resolution operation.
- Normalization: ImageNet mean and standard deviation for both initializations.
- Methods: Crop+Flip (`none`), CutMix, and frozen SimCutMix.
- Frozen SimCutMix: ImageNet ResNet50 embeddings, class-agnostic ranks 21–40,
  uniform sampling, alpha 1, mixing probability 1, no warmup.
- k=20/100: 100 epochs, LR 0.01, milestones 30/55/75, gamma 0.1.
- k=450: 50 epochs, LR 0.1, milestones 15/30/40, gamma 0.2.
- Optimizer: SGD, momentum 0.9, Nesterov, weight decay 0.0005.
- Batch size: 32.

The k=450 schedule differs for compute feasibility. Consequently, changes from
k=100 to k=450 are not a pure data-size ablation and must be described as such.

## Engineering gate

After committing and pulling the implementation and split files on the cluster:

```bash
sbatch scripts/build_guided_support.slurm tinyimagenet 20 0 class_agnostic
```

When support validation passes, run one-epoch smoke jobs:

```bash
sbatch scripts/run_tiny_imagenet.slurm pretrained 20 none smoke 0 0
sbatch scripts/run_tiny_imagenet.slurm scratch 20 simcutmix smoke 0 0
```

Both jobs must complete, use 200 output classes, report 4,000 training examples,
10,000 fixed-validation examples, 10,000 final-test examples, and keep the test
set skipped in smoke mode. Only then submit the 18-cell seed-0 screening matrix.
