# CIFAR-100 k=max Sanity Check Commands

These commands run the supervisor-requested full-training-pool sanity check for
CIFAR-100 with ResNet50. In this repository, `k=max` means `--k 450` for
CIFAR-100 because the split keeps 50 validation images per class first, leaving
450 training images per class.

No learning-rate scheduler is used here. The commands keep the existing AdamW
setup with `--lr 0.001` and `--weight-decay 0.0001`.

## Baselines

```powershell
python -u src\train.py --dataset cifar100 --model resnet50 --k 450 --subset-seed 0 --train-seed 0 --augmentation none --epochs 100 --batch-size 64 --lr 0.001 --weight-decay 0.0001 --num-workers 2
python -u src\train.py --dataset cifar100 --model resnet50 --k 450 --subset-seed 0 --train-seed 0 --augmentation mixup --mixup-alpha 1 --epochs 100 --batch-size 64 --lr 0.001 --weight-decay 0.0001 --num-workers 2
python -u src\train.py --dataset cifar100 --model resnet50 --k 450 --subset-seed 0 --train-seed 0 --augmentation cutmix --epochs 100 --batch-size 64 --lr 0.001 --weight-decay 0.0001 --num-workers 2
python -u src\train.py --dataset cifar100 --model resnet50 --k 450 --subset-seed 0 --train-seed 0 --augmentation augmix --epochs 100 --batch-size 64 --lr 0.001 --weight-decay 0.0001 --num-workers 2
```

## Guided Neighbor Prep

Run these once before the guided SimMixUp/SimCutMix commands.

```powershell
python -u src\proposal\compute_embeddings.py --dataset cifar100 --k 450 --subset-seed 0 --encoder resnet50_imagenet --batch-size 128 --num-workers 2 --device auto
python -u src\proposal\build_neighbors.py --dataset cifar100 --k 450 --subset-seed 0 --encoder resnet50_imagenet --mode class_agnostic --max-neighbors 60 --query-batch-size 512 --device auto
```

Expected neighbor file:

```text
results\experiments\shared\neighbors\cifar100\k450_seed0\neighbors_class_agnostic_K60.pt
```

If neighbor construction runs out of GPU memory, lower `--query-batch-size` to
`256` or `128`. If CUDA is still tight, use `--device cpu`; it will be slower
but avoids GPU memory pressure.

## Guided Runs

These use the strongest SimMixUp setting from the current notes and the
mid-rank SimCutMix setting that gave the best guided result so far:
class-agnostic K60 neighbors, ranks 21-40, `nk=20`, `alpha=1`,
`mix_prob=1`, and no warmup.

```powershell
python -u src\train.py --dataset cifar100 --model resnet50 --k 450 --subset-seed 0 --train-seed 0 --augmentation simmixup --mixup-alpha 1 --epochs 100 --batch-size 64 --lr 0.001 --weight-decay 0.0001 --num-workers 2 --neighbor-path "results\experiments\shared\neighbors\cifar100\k450_seed0\neighbors_class_agnostic_K60.pt" --guided-mode class_agnostic --neighbor-k 20 --neighbor-rank-start 21 --pair-sampling uniform --mix-prob 1 --mix-warmup-epochs 0
python -u src\train.py --dataset cifar100 --model resnet50 --k 450 --subset-seed 0 --train-seed 0 --augmentation simcutmix --mixup-alpha 1 --epochs 100 --batch-size 64 --lr 0.001 --weight-decay 0.0001 --num-workers 2 --neighbor-path "results\experiments\shared\neighbors\cifar100\k450_seed0\neighbors_class_agnostic_K60.pt" --guided-mode class_agnostic --neighbor-k 20 --neighbor-rank-start 21 --pair-sampling uniform --mix-prob 1 --mix-warmup-epochs 0
```

## After Training

Rebuild the manifest so the new runs appear in `results/experiments/manifest.csv`.

```powershell
python -u src\experiments\build_manifest.py
```
