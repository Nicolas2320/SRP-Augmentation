# Experiments v2 Results

Generated: 2026-07-03

- Source: `results/experiments_v2/metrics/*_summary.json` plus the matching epoch CSV files.
- Scope: complete summaries only. All complete runs here use `cifar100`, `resnet50`, `k=100`, `subset_seed=0`, `train_seed=0`, and `epochs=50`.
- `train_acc` is taken from the CSV row at `best_epoch`; `val_acc` and `test_acc` come from the summary JSON.
- `test_acc` is the evaluation of the best validation checkpoint.
- `gap` is `train_acc - val_acc`.
- `delta_vs_mixup` is relative to the `mixup` test accuracy, `0.3664`.
- Git ref: `experiments-v2` at `b9e6104ea5b3bde5f986cea3ffecb4b77bd7b305`.

| Rank | Method | Guided mode | Neighbor file | nk | Rank window | Alpha | Mix prob | Warmup | Best epoch | Train acc | Val acc | Test acc | Test loss | Gap | Delta vs mixup |
|---:|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | simmixup | class_agnostic | `neighbors_class_agnostic_K60.pt` | 20 | 21-40 | 1.0 | 1.00 | 0 | 28 | 0.7866 | 0.3814 | 0.3769 | 2.6007 | 0.4052 | +0.0105 |
| 2 | simmixup | class_agnostic | `neighbors_class_agnostic_K40.pt` | 20 | 21-40 | 1.0 | 1.00 | 0 | 28 | 0.7848 | 0.3744 | 0.3692 | 2.5621 | 0.4104 | +0.0028 |
| 3 | mixup | class_aware (no apply) | n/a | n/a | n/a | 1.0 | 1.00 | 0 | 28 | 0.6659 | 0.3654 | 0.3664 | 2.6172 | 0.3005 | 0.0000 |
| 4 | simmixup | class_agnostic | `neighbors_class_agnostic_K60.pt` | 20 | 41-60 | 1.0 | 1.00 | 0 | 28 | 0.7670 | 0.3668 | 0.3657 | 2.5456 | 0.4002 | -0.0007 |
| 5 | simmixup | class_agnostic | `neighbors_class_agnostic_K40.pt` | 20 | 21-40 | 1.0 | 0.50 | 0 | 30 | 0.8954 | 0.3658 | 0.3635 | 2.7392 | 0.5296 | -0.0029 |
| 6 | simmixup | class_agnostic | `neighbors_class_agnostic_K40.pt` | 20 | 21-40 | 0.7 | 1.00 | 0 | 49 | 0.8354 | 0.3692 | 0.3632 | 2.5538 | 0.4662 | -0.0032 |
| 7 | simmixup | class_agnostic | `neighbors_class_agnostic_K40.pt` | 20 | 21-40 | 0.4 | 1.00 | 0 | 48 | 0.8814 | 0.3618 | 0.3606 | 2.5931 | 0.5196 | -0.0058 |
| 8 | simmixup | class_agnostic | `neighbors_class_agnostic_K20.pt` | 20 | 1-20 | 1.0 | 1.00 | 0 | 39 | 0.8233 | 0.3664 | 0.3594 | 2.6958 | 0.4569 | -0.0070 |
| 9 | simmixup | class_agnostic | `neighbors_class_agnostic_K60.pt` | 40 | 21-60 | 1.0 | 1.00 | 0 | 49 | 0.7824 | 0.3626 | 0.3590 | 3.2581 | 0.4198 | -0.0074 |
| 10 | simmixup | class_agnostic | `neighbors_class_agnostic_K40.pt` | 20 | 21-40 | 1.0 | 0.75 | 0 | 30 | 0.8373 | 0.3690 | 0.3547 | 2.6945 | 0.4683 | -0.0117 |
| 11 | simmixup | class_agnostic | `neighbors_class_agnostic_K40.pt` | 20 | 21-40 | 1.0 | 1.0 | 10 | 38 | 0.7948 | 0.3656 | 0.3570 | 2.5972 | 0.4292 | -0.0094 |
| 12 | simmixup | class_agnostic | `neighbors_class_agnostic_K40.pt` | 20 | 21-40 | 1.0 | 1.0 | 5 | 46 | 0.8137 | 0.3602 | 0.3641 | 2.8019 | 0.4535 | -0.0023 |
| 11 | none | class_aware (no apply) | n/a | n/a | n/a | 1.0 | 1.00 | 0 | 46 | 0.9879 | 0.3206 | 0.3147 | 5.6705 | 0.6673 | -0.0517 |

## Checkpoints Without Summary Metrics

These checkpoint files exist under `results/experiments_v2/checkpoints`, but no matching `*_summary.json` was found under `results/experiments_v2/metrics`, so they are not included in the ranking table.

| Checkpoint run | Status |
|---|---|
| `cifar100_resnet50_k100_seed0_simmixup_class_agnostic_neighbors_class_agnostic_K40_nk20_r21-40_uniform_alpha1_mp1_warm10_epochs50` | Missing summary JSON |
| `cifar100_resnet50_k100_seed0_simmixup_different_label_neighbors_different_label_K40_nk20_r1-20_uniform_alpha1_mp1_warm0_epochs50` | Missing summary JSON |
| `cifar100_resnet50_k100_seed0_simmixup_different_label_neighbors_different_label_K40_nk20_r21-40_uniform_alpha1_mp1_warm0_epochs50` | Missing summary JSON |
