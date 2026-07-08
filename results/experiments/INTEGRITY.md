# Experiment Integrity Notes

Generated after centralizing outputs under `results/experiments/`.

The manifest contains 55 local summary/CSV metric pairs and all listed
`summary_path` and `metrics_path` values exist.

Some summary JSON files still reference artifacts that are not present locally.
These appear to be pre-existing missing artifacts, not files lost during the
centralization.

## Missing Checkpoints

The following `best_model_path` files are referenced by summaries but are absent:

| Summary collection | Missing artifact |
|---|---|
| `baseline_100e` | `results/experiments/shared/checkpoints/cifar100_resnet50_k100_seed0_augmix_epochs100_best.pt` |
| `baseline_100e` | `results/experiments/shared/checkpoints/cifar100_resnet50_k20_seed0_augmix_epochs100_best.pt` |
| `baseline_100e` | `results/experiments/shared/checkpoints/cifar100_resnet50_k50_seed0_augmix_epochs100_best.pt` |
| `baseline_100e` | `results/experiments/shared/checkpoints/cifar100_vit_k20_seed0_augmix_epochs100_best.pt` |
| `baseline_100e` | `results/experiments/shared/checkpoints/cifar100_vit_k50_seed0_augmix_epochs100_best.pt` |
| `simmixup_ablation_v2` | `results/experiments/simmixup_ablation_v2/checkpoints/cifar100_resnet50_k100_seed0_simmixup_class_agnostic_neighbors_class_agnostic_K40_nk20_r21-40_uniform_alpha0p7_mp1_warm0_epochs50_best.pt` |
| `simmixup_ablation_v2` | `results/experiments/simmixup_ablation_v2/checkpoints/cifar100_resnet50_k100_seed0_simmixup_class_agnostic_neighbors_class_agnostic_K40_nk20_r21-40_uniform_alpha1_mp0p75_warm0_epochs50_best.pt` |
| `simmixup_ablation_v2` | `results/experiments/simmixup_ablation_v2/checkpoints/cifar100_resnet50_k100_seed0_simmixup_class_agnostic_neighbors_class_agnostic_K60_nk20_r41-60_uniform_alpha1_mp1_warm0_epochs50_best.pt` |

## Missing Neighbor Artifact

The initial class-aware SimMixUp summary references:

```text
results/experiments/shared/neighbors/cifar100/k100_seed0/neighbors_class_aware_K20.pt
```

That exact file is absent. A K40 class-aware neighbor file exists for the same
k=100 split, but the summary is left pointing to the exact K20 path so the
missing artifact remains explicit.
