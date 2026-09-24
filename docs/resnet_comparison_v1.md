# ResNet-50 comparison v1 — proposed fixed protocol

ResNet-50 defaults to scratch in the CLI, model builder and execution script.
Use `--pretrained` / `--no-pretrained` in Python, or `-Initialization pretrained`
/ `-Initialization scratch` in PowerShell. The summary records `pretrained`
and its value participates in the configuration ID. Historical summaries
without this field have unknown initialization for automatic matching.

All runs use CIFAR-100, subset seed 0, train seed 0, batch 32, SGD, LR 0.01, momentum 0.9, Nesterov, weight decay 0.0005, gamma 0.2.
k=20/50/100: 100 epochs, milestones 30/60/80. k=450: 50 epochs, milestones 15/30/40.
Same ResNet-50 architecture, 224x224 input and ImageNet normalization in both regimes; only initialization differs. All parameters are trainable.

| Display label | CLI method |
|---|---|
| No augmentation (deterministic resize/normalize only) | raw |
| Crop + Flip | none |
| MixUp | mixup |
| CutMix | cutmix |
| AugMix (without JSD consistency loss) | augmix |
| SimMixUp original | simmixup |
| SimCutMix original | simcutmix |

All methods except raw include crop+flip. Mixing alpha=1, mixing probability=1, no warmup. Guided methods use fixed ImageNet-encoder neighbors, class_agnostic, ranks 21–40, uniform sampling, 100% guidance, random patch location. Thus 'scratch' refers to the classifier initialization; guidance still uses an external pretrained encoder.
AugMix uses the existing implementation: severity=3, width=3, random depth, alpha=1, ordinary cross entropy, no JSD loss.

Outputs: results/comparison_v1/scratch or results/comparison_v1/pretrained. Old experiments are retained as historical evidence, excluded from this fixed-protocol cohort. Reusing old runs requires matching the full recipe; the inspected historical runs do not match this recipe.
Best checkpoint selected by validation accuracy; test evaluated once at the end. Report clean train-validation gap, validation last-10 mean and variation, and test accuracy separately. One seed is exploratory; epoch variation is not seed standard deviation. This tests a common fixed recipe, not each regime's optimal hyperparameters.

## Execution

Run from the project root in PowerShell. Each line below runs ONE experiment. Omitting -Execute prints the underlying Python command without training. No training was launched while preparing this plan.

56 total runs. Prioritize CutMix and SimCutMix at k=20/50/100 in both regimes; then remaining methods; k=450 last.

## scratch

```powershell
.\scripts\run_resnet_comparison.ps1 -Initialization scratch -K 20 -Method raw -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization scratch -K 20 -Method none -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization scratch -K 20 -Method mixup -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization scratch -K 20 -Method cutmix -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization scratch -K 20 -Method augmix -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization scratch -K 20 -Method simmixup -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization scratch -K 20 -Method simcutmix -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization scratch -K 50 -Method raw -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization scratch -K 50 -Method none -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization scratch -K 50 -Method mixup -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization scratch -K 50 -Method cutmix -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization scratch -K 50 -Method augmix -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization scratch -K 50 -Method simmixup -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization scratch -K 50 -Method simcutmix -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization scratch -K 100 -Method raw -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization scratch -K 100 -Method none -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization scratch -K 100 -Method mixup -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization scratch -K 100 -Method cutmix -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization scratch -K 100 -Method augmix -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization scratch -K 100 -Method simmixup -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization scratch -K 100 -Method simcutmix -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization scratch -K 450 -Method raw -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization scratch -K 450 -Method none -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization scratch -K 450 -Method mixup -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization scratch -K 450 -Method cutmix -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization scratch -K 450 -Method augmix -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization scratch -K 450 -Method simmixup -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization scratch -K 450 -Method simcutmix -Execute
```

## pretrained

```powershell
.\scripts\run_resnet_comparison.ps1 -Initialization pretrained -K 20 -Method raw -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization pretrained -K 20 -Method none -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization pretrained -K 20 -Method mixup -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization pretrained -K 20 -Method cutmix -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization pretrained -K 20 -Method augmix -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization pretrained -K 20 -Method simmixup -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization pretrained -K 20 -Method simcutmix -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization pretrained -K 50 -Method raw -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization pretrained -K 50 -Method none -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization pretrained -K 50 -Method mixup -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization pretrained -K 50 -Method cutmix -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization pretrained -K 50 -Method augmix -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization pretrained -K 50 -Method simmixup -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization pretrained -K 50 -Method simcutmix -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization pretrained -K 100 -Method raw -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization pretrained -K 100 -Method none -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization pretrained -K 100 -Method mixup -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization pretrained -K 100 -Method cutmix -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization pretrained -K 100 -Method augmix -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization pretrained -K 100 -Method simmixup -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization pretrained -K 100 -Method simcutmix -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization pretrained -K 450 -Method raw -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization pretrained -K 450 -Method none -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization pretrained -K 450 -Method mixup -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization pretrained -K 450 -Method cutmix -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization pretrained -K 450 -Method augmix -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization pretrained -K 450 -Method simmixup -Execute
.\scripts\run_resnet_comparison.ps1 -Initialization pretrained -K 450 -Method simcutmix -Execute
```
