# ResNet-50 comparison v1 — proposed fixed protocol

ResNet-50 defaults to scratch in the CLI, model builder and execution script.
Use `--pretrained` / `--no-pretrained` in Python, or `-Initialization pretrained`
/ `-Initialization scratch` in PowerShell. The summary records `pretrained`
and its value participates in the configuration ID. Historical summaries
without this field have unknown initialization for automatic matching.

All runs use CIFAR-100, subset seed 0, train seed 0, batch 64, SGD, momentum 0.9, Nesterov, weight decay 0.0005.
k=20/50/100: 100 epochs, LR 0.01, milestones 30/55/75, gamma 0.1.
k=450: 50 epochs, LR 0.1, milestones 15/30/40, gamma 0.2.
These schedules follow the majority of historical scratch runs in GitHub main (ae782b5).
AugMix follows the common per-k schedule here, rather than the historical AugMix exception (100 epochs, LR 0.1, milestones 30/60/80, gamma 0.2 at every k).
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

All methods except raw include crop+flip. Mixing alpha=1, no warmup. Baseline CutMix probability=0.5, as in historical main; MixUp and the guided methods mix with probability=1. Therefore CutMix versus SimCutMix changes mixing frequency as well as partner selection, and is not a pairing-only ablation. Guided methods use fixed ImageNet-encoder neighbors, class_agnostic, ranks 21–40, uniform sampling, 100% guidance, random patch location. Thus 'scratch' refers to the classifier initialization; guidance still uses an external pretrained encoder.
AugMix uses the existing implementation: severity=3, width=3, random depth, alpha=1, ordinary cross entropy, no JSD loss.

Outputs: results/comparison_v1/scratch or results/comparison_v1/pretrained. Imported old experiments are historical evidence, excluded from this fixed-protocol cohort. Their optimizer schedules now match for the majority of methods, but historical main used a CIFAR-adapted 3x3/stride-1 stem, no initial maxpool, 32x32 inputs and CIFAR normalization. Current scratch and pretrained both use the standard 224x224 ResNet-50 pipeline. Old runs therefore remain a different architecture/preprocessing cohort.
Best checkpoint selected by validation accuracy; test evaluated once at the end. Report clean train-validation gap, validation last-10 mean and variation, and test accuracy separately. One seed is exploratory; epoch variation is not seed standard deviation. This tests a common fixed recipe, not each regime's optimal hyperparameters.

## Execution

### Imported AugMix evidence

The eight historical AugMix runs (ResNet50 and ViT, k=20/50/100/450) are
stored under `results/comparison_v1/scratch/cifar100/<model>/k<k>/augmix/`.
Initialization was inferred from the model code at the commits introducing
the results: `be0824b` for ResNet50 (`weights=None`, CIFAR-adapted stem), and
`c41ba00` for ViT (randomly initialized VisionTransformer). Each summary
records `pretrained=false`, the source commit and original paths in provenance.
The CSV metrics and reported accuracies are unchanged. Checkpoints were not
included in the imported Git files.

These runs appear in the scratch accuracy-by-k and coverage figures, with a
separate model panel for ViT. Their original 100-epoch LR 0.1 schedule with
milestones 30/60/80 and gamma 0.2 is preserved. They are historical evidence,
not new fixed-protocol runs; recipe-matched comparisons still require the
same training recipe.

### Figures

Use the same plotting program for both initializations:

```powershell
python src/graphs/plot_graphs.py --experiments-dir results/comparison_v1 --output-dir docs/figures
```

The default output root is `docs/figures`. Outputs are written under
`docs/figures/scratch/` and `docs/figures/pretrained/`, each with
the same figure types and a `runs.csv` listing the contributing summaries.
Historical and current scratch runs share `docs/figures/scratch/`; rerunning
the plotter replaces the generated figures and `runs.csv` in that directory.
Training records are not overwritten. Summaries without an
explicit initialization go to `docs/figures/unknown/`. Empty groups produce no figures.
Use `--initialization pretrained` or `--initialization scratch` to select a regime.
Different training recipes are not averaged as repeated seeds.

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
