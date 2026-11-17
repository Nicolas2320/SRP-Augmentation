# Project Status

Last verified: 2026-11-14.

## Current Stage

The ResNet50 implementation and primary experiment program are complete. The
project is now in final analysis and presentation preparation. All primary
ResNet50 experiment cells are available; no additional large matrix is needed.

## Implemented Scope

- Fixed-validation k-shot splits for CIFAR-100 and Tiny ImageNet.
- ResNet50 classifiers trained from scratch or initialized from ImageNet.
- Crop+Flip, MixUp, CutMix, SimMixUp, SimCutMix, and exploratory AugMix.
- Frozen ImageNet-ResNet50 embeddings and exact class-agnostic neighbors.
- Configurable rank windows, sampling, mixing probability, and pair curricula.
- Validation-only selection, evaluation-only test passes, runtime provenance,
  artifact audits, cluster runners, and CPU/GPU smoke validation.

## Primary Evidence

| Evidence | Coverage | Status |
|---|---|---|
| CIFAR-100 final matrix | 2 initializations × 6 k × 4 methods × 3 seeds = 144 tested outcomes | Complete |
| Tiny ImageNet k=20/k=450 | 2 initializations × 2 k × 3 methods × 3 seeds = 36 outcomes | Complete |
| Tiny ImageNet k=100 | 18 of 18 required outcomes | Complete |
| Pair-selection ablation | Four regimes, six strategies, seed-0 validation | Complete |
| Easy-to-hard curriculum | k=20/k=100 validation; k=20 matched test evaluation | Complete negative result |
| Minimal ViT transfer | Delegated 18-run plan | Optional |

The authoritative aggregate values are in
`docs/third_presentation_results.md`, `docs/tiny_imagenet_results.md`, and
`docs/experiment_evidence_status.md`. Per-run summaries remain the underlying
source of truth.

## Current Scientific Conclusion

Similarity-guided CutMix is regime-dependent:

- It leads scratch CIFAR-100 at k=50 and k=100 and leads scratch Tiny ImageNet
  at k=20 and k=100.
- It does not consistently help with an ImageNet-pretrained classifier.
- Standard CutMix leads every pretrained CIFAR-100 budget.
- At full-data Tiny ImageNet, guided pairing provides no advantage.
- A stepwise easy-to-hard curriculum does not beat a matched fixed rank window.

The defensible claim is therefore narrower than “guided pairing always wins”:
it is useful when learning a classifier from scratch with scarce or moderate
labeled data.

## Important Caveats

- CIFAR-100 k=450 uses a predeclared 50-epoch recipe and subset seed 0, unlike
  the lower-k 100-epoch runs.
- Primary SimCutMix uses mixing probability 1 while standard CutMix uses 0.5;
  the separate pair-selection/curriculum controls use matched probability 0.5.
- Three seeds quantify limited run variability but do not justify claims of
  statistical significance for small differences.
- Historical pretrained seed-0 k=20/50/100 records had prior test exposure;
  configuration selection used validation evidence only.
- Local AugMix omits the canonical JSD consistency objective and is not a fair
  paper-faithful baseline.

## Remaining Work

1. Regenerate final figures and run indexes from the audited summaries.
2. Build the final presentation using `docs/presentation_outline.md`.
3. Optionally incorporate the delegated minimal ViT study if it finishes in
   time and passes the same provenance gates.

## Bottom Line

The supervisor-requested ResNet50 evidence is effectively complete. The final
presentation should emphasize a regime-specific positive result, the strong
standard CutMix pretrained baseline, cross-dataset confirmation, and the
negative curriculum result. Further broad experiment expansion would add less
value than producing clear audited figures and the final presentation.
