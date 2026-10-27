# Pair-selection ablation results

This controlled CIFAR-100 ablation holds the CutMix operator, alpha (1.0), mixing
probability (0.5), optimizer, schedule, subset seed, and training seed fixed. Only
the partner-selection rule changes. Values are best validation accuracy from seed 0;
they are selection evidence, not final multi-seed test estimates.

| Initialization | k | Random | Same class | Most similar | Middle | Less similar | Different label |
|---|---:|---:|---:|---:|---:|---:|---:|
| Scratch | 20 | 15.54 | 16.70 | 17.56 | 17.68 | 16.64 | 16.96 |
| Scratch | 100 | 52.98 | 53.30 | 54.06 | 53.68 | 54.90 | 54.32 |
| Scratch | 300 | 74.86 | 73.32 | 74.48 | 73.78 | 74.42 | 74.06 |
| Pretrained | 100 | 78.74 | 76.92 | 78.80 | 78.96 | 79.18 | 78.98 |

## Differences from random CutMix

| Initialization | k | Same class | Most similar | Middle | Less similar | Different label |
|---|---:|---:|---:|---:|---:|---:|
| Scratch | 20 | +1.16 | +2.02 | +2.14 | +1.10 | +1.42 |
| Scratch | 100 | +0.32 | +1.08 | +0.70 | +1.92 | +1.34 |
| Scratch | 300 | -1.54 | -0.38 | -1.08 | -0.44 | -0.80 |
| Pretrained | 100 | -1.82 | +0.06 | +0.22 | +0.44 | +0.24 |

## Interpretation

The ranks 21–40 condition has the largest mean validation difference across the four
settings (+0.75 pp versus random) and wins two settings.
It was therefore frozen before the three-seed confirmation runs. The result is not
universal: middle ranks win at scratch k=20, while random CutMix wins at scratch k=300.
Same-class pairing is weakest on average, especially for pretrained k=100.

`different_label` means the nearest eligible image with a different label; it is not
a truly most-distant or lowest-cosine-similarity condition.

## Test-boundary note

The 20 guided ablation runs were validation-only. Some reused random controls were
later test-evaluated as part of the frozen confirmation matrix, but configuration
selection used only the validation values recorded here.
