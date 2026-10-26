# Tiny ImageNet confirmation results

Final-test accuracy is reported as mean ± sample standard deviation across three paired seeds.
The test set is the labeled official Tiny ImageNet validation partition; the official
unlabeled test partition was never used.

## Scratch

| k | Crop+Flip | CutMix | SimCutMix |
|---:|---:|---:|---:|
| 20 | 12.71 ± 0.63 | 12.00 ± 0.90 | 14.87 ± 1.07 |
| 450 | 60.94 ± 0.69 | 60.56 ± 1.04 | 60.49 ± 0.62 |

## ImageNet pretrained

| k | Crop+Flip | CutMix | SimCutMix |
|---:|---:|---:|---:|
| 20 | 67.49 ± 0.55 | 66.84 ± 0.86 | 65.65 ± 0.55 |
| 450 | 65.44 ± 0.13 | 65.82 ± 0.14 | 65.54 ± 0.37 |

## Paired effects

- Scratch, k=20: SimCutMix − Crop+Flip +2.16 ± 0.54 pp; wins 3/3 seeds.
- Scratch, k=450: SimCutMix − Crop+Flip -0.45 ± 1.12 pp; wins 1/3 seeds.
- Pretrained, k=20: SimCutMix − Crop+Flip -1.85 ± 0.01 pp; wins 0/3 seeds.
- Pretrained, k=450: SimCutMix − Crop+Flip +0.11 ± 0.24 pp; wins 2/3 seeds.

## Interpretation

The positive result is regime-specific: SimCutMix improves low-data training from
scratch, but it does not improve the pretrained k=20 condition or either k=450
condition. The k=450 recipe differs in learning rate and epoch count, so changes
between k=20 and k=450 are not a pure data-size ablation.

The six k=100 screening runs remain validation-only and are excluded from final-test
claims because they were not repeated across three seeds.

## Provenance

- Final evaluation used Git commit `97cf21e1a93911020703ee3b6994c54570a2956e`.
- `35/36` evaluation records report `git_dirty=true` because concurrent
  jobs observed other tracked summary files being updated; source code and scientific
  configurations were unchanged.
- Validation accuracy recomputed from every saved checkpoint exactly matched the
  validation accuracy used to select that checkpoint.
