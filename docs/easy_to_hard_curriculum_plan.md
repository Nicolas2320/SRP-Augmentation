# Easy-to-Hard Pair Curriculum Plan

Status: complete. Three-seed validation was followed by a matched scratch
k=20 test evaluation of the curriculum and fixed-window control. The
curriculum did not improve on the fixed window.

## Frozen Question

Does changing SimCutMix partner difficulty over training improve validation
accuracy relative to a fixed less-similar window and random CutMix pairing?

The schedule is declared before examining curriculum results:

| Phase | Epochs in a 100-epoch run | Active neighbor ranks |
|---|---:|---:|
| Early | 1-33 | 21-40 |
| Middle | 34-66 | 11-30 |
| Late | 67-100 | 1-10 |

Rank 1 is the most similar saved neighbor. Partners therefore become more
similar as training progresses. Sampling remains uniform inside each active
window.

## Controlled Comparison

Use CIFAR-100 training from scratch at `k=20` and `k=100`, with paired
`(subset_seed, train_seed)` values `(0,0)`, `(1,1)`, and `(2,2)`.

| Condition | Mix probability | Source |
|---|---:|---|
| Random partner CutMix | 0.5 | Reuse completed three-seed CutMix results |
| Fixed SimCutMix ranks 21-40 | 0.5 | Reuse seed 0 pair ablation; add seeds 1 and 2 |
| Stepwise SimCutMix curriculum | 0.5 | Run seeds 0, 1, and 2 |

This required ten new full jobs: four fixed-window confirmations and six
curriculum jobs. They were run validation-only (`--skip-test`), so the schedule
was selected without test-set feedback.

## Execution Order

1. Run CPU tests locally.
2. Run one three-epoch GPU smoke job at scratch `k=20`.
3. Verify the log reports windows 21-40, 11-30, and 1-10 in epochs 1, 2, and 3.
4. Submit the ten validation-only jobs.
5. Compare paired validation deltas before deciding whether any final test
   confirmation is warranted.

## Results

Validation accuracy, mean ± sample SD:

| k | Random CutMix | Fixed ranks 21-40 | Easy-to-hard |
|---:|---:|---:|---:|
| 20 | 15.50 ± 1.20 | 16.78 ± 0.94 | 16.97 ± 0.78 |
| 100 | 52.03 ± 0.82 | 53.61 ± 1.24 | 53.65 ± 1.04 |

Scratch k=20 test accuracy:

| Condition | Test accuracy |
|---|---:|
| Crop+Flip | **17.51 ± 0.45** |
| Random CutMix | 15.54 ± 0.59 |
| Fixed ranks 21-40, p=0.5 | 17.18 ± 0.76 |
| Easy-to-hard curriculum, p=0.5 | 17.10 ± 0.85 |

The curriculum-minus-fixed paired delta is -0.07 ± 0.11 pp and the
curriculum wins 0/3 seeds. The fixed window is simpler and at least as good;
the curriculum should be reported as a negative extension result.
