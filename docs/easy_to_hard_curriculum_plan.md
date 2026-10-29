# Easy-to-Hard Pair Curriculum Plan

Status: implementation phase. This is an optional extension requested after the
third-presentation must-have experiments were completed.

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

This requires ten new full jobs: four fixed-window confirmations and six
curriculum jobs. All new jobs are validation-only (`--skip-test`). No test result
will be used to choose the schedule.

## Execution Order

1. Run CPU tests locally.
2. Run one three-epoch GPU smoke job at scratch `k=20`.
3. Verify the log reports windows 21-40, 11-30, and 1-10 in epochs 1, 2, and 3.
4. Submit the ten validation-only jobs.
5. Compare paired validation deltas before deciding whether any final test
   confirmation is warranted.
