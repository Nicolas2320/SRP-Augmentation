import unittest

import pandas as pd

from src.experiments.analyze_third_presentation import (
    aggregate_runs,
    expected_subset_seed,
    paired_deltas,
)


class ThirdPresentationAnalysisTests(unittest.TestCase):
    def test_k450_reuses_subset_zero(self):
        self.assertEqual(expected_subset_seed(450, 2), 0)
        self.assertEqual(expected_subset_seed(300, 2), 2)

    def test_aggregation_uses_sample_standard_deviation(self):
        runs = pd.DataFrame(
            {
                "initialization": ["scratch"] * 3,
                "k": [20] * 3,
                "method": ["none"] * 3,
                "best_val_acc": [0.1, 0.2, 0.3],
                "test_acc": [0.2, 0.3, 0.4],
            }
        )
        row = aggregate_runs(runs).iloc[0]
        self.assertEqual(row["n"], 3)
        self.assertAlmostEqual(row["mean_test_acc"], 0.3)
        self.assertAlmostEqual(row["std_test_acc"], 0.1)

    def test_paired_delta_is_computed_within_seed(self):
        runs = pd.DataFrame(
            [
                {
                    "initialization": "scratch",
                    "k": 20,
                    "train_seed": seed,
                    "method": method,
                    "test_acc": value,
                }
                for seed, values in enumerate(
                    ((0.2, 0.3, 0.4), (0.4, 0.5, 0.7), (0.6, 0.7, 0.8))
                )
                for method, value in zip(("none", "cutmix", "simcutmix"), values)
            ]
        )
        paired = paired_deltas(runs)
        rows = paired[paired["comparison"] == "simcutmix_vs_cutmix"]
        self.assertEqual(list(rows["delta_test_pp"].round(6)), [10.0, 20.0, 10.0])
        self.assertAlmostEqual(rows.iloc[0]["mean_delta_test_pp"], 40.0 / 3.0)
        self.assertEqual(rows.iloc[0]["seed_wins"], 3)


if __name__ == "__main__":
    unittest.main()
