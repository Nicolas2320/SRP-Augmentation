import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.graphs.plot_graphs import (
    aggregate_runs,
    build_all_baseline_comparisons,
    load_summary_metrics,
    spread_label_positions,
)


def write_run(
    experiments_dir: Path,
    run_name: str,
    *,
    augmentation: str,
    epochs: int = 100,
    test_acc: float,
) -> None:
    run_dir = experiments_dir / run_name
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.csv").write_text(
        "epoch,train_acc,val_acc\n1,0.10,0.20\n2,0.20,0.30\n",
        encoding="utf-8",
    )
    summary = {
        "dataset": "cifar100",
        "model": "resnet50",
        "k": 100,
        "subset_seed": 0,
        "train_seed": 0,
        "augmentation": augmentation,
        "epochs": epochs,
        "batch_size": 64,
        "lr": 0.01,
        "weight_decay": 0.0005,
        "optimizer": "sgd",
        "momentum": 0.9,
        "nesterov": True,
        "lr_milestones": [30, 55, 75],
        "lr_gamma": 0.1,
        "num_train": 10000,
        "num_val": 5000,
        "num_test": 10000,
        "best_epoch": 2,
        "best_val_acc": 0.30,
        "test_acc_best_checkpoint": test_acc,
        # Deliberately stale: the loader should prefer the canonical sibling CSV.
        "metrics_path": "results/experiments/old/location/metrics.csv",
        "mixup_alpha": 1.0,
        "cutmix_prob": 0.5,
    }
    if augmentation == "simcutmix":
        summary.update(
            {
                "guided_mode": "class_agnostic",
                "neighbor_k": 20,
                "neighbor_rank_start": 21,
                "pair_sampling": "uniform",
                "mix_prob": 1.0,
                "mix_warmup_epochs": 0,
                "anchor_selection": "top_fraction",
                "anchor_top_pct": 0.2,
                "anchor_score_power": 1.0,
                "dynamic_neighbor_pool": False,
            }
        )
    (run_dir / "summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )


class PlotGraphDataTests(unittest.TestCase):
    def test_loader_and_matching_use_recipe_not_augmentation_parameters(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            experiments_dir = Path(temp_dir) / "results" / "experiments"
            write_run(
                experiments_dir,
                "baseline",
                augmentation="cutmix",
                test_acc=0.43,
            )
            write_run(
                experiments_dir,
                "baseline_none",
                augmentation="none",
                test_acc=0.35,
            )
            write_run(
                experiments_dir,
                "proposal_matched",
                augmentation="simcutmix",
                test_acc=0.46,
            )
            write_run(
                experiments_dir,
                "proposal_different_budget",
                augmentation="simcutmix",
                epochs=50,
                test_acc=0.44,
            )

            runs = load_summary_metrics(experiments_dir)
            comparisons = build_all_baseline_comparisons(runs)

            self.assertTrue(runs["metrics_exists"].all())
            self.assertEqual(len(comparisons), 2)
            self.assertEqual(set(comparisons["baseline_label"]), {"CutMix", "No augmentation"})
            cutmix = comparisons[comparisons["baseline_label"] == "CutMix"].iloc[0]
            self.assertAlmostEqual(cutmix["delta_test_pp"], 3.0)
            self.assertEqual(cutmix["proposal_label"], "SimCutMix")
            self.assertEqual(cutmix["baseline_best_epoch"], 2)
            self.assertEqual(cutmix["proposal_best_epoch"], 2)

    def test_single_runs_keep_uncertainty_missing(self):
        runs = pd.DataFrame(
            [
                {
                    "dataset": "cifar100",
                    "model": "resnet50",
                    "k": 100,
                    "augmentation": "cutmix",
                    "method_name": "CutMix",
                    "series_key": "cutmix|{}",
                    "series_label": "CutMix",
                    "test_acc": 0.43,
                    "best_val_acc": 0.44,
                    "run_id": "only-run",
                    "epochs": 100,
                }
            ]
        )

        aggregated = aggregate_runs(runs)

        self.assertEqual(aggregated.iloc[0]["runs"], 1)
        self.assertTrue(pd.isna(aggregated.iloc[0]["std_test_acc"]))

    def test_label_spreading_preserves_order_and_minimum_gap(self):
        values = [71.1, 69.0, 72.7, 71.6]
        positions = spread_label_positions(
            values,
            minimum_gap=3.0,
            lower=1.0,
            upper=89.0,
        )

        ordered = [
            position
            for _, position in sorted(zip(values, positions), key=lambda pair: pair[0])
        ]
        self.assertTrue(
            all(
                right - left >= 3.0
                for left, right in zip(ordered, ordered[1:])
            )
        )
        self.assertEqual(
            sorted(range(len(values)), key=lambda index: values[index]),
            sorted(range(len(positions)), key=lambda index: positions[index]),
        )


if __name__ == "__main__":
    unittest.main()
