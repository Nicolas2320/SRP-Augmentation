import unittest

from src.experiments.analyze_pair_ablation import (
    strategy_from_summary,
    validate_summary,
)


def guided_summary(**overrides):
    summary = {
        "dataset": "cifar100",
        "model": "resnet50",
        "pretrained": False,
        "k": 100,
        "subset_seed": 0,
        "train_seed": 0,
        "augmentation": "simcutmix",
        "epochs": 100,
        "batch_size": 32,
        "optimizer": "sgd",
        "lr": 0.01,
        "momentum": 0.9,
        "nesterov": True,
        "weight_decay": 0.0005,
        "lr_milestones": [30, 55, 75],
        "lr_gamma": 0.1,
        "cutmix_alpha": 1.0,
        "cutmix_prob": 0.5,
        "pair_sampling": "uniform",
        "mix_prob": 0.5,
        "mix_warmup_epochs": 0,
        "guided_mode": "class_agnostic",
        "neighbor_k": 20,
        "neighbor_rank_start": 21,
        "neighbor_path": "neighbors.pt",
        "best_val_acc": 0.5,
        "test_evaluated": False,
    }
    summary.update(overrides)
    return summary


class PairAblationAnalysisTests(unittest.TestCase):
    def test_strategy_mapping_distinguishes_all_guided_conditions(self):
        cases = (
            ({"guided_mode": "class_aware", "neighbor_k": 10, "neighbor_rank_start": 1}, "same_class"),
            ({"guided_mode": "class_agnostic", "neighbor_k": 10, "neighbor_rank_start": 1}, "most_similar"),
            ({"guided_mode": "class_agnostic", "neighbor_k": 10, "neighbor_rank_start": 11}, "middle_similarity"),
            ({"guided_mode": "class_agnostic", "neighbor_k": 20, "neighbor_rank_start": 21}, "less_similar"),
            ({"guided_mode": "different_label", "neighbor_k": 10, "neighbor_rank_start": 1}, "different_label"),
        )
        for overrides, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(strategy_from_summary(guided_summary(**overrides)), expected)

    def test_validation_accepts_frozen_less_similar_condition(self):
        validate_summary(guided_summary(), "scratch", 100, "less_similar")

    def test_validation_rejects_mixing_probability_confounds(self):
        with self.assertRaisesRegex(ValueError, "mix_prob"):
            validate_summary(
                guided_summary(mix_prob=1.0),
                "scratch",
                100,
                "less_similar",
            )

    def test_unknown_rank_window_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown guided strategy"):
            strategy_from_summary(guided_summary(neighbor_rank_start=5))


if __name__ == "__main__":
    unittest.main()
