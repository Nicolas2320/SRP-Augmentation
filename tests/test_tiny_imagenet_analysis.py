import unittest

from src.experiments.analyze_tiny_imagenet import (
    expected_subset_seed,
    validate_summary,
)


def canonical_summary(**overrides):
    summary = {
        "dataset": "tinyimagenet",
        "model": "resnet50",
        "pretrained": False,
        "k": 20,
        "subset_seed": 1,
        "train_seed": 1,
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
        "num_train": 4000,
        "num_val": 10000,
        "num_test": 10000,
        "test_evaluated": True,
        "cutmix_alpha": 1.0,
        "cutmix_prob": 0.5,
        "guided_mode": "class_agnostic",
        "neighbor_k": 20,
        "neighbor_rank_start": 21,
        "pair_sampling": "uniform",
        "mix_prob": 1.0,
        "mix_warmup_epochs": 0,
        "best_val_acc": 0.2,
        "validation_acc_recomputed": 0.2,
        "test_acc_best_checkpoint": 0.19,
        "test_loss_best_checkpoint": 2.0,
    }
    summary.update(overrides)
    return summary


class TinyImageNetAnalysisTests(unittest.TestCase):
    def test_subset_seed_policy(self):
        self.assertEqual(expected_subset_seed(20, 2), 2)
        self.assertEqual(expected_subset_seed(450, 2), 0)

    def test_frozen_summary_validation_accepts_canonical_configuration(self):
        validate_summary(canonical_summary(), "scratch", 20, "simcutmix", 1)

    def test_frozen_summary_validation_rejects_test_or_validation_mismatch(self):
        with self.assertRaisesRegex(ValueError, "test_evaluated"):
            validate_summary(
                canonical_summary(test_evaluated=False),
                "scratch",
                20,
                "simcutmix",
                1,
            )
        with self.assertRaisesRegex(ValueError, "recomputed validation"):
            validate_summary(
                canonical_summary(validation_acc_recomputed=0.21),
                "scratch",
                20,
                "simcutmix",
                1,
            )


if __name__ == "__main__":
    unittest.main()
