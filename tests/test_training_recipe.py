import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train import (
    ExperimentConfig,
    build_lr_scheduler,
    build_optimizer,
    experiment_config_id,
    experiment_output_dir,
    format_neighbor_rank_summary,
    get_transforms,
)


def training_config(**overrides):
    values = {
        "dataset": "cifar100",
        "model": "resnet50",
        "k": 450,
        "subset_seed": 0,
        "augmentation": "cutmix",
        "mixup_alpha": 1.0,
        "epochs": 100,
        "batch_size": 128,
        "lr": 0.1,
        "weight_decay": 5e-4,
        "train_seed": 0,
        "data_root": "data/raw",
        "split_root": "data/splits",
        "output_root": "results/experiments",
        "num_workers": 0,
        "neighbor_path": None,
        "guided_mode": "class_agnostic",
        "neighbor_k": 20,
        "neighbor_rank_start": 1,
        "pair_sampling": "uniform",
        "mix_prob": 1.0,
        "mix_warmup_epochs": 0,
        "anchor_score_path": None,
        "anchor_selection": "top_fraction",
        "anchor_top_pct": 0.2,
        "anchor_score_power": 1.0,
    }
    values.update(overrides)
    return ExperimentConfig(**values)


class TrainingRecipeTests(unittest.TestCase):
    def test_every_training_pipeline_starts_with_crop_and_flip(self):
        for augmentation in (
            "none",
            "mixup",
            "cutmix",
            "augmix",
            "simmixup",
            "simcutmix",
        ):
            with self.subTest(augmentation=augmentation):
                train_transform, eval_transform = get_transforms(
                    "cifar100",
                    augmentation=augmentation,
                    seed=0,
                )
                train_names = [type(step).__name__ for step in train_transform.transforms]
                eval_names = [type(step).__name__ for step in eval_transform.transforms]
                self.assertEqual(train_names[:2], ["RandomCrop", "RandomHorizontalFlip"])
                self.assertEqual(eval_names, ["ToTensor", "Normalize"])

    def test_optimizer_is_sgd_with_nesterov(self):
        config = training_config()
        optimizer = build_optimizer(config, nn.Linear(2, 2))

        self.assertIsInstance(optimizer, torch.optim.SGD)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.1)
        self.assertEqual(optimizer.param_groups[0]["momentum"], 0.9)
        self.assertTrue(optimizer.param_groups[0]["nesterov"])
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 5e-4)

    def test_scheduler_scales_200_epoch_milestones_to_100_epochs(self):
        config = training_config()
        model = nn.Linear(2, 2)
        optimizer = build_optimizer(config, model)
        scheduler = build_lr_scheduler(config, optimizer)
        lrs_used = []

        for _epoch in range(1, 82):
            lrs_used.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()

        self.assertAlmostEqual(lrs_used[0], 0.1)
        self.assertAlmostEqual(lrs_used[29], 0.1)
        self.assertAlmostEqual(lrs_used[30], 0.02)
        self.assertAlmostEqual(lrs_used[60], 0.004)
        self.assertAlmostEqual(lrs_used[80], 0.0008)

    def test_output_folder_is_short_readable_and_collision_safe(self):
        config = training_config()
        output_dir = experiment_output_dir(config)

        self.assertEqual(output_dir.parts[-2], "cutmix")
        self.assertEqual(
            output_dir.parts[-1],
            f"e100_s0_t0_c{experiment_config_id(config)}",
        )
        self.assertNotIn("standard_cifar_recipe", output_dir.parts)
        self.assertNotIn("scheduled_training", output_dir.parts)
        self.assertLess(len(output_dir.as_posix()), 100)

    def test_config_id_changes_when_hidden_recipe_setting_changes(self):
        original = training_config()
        different_lr = training_config(lr=0.02)

        self.assertNotEqual(
            experiment_config_id(original),
            experiment_config_id(different_lr),
        )

    def test_neighbor_rank_log_omits_per_rank_statistics(self):
        class GuidedDataset:
            neighbor_indices = torch.empty((4, 20), dtype=torch.long)
            neighbor_rank_start = 21

            @staticmethod
            def sampled_neighbor_rank_counts():
                raise AssertionError("rank counts should not be computed for logging")

        summary = format_neighbor_rank_summary(GuidedDataset(), effective_mix_prob=1.0)

        self.assertEqual(
            summary,
            " | neighbor_k=20 | neighbor_rank_window=21-40",
        )
        self.assertNotIn("mean=", summary)
        self.assertNotIn("rank_counts=", summary)


if __name__ == "__main__":
    unittest.main()
