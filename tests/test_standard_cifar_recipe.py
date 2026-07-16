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
    experiment_output_dir,
    get_transforms,
)


def standard_config(**overrides):
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


class StandardCifarRecipeTests(unittest.TestCase):
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

    def test_standard_optimizer_is_sgd_with_nesterov(self):
        config = standard_config()
        optimizer = build_optimizer(config, nn.Linear(2, 2))

        self.assertIsInstance(optimizer, torch.optim.SGD)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.1)
        self.assertEqual(optimizer.param_groups[0]["momentum"], 0.9)
        self.assertTrue(optimizer.param_groups[0]["nesterov"])
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 5e-4)

    def test_scheduler_scales_200_epoch_milestones_to_100_epochs(self):
        config = standard_config()
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

    def test_standard_recipe_has_separate_output_folder(self):
        output_dir = experiment_output_dir(standard_config())
        self.assertIn("standard_cifar_recipe", output_dir.parts)
        self.assertTrue(any("bs128" in part for part in output_dir.parts))
        self.assertTrue(any("cp0p5" in part for part in output_dir.parts))


if __name__ == "__main__":
    unittest.main()
