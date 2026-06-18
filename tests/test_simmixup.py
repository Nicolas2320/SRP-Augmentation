import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.augmentations.similarity_guided import apply_simmixup, simmixup_criterion
from src.train import ExperimentConfig, experiment_name, load_neighbor_payload, train_one_epoch


class FixedRng:
    def __init__(self, lam=0.25, random_value=0.0):
        self.lam = lam
        self.random_value = random_value

    def beta(self, alpha_a, alpha_b):
        return self.lam

    def random(self):
        return self.random_value


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4, 2),
        )

    def forward(self, images):
        return self.net(images)


class SimMixUpTests(unittest.TestCase):
    def setUp(self):
        self.images_i = torch.zeros(3, 1, 2, 2)
        self.images_j = torch.ones(3, 1, 2, 2)
        self.targets_i = torch.tensor([0, 1, 0])
        self.targets_j = torch.tensor([1, 0, 1])

    def test_fixed_lambda_mixes_artificial_tensors(self):
        mixed, targets_i, targets_j, lam = apply_simmixup(
            self.images_i,
            self.targets_i,
            self.images_j,
            self.targets_j,
            alpha=1.0,
            rng=FixedRng(lam=0.25),
        )

        self.assertEqual(lam, 0.25)
        self.assertTrue(torch.allclose(mixed, torch.full_like(mixed, 0.75)))
        self.assertTrue(torch.equal(targets_i, self.targets_i))
        self.assertTrue(torch.equal(targets_j, self.targets_j))

    def test_lambda_one_gives_original_image(self):
        mixed, *_ = apply_simmixup(
            self.images_i,
            self.targets_i,
            self.images_j,
            self.targets_j,
            rng=FixedRng(lam=1.0),
        )

        self.assertTrue(torch.equal(mixed, self.images_i))

    def test_lambda_zero_gives_partner_image(self):
        mixed, *_ = apply_simmixup(
            self.images_i,
            self.targets_i,
            self.images_j,
            self.targets_j,
            rng=FixedRng(lam=0.0),
        )

        self.assertTrue(torch.equal(mixed, self.images_j))

    def test_alpha_non_positive_disables_mixing(self):
        mixed, targets_i, targets_j, lam = apply_simmixup(
            self.images_i,
            self.targets_i,
            self.images_j,
            self.targets_j,
            alpha=0.0,
            rng=FixedRng(lam=0.25),
        )

        self.assertEqual(lam, 1.0)
        self.assertTrue(torch.equal(mixed, self.images_i))
        self.assertTrue(torch.equal(targets_i, self.targets_i))
        self.assertTrue(torch.equal(targets_j, self.targets_i))

    def test_mix_prob_zero_disables_mixing(self):
        mixed, targets_i, targets_j, lam = apply_simmixup(
            self.images_i,
            self.targets_i,
            self.images_j,
            self.targets_j,
            mix_prob=0.0,
            rng=FixedRng(lam=0.25),
        )

        self.assertEqual(lam, 1.0)
        self.assertTrue(torch.equal(mixed, self.images_i))
        self.assertTrue(torch.equal(targets_i, self.targets_i))
        self.assertTrue(torch.equal(targets_j, self.targets_i))

    def test_mixed_labels_are_correct(self):
        _, targets_i, targets_j, _ = apply_simmixup(
            self.images_i,
            self.targets_i,
            self.images_j,
            self.targets_j,
            rng=FixedRng(lam=0.25),
        )

        self.assertTrue(torch.equal(targets_i, self.targets_i))
        self.assertTrue(torch.equal(targets_j, self.targets_j))

    def test_loss_equals_weighted_cross_entropy_terms(self):
        logits = torch.tensor(
            [
                [2.0, 0.5],
                [0.1, 1.7],
                [1.2, 0.8],
            ],
            dtype=torch.float32,
        )
        criterion = nn.CrossEntropyLoss()
        lam = 0.25

        loss = simmixup_criterion(
            criterion=criterion,
            predictions=logits,
            targets_i=self.targets_i,
            targets_j=self.targets_j,
            lam=lam,
        )
        expected = lam * criterion(logits, self.targets_i) + (1.0 - lam) * criterion(
            logits,
            self.targets_j,
        )

        self.assertTrue(torch.allclose(loss, expected))

    def test_batch_shape_remains_bchw(self):
        mixed, *_ = apply_simmixup(
            self.images_i,
            self.targets_i,
            self.images_j,
            self.targets_j,
            rng=FixedRng(lam=0.25),
        )

        self.assertEqual(tuple(mixed.shape), (3, 1, 2, 2))

    def test_training_loss_is_not_nan(self):
        images_i = torch.zeros(4, 1, 2, 2)
        targets_i = torch.tensor([0, 1, 0, 1])
        images_j = torch.ones(4, 1, 2, 2)
        targets_j = torch.tensor([1, 0, 1, 0])
        idx_i = torch.arange(4)
        idx_j = torch.arange(10, 14)
        loader = DataLoader(
            TensorDataset(images_i, targets_i, images_j, targets_j, idx_i, idx_j),
            batch_size=2,
        )
        model = TinyClassifier()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        loss, accuracy = train_one_epoch(
            model=model,
            dataloader=loader,
            criterion=criterion,
            optimizer=optimizer,
            device=torch.device("cpu"),
            augmentation="simmixup",
            mixup_alpha=1.0,
            mix_prob=1.0,
            augmentation_rng=FixedRng(lam=0.25),
        )

        self.assertTrue(np.isfinite(loss))
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_experiment_name_separates_simmixup_variants(self):
        base = dict(
            dataset="cifar100",
            model="resnet50",
            k=100,
            subset_seed=0,
            augmentation="simmixup",
            mixup_alpha=1.0,
            epochs=50,
            batch_size=64,
            lr=1e-3,
            weight_decay=1e-4,
            train_seed=0,
            data_root="data/raw",
            split_root="data/splits",
            output_root="results",
            num_workers=0,
            neighbor_path="neighbors.pt",
            guided_mode="class_aware",
            neighbor_k=10,
            neighbor_rank_start=1,
            pair_sampling="uniform",
            mix_prob=1.0,
            mix_warmup_epochs=0,
        )
        class_aware = ExperimentConfig(**base)
        class_agnostic = ExperimentConfig(**{**base, "guided_mode": "class_agnostic"})
        lower_mix_prob = ExperimentConfig(**{**base, "mix_prob": 0.5})
        later_rank_window = ExperimentConfig(**{**base, "neighbor_rank_start": 21})

        self.assertNotEqual(experiment_name(class_aware), experiment_name(class_agnostic))
        self.assertNotEqual(experiment_name(class_aware), experiment_name(lower_mix_prob))
        self.assertNotEqual(experiment_name(class_aware), experiment_name(later_rank_window))
        self.assertIn("class_aware", experiment_name(class_aware))
        self.assertIn("r1-10", experiment_name(class_aware))
        self.assertIn("r21-30", experiment_name(later_rank_window))
        self.assertIn("mp0p5", experiment_name(lower_mix_prob))

    def test_load_neighbor_payload_selects_rank_window(self):
        payload = {
            "mode": "class_agnostic",
            "original_indices": torch.tensor([0, 1]),
            "neighbor_indices": torch.tensor(
                [
                    [10, 11, 12, 13],
                    [20, 21, 22, 23],
                ]
            ),
            "similarities": torch.tensor(
                [
                    [0.9, 0.8, 0.7, 0.6],
                    [0.5, 0.4, 0.3, 0.2],
                ]
            ),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            neighbor_path = Path(temp_dir) / "neighbors.pt"
            torch.save(payload, neighbor_path)

            loaded = load_neighbor_payload(
                path=str(neighbor_path),
                guided_mode="class_agnostic",
                neighbor_k=2,
                neighbor_rank_start=3,
            )

        self.assertTrue(torch.equal(loaded["neighbor_indices"], torch.tensor([[12, 13], [22, 23]])))
        self.assertTrue(
            torch.equal(
                loaded["similarities"],
                torch.tensor([[0.7, 0.6], [0.3, 0.2]]),
            )
        )
        self.assertEqual(loaded["num_neighbors"], 2)
        self.assertEqual(loaded["neighbor_rank_start"], 3)
        self.assertEqual(loaded["neighbor_rank_end"], 4)


if __name__ == "__main__":
    unittest.main()
