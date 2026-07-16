import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.augmentations.similarity_guided import apply_simcutmix, simcutmix_rand_bbox
from src.data.indexed_dataset import GuidedPairDataset
from src.train import ExperimentConfig, experiment_name, train_one_epoch


class FixedRng:
    def __init__(self, lam=0.25, random_value=0.0, integers=(4, 4)):
        self.lam = lam
        self.random_value = random_value
        self._integers = list(integers)
        self._integer_index = 0

    def beta(self, alpha_a, alpha_b):
        return self.lam

    def random(self):
        return self.random_value

    def integers(self, high):
        value = self._integers[self._integer_index]
        self._integer_index += 1
        return value % high


class TinyImageDataset(Dataset):
    def __init__(self):
        self.labels = {10: 0, 11: 0, 12: 0, 20: 1, 21: 1, 22: 1}

    def __len__(self):
        return 30

    def __getitem__(self, index):
        image = torch.full((1, 8, 8), float(index))
        return image, self.labels[int(index)]


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 2),
        )

    def forward(self, images):
        return self.net(images)


def class_aware_neighbors():
    return {
        "mode": "class_aware",
        "original_indices": torch.tensor([10, 11, 12, 20, 21, 22]),
        "neighbor_indices": torch.tensor(
            [
                [11, 12],
                [10, 12],
                [11, 10],
                [21, 22],
                [20, 22],
                [21, 20],
            ]
        ),
    }


def class_agnostic_neighbors():
    return {
        "mode": "class_agnostic",
        "original_indices": torch.tensor([10, 11, 12, 20, 21, 22]),
        "neighbor_indices": torch.tensor(
            [
                [20, 11],
                [21, 10],
                [22, 11],
                [10, 21],
                [11, 20],
                [12, 21],
            ]
        ),
    }


def paired_batch(dataset):
    rows = [dataset[position] for position in range(len(dataset))]
    images_i = torch.stack([row[0] for row in rows])
    targets_i = torch.tensor([row[1] for row in rows])
    images_j = torch.stack([row[2] for row in rows])
    targets_j = torch.tensor([row[3] for row in rows])
    return images_i, targets_i, images_j, targets_j


class SimCutMixTests(unittest.TestCase):
    def setUp(self):
        self.images_i = torch.zeros(3, 1, 8, 8)
        self.images_j = torch.ones(3, 1, 8, 8)
        self.targets_i = torch.tensor([0, 1, 0])
        self.targets_j = torch.tensor([1, 0, 1])

    def test_bbox_coordinates_are_valid(self):
        bbx1, bby1, bbx2, bby2 = simcutmix_rand_bbox(
            self.images_i.size(),
            lam=0.25,
            rng=np.random.default_rng(0),
        )

        self.assertGreaterEqual(bbx1, 0)
        self.assertGreaterEqual(bby1, 0)
        self.assertLessEqual(bbx2, self.images_i.size(-1))
        self.assertLessEqual(bby2, self.images_i.size(-2))
        self.assertLessEqual(bbx1, bbx2)
        self.assertLessEqual(bby1, bby2)

    def test_pasted_area_changes_the_image(self):
        mixed, *_ = apply_simcutmix(
            self.images_i,
            self.targets_i,
            self.images_j,
            self.targets_j,
            rng=FixedRng(lam=0.25, integers=(4, 4)),
        )

        self.assertGreater(int((mixed != self.images_i).sum().item()), 0)

    def test_adjusted_lambda_matches_patch_area(self):
        mixed, _, _, lam = apply_simcutmix(
            self.images_i,
            self.targets_i,
            self.images_j,
            self.targets_j,
            rng=FixedRng(lam=0.25, integers=(4, 4)),
        )

        changed_area = int((mixed[0, 0] != self.images_i[0, 0]).sum().item())
        image_area = self.images_i.size(-1) * self.images_i.size(-2)
        expected_lam = 1.0 - changed_area / image_area

        self.assertEqual(lam, expected_lam)

    def test_sample_mix_probability_selects_individual_rows(self):
        mixed, _, _, lam = apply_simcutmix(
            self.images_i,
            self.targets_i,
            self.images_j,
            self.targets_j,
            sample_mix_prob=torch.tensor([1.0, 0.0, 1.0]),
            rng=FixedRng(lam=0.25, integers=(4, 4)),
        )

        self.assertTrue(torch.is_tensor(lam))
        self.assertLess(float(lam[0].item()), 1.0)
        self.assertEqual(float(lam[1].item()), 1.0)
        self.assertLess(float(lam[2].item()), 1.0)
        self.assertGreater(int((mixed[0] != self.images_i[0]).sum().item()), 0)
        self.assertEqual(int((mixed[1] != self.images_i[1]).sum().item()), 0)
        self.assertGreater(int((mixed[2] != self.images_i[2]).sum().item()), 0)

    def test_labels_match_selected_neighbor_pairs(self):
        _, targets_i, targets_j, _ = apply_simcutmix(
            self.images_i,
            self.targets_i,
            self.images_j,
            self.targets_j,
            rng=FixedRng(lam=0.25, integers=(4, 4)),
        )

        self.assertTrue(torch.equal(targets_i, self.targets_i))
        self.assertTrue(torch.equal(targets_j, self.targets_j))

    def test_fixed_random_seed_gives_same_boxes(self):
        first, _, _, first_lam = apply_simcutmix(
            self.images_i,
            self.targets_i,
            self.images_j,
            self.targets_j,
            rng=np.random.default_rng(123),
        )
        second, _, _, second_lam = apply_simcutmix(
            self.images_i,
            self.targets_i,
            self.images_j,
            self.targets_j,
            rng=np.random.default_rng(123),
        )

        self.assertEqual(first_lam, second_lam)
        self.assertTrue(torch.equal(first, second))

    def test_mix_prob_zero_gives_original_image(self):
        mixed, targets_i, targets_j, lam = apply_simcutmix(
            self.images_i,
            self.targets_i,
            self.images_j,
            self.targets_j,
            mix_prob=0.0,
            rng=FixedRng(lam=0.25, integers=(4, 4)),
        )

        self.assertEqual(lam, 1.0)
        self.assertTrue(torch.equal(mixed, self.images_i))
        self.assertTrue(torch.equal(targets_i, self.targets_i))
        self.assertTrue(torch.equal(targets_j, self.targets_i))

    def test_class_aware_simcutmix_keeps_matching_labels(self):
        dataset = GuidedPairDataset(
            TinyImageDataset(),
            train_indices=[10, 11, 12, 20, 21, 22],
            neighbor_index=class_aware_neighbors(),
            mode="class_aware",
            seed=0,
        )
        images_i, targets_i, images_j, targets_j = paired_batch(dataset)

        _, targets_a, targets_b, _ = apply_simcutmix(
            images_i,
            targets_i,
            images_j,
            targets_j,
            rng=FixedRng(lam=0.25, integers=(4, 4)),
        )

        self.assertTrue(torch.equal(targets_a, targets_b))

    def test_class_agnostic_simcutmix_allows_mismatched_labels(self):
        dataset = GuidedPairDataset(
            TinyImageDataset(),
            train_indices=[10, 11, 12, 20, 21, 22],
            neighbor_index=class_agnostic_neighbors(),
            mode="class_agnostic",
            seed=0,
        )
        images_i, targets_i, images_j, targets_j = paired_batch(dataset)

        _, targets_a, targets_b, _ = apply_simcutmix(
            images_i,
            targets_i,
            images_j,
            targets_j,
            rng=FixedRng(lam=0.25, integers=(4, 4)),
        )

        self.assertTrue(torch.any(targets_a != targets_b).item())

    def test_training_loss_is_not_nan(self):
        dataset = GuidedPairDataset(
            TinyImageDataset(),
            train_indices=[10, 11, 12, 20],
            neighbor_index={
                "mode": "class_agnostic",
                "original_indices": torch.tensor([10, 11, 12, 20]),
                "neighbor_indices": torch.tensor([[20], [10], [11], [12]]),
            },
            mode="class_agnostic",
            seed=0,
        )
        loader = DataLoader(dataset, batch_size=2)
        model = TinyClassifier()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        loss, accuracy = train_one_epoch(
            model=model,
            dataloader=loader,
            criterion=criterion,
            optimizer=optimizer,
            device=torch.device("cpu"),
            augmentation="simcutmix",
            mixup_alpha=1.0,
            mix_prob=1.0,
            augmentation_rng=FixedRng(lam=0.25, integers=(4, 4, 4, 4)),
        )

        self.assertTrue(np.isfinite(loss))
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_experiment_name_separates_simcutmix_variants(self):
        base = dict(
            dataset="cifar100",
            model="resnet50",
            k=100,
            subset_seed=0,
            augmentation="simcutmix",
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
            anchor_score_path=None,
            anchor_selection="top_fraction",
            anchor_top_pct=0.2,
            anchor_score_power=1.0,
        )
        class_aware = ExperimentConfig(**base)
        class_agnostic = ExperimentConfig(**{**base, "guided_mode": "class_agnostic"})

        self.assertNotEqual(experiment_name(class_aware), experiment_name(class_agnostic))
        self.assertIn("simcutmix", experiment_name(class_aware))
        self.assertIn("class_aware", experiment_name(class_aware))


if __name__ == "__main__":
    unittest.main()
