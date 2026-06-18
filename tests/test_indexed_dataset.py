import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.augmentations.cutmix import CutMix
from src.data.indexed_dataset import GuidedPairDataset, IndexedDataset
from src.train import train_one_epoch


class SyntheticIndexedDataset(Dataset):
    def __init__(self):
        self.labels = {10: 0, 11: 0, 12: 0, 20: 1, 21: 1, 22: 1}

    def __len__(self):
        return 30

    def __getitem__(self, index):
        return torch.tensor([float(index)]), self.labels[int(index)]


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4, 2),
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


class IndexedDatasetTests(unittest.TestCase):
    def setUp(self):
        self.base_dataset = SyntheticIndexedDataset()
        self.train_indices = [10, 11, 12, 20, 21, 22]

    def test_indexed_dataset_returns_global_index(self):
        dataset = IndexedDataset(self.base_dataset, self.train_indices)

        image, label, global_index = dataset[0]

        self.assertEqual(len(dataset), len(self.train_indices))
        self.assertEqual(float(image.item()), 10.0)
        self.assertEqual(label, 0)
        self.assertEqual(global_index, 10)

    def test_guided_pair_dataset_length_equals_num_train(self):
        dataset = GuidedPairDataset(
            self.base_dataset,
            self.train_indices,
            class_aware_neighbors(),
            mode="class_aware",
            seed=0,
        )

        self.assertEqual(len(dataset), len(self.train_indices))

    def test_anchor_index_is_always_from_kshot_subset(self):
        dataset = GuidedPairDataset(
            self.base_dataset,
            self.train_indices,
            class_aware_neighbors(),
            mode="class_aware",
            seed=0,
        )

        anchor_indices = [dataset[position][4] for position in range(len(dataset))]

        self.assertTrue(set(anchor_indices).issubset(set(self.train_indices)))

    def test_partner_index_is_always_saved_neighbor(self):
        neighbors = class_aware_neighbors()
        dataset = GuidedPairDataset(
            self.base_dataset,
            self.train_indices,
            neighbors,
            mode="class_aware",
            seed=0,
        )
        neighbor_map = {
            int(idx): set(row.tolist())
            for idx, row in zip(neighbors["original_indices"], neighbors["neighbor_indices"])
        }

        for position in range(len(dataset)):
            *_, idx_i, idx_j = dataset[position]
            self.assertIn(idx_j, neighbor_map[idx_i])

    def test_class_aware_pairs_have_same_label(self):
        dataset = GuidedPairDataset(
            self.base_dataset,
            self.train_indices,
            class_aware_neighbors(),
            mode="class_aware",
            seed=0,
        )

        for position in range(len(dataset)):
            _, label_i, _, label_j, _, _ = dataset[position]
            self.assertEqual(label_i, label_j)

    def test_class_agnostic_pairs_can_cross_labels(self):
        dataset = GuidedPairDataset(
            self.base_dataset,
            self.train_indices,
            class_agnostic_neighbors(),
            mode="class_agnostic",
            seed=0,
        )

        labels = [(dataset[position][1], dataset[position][3]) for position in range(len(dataset))]

        self.assertTrue(any(label_i != label_j for label_i, label_j in labels))

    def test_same_seed_gives_same_partner_sequence(self):
        first = GuidedPairDataset(
            self.base_dataset,
            self.train_indices,
            class_aware_neighbors(),
            mode="class_aware",
            seed=0,
        )
        second = GuidedPairDataset(
            self.base_dataset,
            self.train_indices,
            class_aware_neighbors(),
            mode="class_aware",
            seed=0,
        )

        first_partners = [first[position][5] for position in range(len(first))]
        second_partners = [second[position][5] for position in range(len(second))]

        self.assertEqual(first_partners, second_partners)

    def test_different_seed_changes_partner_sequence(self):
        first = GuidedPairDataset(
            self.base_dataset,
            self.train_indices,
            class_aware_neighbors(),
            mode="class_aware",
            seed=0,
        )
        second = GuidedPairDataset(
            self.base_dataset,
            self.train_indices,
            class_aware_neighbors(),
            mode="class_aware",
            seed=1,
        )

        first_partners = [first[position][5] for position in range(len(first))]
        second_partners = [second[position][5] for position in range(len(second))]

        self.assertNotEqual(first_partners, second_partners)

    def test_sampled_neighbor_rank_counts_cover_current_epoch_pairs(self):
        dataset = GuidedPairDataset(
            self.base_dataset,
            self.train_indices,
            class_aware_neighbors(),
            mode="class_aware",
            seed=0,
        )

        counts = dataset.sampled_neighbor_rank_counts()

        self.assertEqual(counts.numel(), 2)
        self.assertEqual(int(counts.sum().item()), len(self.train_indices))
        self.assertTrue(torch.all(counts >= 0))


class ExistingTrainingShapeTests(unittest.TestCase):
    def test_existing_augmentation_modes_still_train_on_two_item_batches(self):
        images = torch.randn(4, 1, 2, 2)
        targets = torch.tensor([0, 1, 0, 1])
        device = torch.device("cpu")

        for augmentation in ("none", "mixup", "cutmix", "augmix"):
            model = TinyClassifier().to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            loader = DataLoader(TensorDataset(images.clone(), targets.clone()), batch_size=2)
            cutmix = CutMix(alpha=1.0, seed=0) if augmentation == "cutmix" else None

            loss, accuracy = train_one_epoch(
                model=model,
                dataloader=loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                augmentation=augmentation,
                cutmix=cutmix,
                mixup_alpha=1.0,
                augmentation_rng=np.random.default_rng(0),
            )

            self.assertTrue(np.isfinite(loss))
            self.assertGreaterEqual(accuracy, 0.0)
            self.assertLessEqual(accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
