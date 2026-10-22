import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.audit_splits import audit_fixed_split
from src.data.make_splits import create_kshot_from_pool, max_k_per_class


class SplitGenerationTests(unittest.TestCase):
    def setUp(self):
        self.targets = [0] * 6 + [1] * 6 + [2] * 6
        self.val_indices = {0, 1, 6, 7, 12, 13}
        self.train_pool_indices = [
            index for index in range(len(self.targets)) if index not in self.val_indices
        ]

    def test_max_k_uses_every_available_training_example_per_class(self):
        max_k = max_k_per_class(self.targets, self.train_pool_indices)
        train_indices = create_kshot_from_pool(
            self.targets,
            self.train_pool_indices,
            k=max_k,
            seed=0,
        )

        self.assertEqual(max_k, 4)
        self.assertEqual(train_indices, sorted(self.train_pool_indices))
        self.assertTrue(set(train_indices).isdisjoint(self.val_indices))

    def test_max_k_uses_smallest_class_when_pool_is_unbalanced(self):
        self.assertEqual(max_k_per_class(self.targets, self.train_pool_indices[:-1]), 3)

    def test_max_k_rejects_empty_pool(self):
        with self.assertRaises(ValueError):
            max_k_per_class(self.targets, [])

    def test_same_seed_produces_nested_k_shot_subsets(self):
        smaller = set(create_kshot_from_pool(self.targets, self.train_pool_indices, 2, 7))
        larger = set(create_kshot_from_pool(self.targets, self.train_pool_indices, 3, 7))

        self.assertTrue(smaller.issubset(larger))

    def test_fixed_split_audit_requires_exact_partition_and_balance(self):
        fixed = {
            "dataset": "synthetic",
            "val_per_class": 2,
            "num_val": 6,
            "num_train_pool": 12,
            "val_indices": sorted(self.val_indices),
            "train_pool_indices": self.train_pool_indices,
        }

        train_pool, validation = audit_fixed_split(fixed, self.targets, "synthetic")

        self.assertEqual(train_pool, set(self.train_pool_indices))
        self.assertEqual(validation, self.val_indices)

        fixed["train_pool_indices"] = self.train_pool_indices[:-1]
        with self.assertRaisesRegex(AssertionError, "partition"):
            audit_fixed_split(fixed, self.targets, "synthetic")


if __name__ == "__main__":
    unittest.main()
