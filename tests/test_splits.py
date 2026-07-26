import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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


if __name__ == "__main__":
    unittest.main()
