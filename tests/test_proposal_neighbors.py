import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.proposal.build_neighbors import build_neighbors, effective_neighbor_count
from src.proposal.inspect_neighbors import validate_embeddings, validate_neighbors


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        digest.update(f.read())
    return digest.hexdigest()


class ProposalNeighborTests(unittest.TestCase):
    def setUp(self):
        self.embeddings = torch.tensor(
            [
                [1.00, 0.00, 0.00],
                [0.95, 0.05, 0.00],
                [0.90, 0.10, 0.00],
                [0.00, 1.00, 0.00],
                [0.05, 0.95, 0.00],
                [0.10, 0.90, 0.00],
            ],
            dtype=torch.float32,
        )
        self.labels = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
        self.indices = torch.tensor([10, 11, 12, 20, 21, 22], dtype=torch.long)

    def test_class_aware_neighbors_respect_labels_and_are_sorted(self):
        payload = build_neighbors(
            self.embeddings,
            self.labels,
            self.indices,
            mode="class_aware",
            max_neighbors=5,
        )

        self.assertEqual(payload["num_neighbors"], 2)
        self.assertFalse((payload["neighbor_indices"] == self.indices[:, None]).any())
        self.assertTrue((payload["neighbor_labels"] == self.labels[:, None]).all())
        self.assertTrue((payload["similarities"][:, :-1] >= payload["similarities"][:, 1:]).all())
        validate_neighbors(payload, "class_aware")

    def test_class_agnostic_neighbors_can_cross_labels(self):
        payload = build_neighbors(
            self.embeddings,
            self.labels,
            self.indices,
            mode="class_agnostic",
            max_neighbors=5,
        )

        self.assertEqual(payload["num_neighbors"], 5)
        self.assertTrue((payload["neighbor_labels"] != self.labels[:, None]).any())
        self.assertFalse((payload["neighbor_indices"] == self.indices[:, None]).any())
        self.assertTrue((payload["similarities"][:, :-1] >= payload["similarities"][:, 1:]).all())
        validate_neighbors(payload, "class_agnostic")

    def test_different_label_neighbors_exclude_same_label_pairs(self):
        payload = build_neighbors(
            self.embeddings,
            self.labels,
            self.indices,
            mode="different_label",
            max_neighbors=5,
        )

        self.assertEqual(payload["num_neighbors"], 3)
        self.assertTrue((payload["neighbor_labels"] != self.labels[:, None]).all())
        self.assertFalse((payload["neighbor_indices"] == self.indices[:, None]).any())
        self.assertTrue((payload["similarities"][:, :-1] >= payload["similarities"][:, 1:]).all())
        validate_neighbors(payload, "different_label")

    def test_class_aware_neighbor_count_is_capped_at_k_minus_one(self):
        self.assertEqual(effective_neighbor_count(self.labels, "class_aware", 10), 2)
        self.assertEqual(effective_neighbor_count(self.labels, "class_aware", 1), 1)
        self.assertEqual(effective_neighbor_count(self.labels, "class_agnostic", 10), 5)
        self.assertEqual(effective_neighbor_count(self.labels, "different_label", 10), 3)

    def test_embedding_validation_checks_split_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            split_path = tmp_path / "k3_seed0.json"
            val_path = tmp_path / "fixed_validation_split.json"
            train_split = {
                "dataset": "synthetic",
                "k": 3,
                "subset_seed": 0,
                "num_train": 6,
                "train_indices": self.indices.tolist(),
            }
            val_split = {
                "dataset": "synthetic",
                "num_val": 2,
                "val_indices": [99, 100],
            }
            write_json(split_path, train_split)
            write_json(val_path, val_split)

            metadata = {
                "dataset": "synthetic",
                "k": 3,
                "subset_seed": 0,
                "encoder": "synthetic_encoder",
                "split_hash": sha256_file(split_path),
                "num_train": 6,
                "embedding_dim": 3,
                "source_dataset_partition": "torchvision_train_true",
                "uses_test_samples": False,
            }
            embedding_payload = {
                "embeddings": self.embeddings,
                "original_indices": self.indices,
            }

            messages = validate_embeddings(
                embedding_payload,
                metadata,
                train_split,
                val_split,
                sha256_file(split_path),
            )
            self.assertTrue(any("embeddings shape OK" in message for message in messages))

    def test_embedding_validation_rejects_validation_overlap(self):
        train_split = {
            "num_train": 6,
            "train_indices": self.indices.tolist(),
        }
        val_split = {
            "val_indices": [10],
        }
        metadata = {
            "dataset": "synthetic",
            "k": 3,
            "subset_seed": 0,
            "encoder": "synthetic_encoder",
            "split_hash": "hash",
            "num_train": 6,
            "embedding_dim": 3,
            "source_dataset_partition": "torchvision_train_true",
            "uses_test_samples": False,
        }
        embedding_payload = {
            "embeddings": self.embeddings,
            "original_indices": self.indices,
        }

        with self.assertRaises(AssertionError):
            validate_embeddings(embedding_payload, metadata, train_split, val_split, "hash")


if __name__ == "__main__":
    unittest.main()
