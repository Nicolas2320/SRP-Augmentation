import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments.audit_artifacts import build_audit


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class ArtifactAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo_root = Path(self.temp_dir.name)
        self.results_root = self.repo_root / "results" / "experiments"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_audit_classifies_references_and_review_candidates(self):
        complete_run = self.results_root / "cifar100" / "run_complete"
        complete_run.mkdir(parents=True)
        (complete_run / "metrics.csv").write_text("epoch,val_acc\n1,0.5\n")
        (complete_run / "checkpoint_best.pt").write_bytes(b"checkpoint")
        write_json(
            complete_run / "summary.json",
            {
                "metrics_path": "results\\experiments\\cifar100\\run_complete\\metrics.csv",
                "best_model_path": (
                    "results\\experiments\\cifar100\\run_complete\\checkpoint_best.pt"
                ),
            },
        )

        neighbor_root = self.results_root / "shared" / "neighbors" / "fixture"
        neighbor_root.mkdir(parents=True)
        (neighbor_root / "embeddings.pt").write_bytes(b"embeddings")
        (neighbor_root / "unused_neighbors.pt").write_bytes(b"neighbors")
        write_json(
            neighbor_root / "metadata.json",
            {
                "embedding_path": (
                    "results/experiments/shared/neighbors/fixture/embeddings.pt"
                )
            },
        )

        shared_checkpoint = (
            self.results_root / "shared" / "checkpoints" / "historical.pt"
        )
        shared_checkpoint.parent.mkdir(parents=True)
        shared_checkpoint.write_bytes(b"historical")

        incomplete_checkpoint = (
            self.results_root / "cifar100" / "incomplete" / "checkpoint_best.pt"
        )
        incomplete_checkpoint.parent.mkdir(parents=True)
        incomplete_checkpoint.write_bytes(b"incomplete")

        outside = self.repo_root / "results" / "experiments_v2" / "old.csv"
        outside.parent.mkdir(parents=True)
        outside.write_text("legacy")

        report = build_audit(
            repo_root=self.repo_root,
            results_root=self.results_root,
        )

        self.assertEqual(report["summary_records"]["count"], 1)
        self.assertEqual(report["summary_records"]["complete_metric_pairs"], 1)
        self.assertEqual(report["references"]["missing_count"], 0)
        self.assertEqual(report["references"]["metadata_missing_count"], 0)

        categories = report["pt_artifacts"]["categories"]
        self.assertEqual(categories["referenced_by_summary"]["count"], 1)
        self.assertEqual(categories["referenced_by_metadata"]["count"], 1)
        self.assertEqual(categories["unreferenced_neighbor_support"]["count"], 1)
        self.assertEqual(categories["unreferenced_shared_checkpoint"]["count"], 1)
        self.assertEqual(categories["incomplete_run_checkpoint"]["count"], 1)

        review_categories = {
            item["category"] for item in report["retention_review"]
        }
        self.assertEqual(
            review_categories,
            {
                "unreferenced_shared_checkpoint",
                "incomplete_run_checkpoint",
            },
        )
        self.assertEqual(report["outside_canonical"][0]["path"], "results/experiments_v2")

    def test_audit_reports_missing_metric_and_checkpoint_paths(self):
        run = self.results_root / "cifar100" / "missing"
        write_json(
            run / "summary.json",
            {
                "metrics_path": "results/experiments/cifar100/missing/metrics.csv",
                "best_model_path": (
                    "results/experiments/cifar100/missing/checkpoint_best.pt"
                ),
            },
        )

        report = build_audit(
            repo_root=self.repo_root,
            results_root=Path("results/experiments"),
        )

        self.assertEqual(report["summary_records"]["count"], 1)
        self.assertEqual(report["summary_records"]["complete_metric_pairs"], 0)
        self.assertEqual(report["references"]["missing_count"], 2)
        self.assertEqual(
            {item["field"] for item in report["references"]["summary_missing"]},
            {"metrics_path", "best_model_path"},
        )

    def test_audit_reports_missing_metadata_payloads(self):
        metadata = self.results_root / "shared" / "neighbors" / "metadata.json"
        write_json(
            metadata,
            {
                "embedding_path": (
                    "results/experiments/shared/neighbors/missing_embeddings.pt"
                )
            },
        )

        report = build_audit(
            repo_root=self.repo_root,
            results_root=self.results_root,
        )

        self.assertEqual(report["references"]["missing_count"], 1)
        self.assertEqual(report["references"]["summary_missing_count"], 0)
        self.assertEqual(report["references"]["metadata_missing_count"], 1)
        self.assertEqual(
            report["references"]["metadata_missing"][0]["path"],
            "results/experiments/shared/neighbors/missing_embeddings.pt",
        )


if __name__ == "__main__":
    unittest.main()
