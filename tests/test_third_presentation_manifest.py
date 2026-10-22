import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.build_third_presentation_manifest import (
    build_rows,
    expected_settings,
)


class ThirdPresentationManifestTests(unittest.TestCase):
    def test_guided_screening_profiles_are_method_specific(self):
        simmixup = expected_settings("pretrained", 100, "simmixup")
        simcutmix = expected_settings("pretrained", 100, "simcutmix")

        self.assertEqual((simmixup["neighbor_rank_start"], simmixup["neighbor_k"]), (1, 10))
        self.assertEqual((simcutmix["neighbor_rank_start"], simcutmix["neighbor_k"]), (21, 20))

    def test_manifest_has_one_row_per_phase1_cell_and_reuses_exact_match(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            results = root / "results"
            summary_path = results / "exact" / "summary.json"
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text(
                json.dumps(expected_settings("pretrained", 20, "none")),
                encoding="utf-8",
            )
            rows = build_rows(results, root / "manifest.csv")

        self.assertEqual(len(rows), 60)
        exact = next(row for row in rows if row["job_key"] == "phase1-pretrained-k20-none-s0-t0")
        self.assertEqual(exact["candidate_classification"], "reusable")
        self.assertEqual(exact["status"], "complete")


if __name__ == "__main__":
    unittest.main()
