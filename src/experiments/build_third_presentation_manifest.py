"""Build the auditable Phase-1 CIFAR-100 job manifest.

The manifest has one row per intended seed-0 screening cell. Existing results
are reused only when every material setting matches the current protocol.
Operational fields are preserved when rebuilding an existing manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path
from typing import Any


INITIALIZATIONS = ("scratch", "pretrained")
K_VALUES = (20, 50, 100, 200, 300, 450)
METHODS = ("none", "mixup", "cutmix", "simmixup", "simcutmix")
FIELDS = (
    "job_key",
    "phase",
    "initialization",
    "k",
    "method",
    "subset_seed",
    "train_seed",
    "status",
    "candidate_classification",
    "candidate_summary",
    "incompatibilities",
    "submission_command",
    "slurm_job_id",
    "last_checked",
    "notes",
)


def expected_settings(initialization: str, k: int, method: str) -> dict[str, Any]:
    large = k == 450
    settings: dict[str, Any] = {
        "dataset": "cifar100",
        "model": "resnet50",
        "pretrained": initialization == "pretrained",
        "k": k,
        "subset_seed": 0,
        "train_seed": 0,
        "augmentation": method,
        "epochs": 50 if large else 100,
        "batch_size": 32,
        "optimizer": "sgd",
        "lr": 0.1 if large else 0.01,
        "momentum": 0.9,
        "nesterov": True,
        "weight_decay": 0.0005,
        "lr_milestones": [15, 30, 40] if large else [30, 55, 75],
        "lr_gamma": 0.2 if large else 0.1,
    }
    if method == "mixup":
        settings["mixup_alpha"] = 1.0
    if method == "cutmix":
        settings.update({"cutmix_alpha": 1.0, "cutmix_prob": 0.5})
    if method in {"simmixup", "simcutmix"}:
        settings.update(
            {
                "guided_mode": "class_agnostic",
                "neighbor_k": 10 if method == "simmixup" else 20,
                "neighbor_rank_start": 1 if method == "simmixup" else 21,
                "pair_sampling": "uniform",
                "mix_prob": 1.0,
                "mix_warmup_epochs": 0,
            }
        )
        settings["mixup_alpha" if method == "simmixup" else "cutmix_alpha"] = 1.0
    return settings


def normalized(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    return value


def mismatch_reasons(summary: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    reasons = []
    provenance = summary.get("provenance") or {}
    if provenance.get("collection") == "historical_from_scratch":
        reasons.append("historical architecture/recipe cohort")
    for key, expected_value in expected.items():
        actual = normalized(summary.get(key))
        if actual != normalized(expected_value):
            reasons.append(f"{key}={actual!r} expected {expected_value!r}")
    return reasons


def summaries(results_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    records = []
    for path in sorted(results_root.rglob("summary.json")):
        try:
            records.append((path, json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError):
            continue
    return records


def candidate_records(
    records: list[tuple[Path, dict[str, Any]]],
    expected: dict[str, Any],
) -> list[tuple[Path, dict[str, Any]]]:
    keys = ("dataset", "model", "pretrained", "k", "subset_seed", "train_seed", "augmentation")
    return [
        (path, summary)
        for path, summary in records
        if all(normalized(summary.get(key)) == normalized(expected[key]) for key in keys)
    ]


def load_mutable_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["job_key"]: row for row in csv.DictReader(handle)}


def build_rows(results_root: Path, output_path: Path) -> list[dict[str, str]]:
    records = summaries(results_root)
    previous = load_mutable_rows(output_path)
    rows: list[dict[str, str]] = []

    for initialization in INITIALIZATIONS:
        for k in K_VALUES:
            for method in METHODS:
                expected = expected_settings(initialization, k, method)
                candidates = candidate_records(records, expected)
                compatible: tuple[Path, dict[str, Any]] | None = None
                candidate_path = ""
                incompatibilities = ""
                classification = "missing"
                status = "missing"

                for path, summary in candidates:
                    reasons = mismatch_reasons(summary, expected)
                    if not reasons:
                        compatible = (path, summary)
                        break

                if compatible:
                    candidate_path = compatible[0].as_posix()
                    classification = "reusable"
                    status = "complete"
                elif candidates:
                    candidate_path = candidates[0][0].as_posix()
                    reasons = mismatch_reasons(candidates[0][1], expected)
                    incompatibilities = "; ".join(reasons)
                    classification = "incompatible"
                    status = "excluded-incompatible"

                job_key = f"phase1-{initialization}-k{k}-{method}-s0-t0"
                old = previous.get(job_key, {})
                if classification != "reusable" and old.get("status") in {
                    "queued", "running", "complete", "failed"
                }:
                    status = old["status"]

                row = {
                    "job_key": job_key,
                    "phase": "phase1-screening",
                    "initialization": initialization,
                    "k": str(k),
                    "method": method,
                    "subset_seed": "0",
                    "train_seed": "0",
                    "status": status,
                    "candidate_classification": classification,
                    "candidate_summary": candidate_path,
                    "incompatibilities": incompatibilities,
                    "submission_command": (
                        "sbatch scripts/run_resnet_comparison.slurm "
                        f"{initialization} {k} {method} screen 0 0"
                    ),
                    "slurm_job_id": old.get("slurm_job_id", ""),
                    "last_checked": old.get("last_checked", str(date.today())),
                    "notes": old.get("notes", ""),
                }
                rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/manifests/third_presentation_phase1.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows(args.results_root, args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[str, int] = {}
    for row in rows:
        key = row["candidate_classification"]
        counts[key] = counts.get(key, 0) + 1
    print(f"Wrote {len(rows)} Phase-1 cells to {args.output}")
    print("Candidate audit: " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))


if __name__ == "__main__":
    main()
