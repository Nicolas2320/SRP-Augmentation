"""Audit canonical experiment records and local large artifacts.

The audit is read-only unless ``--json-output`` is supplied. It answers three
maintenance questions:

1. Do canonical summaries have their expected metrics and referenced files?
2. Which local ``.pt`` files are connected to summaries or metadata?
3. Which checkpoints require a manual retention decision?

The tool never deletes, moves, or rewrites experiment artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator


SUMMARY_REFERENCE_FIELDS = (
    "metrics_path",
    "best_model_path",
    "neighbor_path",
    "anchor_score_path",
)

REVIEW_CATEGORIES = {
    "incomplete_run_checkpoint",
    "run_checkpoint_not_referenced",
    "unreferenced_shared_checkpoint",
    "unclassified_pt",
}


def resolve_recorded_path(repo_root: Path, value: str) -> Path:
    """Resolve a repository path recorded with Windows or POSIX separators."""
    normalized = str(value).replace("\\", os.sep).replace("/", os.sep)
    path = Path(normalized)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def repo_relative(path: Path, repo_root: Path) -> str:
    """Return a stable forward-slash path when a file is inside the repository."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def iter_pt_references(value: Any) -> Iterator[str]:
    """Yield every nested string value that names a PyTorch payload."""
    if isinstance(value, dict):
        for child in value.values():
            yield from iter_pt_references(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_pt_references(child)
    elif isinstance(value, str) and value.lower().endswith((".pt", ".pth")):
        yield value


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object and give the caller a useful path on failure."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def classify_pt_artifact(
    path: Path,
    results_root: Path,
    summary_references: set[Path],
    metadata_references: set[Path],
) -> str:
    """Classify one local tensor/checkpoint artifact by its known references."""
    resolved = path.resolve()
    if resolved in summary_references:
        return "referenced_by_summary"
    if resolved in metadata_references:
        return "referenced_by_metadata"

    relative_parts = path.relative_to(results_root).parts
    if path.name == "checkpoint_best.pt":
        if (path.parent / "summary.json").exists():
            return "run_checkpoint_not_referenced"
        return "incomplete_run_checkpoint"
    if relative_parts[:2] == ("shared", "checkpoints"):
        return "unreferenced_shared_checkpoint"
    if relative_parts[:2] == ("shared", "neighbors"):
        return "unreferenced_neighbor_support"
    return "unclassified_pt"


def build_audit(repo_root: Path, results_root: Path) -> dict[str, Any]:
    """Build a serializable audit report without modifying the repository."""
    repo_root = repo_root.resolve()
    if not results_root.is_absolute():
        results_root = repo_root / results_root
    results_root = results_root.resolve()

    summary_paths = sorted(results_root.rglob("summary.json"))
    summary_references: dict[Path, list[dict[str, str]]] = defaultdict(list)
    missing_references: list[dict[str, str]] = []
    missing_metric_pairs: list[str] = []

    for summary_path in summary_paths:
        summary = load_json(summary_path)
        summary_relative = repo_relative(summary_path, repo_root)

        sibling_metrics = summary_path.parent / "metrics.csv"
        if not sibling_metrics.exists():
            missing_metric_pairs.append(summary_relative)

        for field in SUMMARY_REFERENCE_FIELDS:
            value = summary.get(field)
            if not value:
                continue
            target = resolve_recorded_path(repo_root, str(value))
            summary_references[target].append(
                {"summary": summary_relative, "field": field}
            )
            if not target.exists():
                missing_references.append(
                    {
                        "summary": summary_relative,
                        "field": field,
                        "path": repo_relative(target, repo_root),
                    }
                )

    metadata_references: dict[Path, list[str]] = defaultdict(list)
    missing_metadata_references: list[dict[str, str]] = []
    for metadata_path in sorted(results_root.rglob("metadata.json")):
        metadata = load_json(metadata_path)
        for value in iter_pt_references(metadata):
            target = resolve_recorded_path(repo_root, value)
            metadata_relative = repo_relative(metadata_path, repo_root)
            metadata_references[target].append(metadata_relative)
            if not target.exists():
                missing_metadata_references.append(
                    {
                        "metadata": metadata_relative,
                        "path": repo_relative(target, repo_root),
                    }
                )

    category_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"count": 0, "size_bytes": 0}
    )
    pt_artifacts: list[dict[str, Any]] = []
    retention_review: list[dict[str, Any]] = []

    summary_reference_set = set(summary_references)
    metadata_reference_set = set(metadata_references)

    for path in sorted(
        [
            *results_root.rglob("*.pt"),
            *results_root.rglob("*.pth"),
        ]
    ):
        size_bytes = path.stat().st_size
        category = classify_pt_artifact(
            path=path,
            results_root=results_root,
            summary_references=summary_reference_set,
            metadata_references=metadata_reference_set,
        )
        category_stats[category]["count"] += 1
        category_stats[category]["size_bytes"] += size_bytes

        item = {
            "path": repo_relative(path, repo_root),
            "size_bytes": size_bytes,
            "category": category,
            "summary_reference_count": len(summary_references.get(path.resolve(), [])),
            "metadata_reference_count": len(metadata_references.get(path.resolve(), [])),
        }
        pt_artifacts.append(item)
        if category in REVIEW_CATEGORIES:
            retention_review.append(item)

    outside_canonical_files: list[dict[str, Any]] = []
    results_parent = results_root.parent
    if results_parent.exists():
        for sibling in sorted(results_parent.iterdir()):
            if sibling.resolve() == results_root:
                continue
            if sibling.is_dir():
                files = [path for path in sibling.rglob("*") if path.is_file()]
                if files:
                    outside_canonical_files.append(
                        {
                            "path": repo_relative(sibling, repo_root),
                            "file_count": len(files),
                            "size_bytes": sum(path.stat().st_size for path in files),
                        }
                    )
            elif sibling.is_file():
                outside_canonical_files.append(
                    {
                        "path": repo_relative(sibling, repo_root),
                        "file_count": 1,
                        "size_bytes": sibling.stat().st_size,
                    }
                )

    return {
        "repo_root": str(repo_root),
        "results_root": repo_relative(results_root, repo_root),
        "summary_records": {
            "count": len(summary_paths),
            "complete_metric_pairs": len(summary_paths) - len(missing_metric_pairs),
            "missing_metric_pairs": missing_metric_pairs,
        },
        "references": {
            "missing_count": (
                len(missing_references) + len(missing_metadata_references)
            ),
            "summary_missing_count": len(missing_references),
            "summary_missing": missing_references,
            "metadata_missing_count": len(missing_metadata_references),
            "metadata_missing": missing_metadata_references,
        },
        "pt_artifacts": {
            "count": len(pt_artifacts),
            "size_bytes": sum(item["size_bytes"] for item in pt_artifacts),
            "categories": dict(sorted(category_stats.items())),
            "items": pt_artifacts,
        },
        "retention_review": sorted(
            retention_review,
            key=lambda item: (-item["size_bytes"], item["path"]),
        ),
        "outside_canonical": outside_canonical_files,
    }


def format_size(size_bytes: int) -> str:
    """Format byte counts using binary units."""
    value = float(size_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def print_report(report: dict[str, Any], details: bool = False) -> None:
    """Print a concise human-readable audit."""
    summaries = report["summary_records"]
    references = report["references"]
    artifacts = report["pt_artifacts"]

    print("Results artifact audit")
    print(f"  canonical summaries: {summaries['count']}")
    print(f"  complete summary/metrics pairs: {summaries['complete_metric_pairs']}")
    print(f"  missing recorded references: {references['missing_count']}")
    print(
        f"  local .pt/.pth artifacts: {artifacts['count']} "
        f"({format_size(artifacts['size_bytes'])})"
    )
    print(f"  retention-review candidates: {len(report['retention_review'])}")
    print(f"  noncanonical result locations: {len(report['outside_canonical'])}")

    print("\nArtifact categories:")
    for category, values in artifacts["categories"].items():
        print(
            f"  {category}: {values['count']} "
            f"({format_size(values['size_bytes'])})"
        )

    if references["summary_missing"]:
        print("\nMissing summary references:")
        for item in references["summary_missing"]:
            print(
                f"  {item['field']}: {item['path']} "
                f"(from {item['summary']})"
            )

    if references["metadata_missing"]:
        print("\nMissing metadata references:")
        for item in references["metadata_missing"]:
            print(f"  {item['path']} (from {item['metadata']})")

    if report["outside_canonical"]:
        print("\nNoncanonical result locations:")
        for item in report["outside_canonical"]:
            print(
                f"  {item['path']}: {item['file_count']} files "
                f"({format_size(item['size_bytes'])})"
            )

    if details and report["retention_review"]:
        print("\nRetention-review candidates:")
        for item in report["retention_review"]:
            print(
                f"  {item['category']}: {item['path']} "
                f"({format_size(item['size_bytes'])})"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit experiment records and local artifacts without deleting them."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/experiments"),
        help="Canonical experiment root.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path for the full machine-readable report.",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Print every artifact that requires a retention decision.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    report = build_audit(repo_root=repo_root, results_root=args.results_root)
    print_report(report, details=args.details)

    if args.json_output is not None:
        output_path = args.json_output
        if not output_path.is_absolute():
            output_path = repo_root / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote audit JSON: {repo_relative(output_path, repo_root)}")


if __name__ == "__main__":
    main()
