"""Build a CSV index for saved experiment outputs.

The manifest is non-destructive: it reads existing summary JSON and CSV files
under ``results/`` and writes ``results/experiments/manifest.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


MANIFEST_COLUMNS = [
    "experiment_id",
    "collection",
    "dataset",
    "model",
    "k",
    "subset_seed",
    "train_seed",
    "augmentation",
    "epochs",
    "best_epoch",
    "train_acc_at_best",
    "best_val_acc",
    "test_acc",
    "gap_train_val",
    "test_loss",
    "guided_mode",
    "neighbor_file",
    "neighbor_k",
    "neighbor_rank_window",
    "pair_sampling",
    "mixup_alpha",
    "mix_prob",
    "mix_warmup_epochs",
    "anchor_selection",
    "anchor_top_pct",
    "anchor_score_power",
    "summary_path",
    "metrics_path",
    "best_model_path",
]


def iter_summary_paths(results_root: Path) -> list[Path]:
    """Find both canonical and legacy experiment summary files."""
    paths = set(results_root.rglob("*_summary.json"))
    paths.update(results_root.rglob("summary.json"))
    return sorted(paths)


def collection_name(path: Path, results_root: Path) -> str:
    rel = path.relative_to(results_root)
    parts = rel.parts
    if len(parts) >= 6 and parts[0] == "experiments":
        if parts[4] == "legacy" and len(parts) >= 7:
            return "/".join(parts[1:6])
        return "/".join(parts[1:5])
    if len(parts) >= 2 and parts[0].startswith("experiments_"):
        return parts[0]
    if len(parts) >= 2 and parts[0] == "final_stage":
        return "final_stage"
    return parts[0]


def read_best_epoch_row(csv_path: Path, best_epoch: int) -> dict[str, str]:
    if not csv_path.exists():
        return {}

    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                epoch = int(float(row.get("epoch", "")))
            except ValueError:
                continue
            if epoch == best_epoch:
                return row
    return {}


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt_float(value: Any) -> str:
    number = as_float(value)
    if number is None:
        return ""
    return f"{number:.6g}"


def optional_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def neighbor_file(summary: dict[str, Any]) -> str:
    path = summary.get("neighbor_path")
    if not path:
        return ""
    return Path(str(path).replace("\\", "/")).name


def rank_window(summary: dict[str, Any]) -> str:
    if summary.get("augmentation") not in {"simmixup", "simcutmix"}:
        return ""
    start = summary.get("neighbor_rank_start", 1)
    count = summary.get("neighbor_k")
    if count is None:
        return ""
    start_i = int(start)
    return f"{start_i}-{start_i + int(count) - 1}"


def resolve_metrics_path(summary_path: Path, summary: dict[str, Any], repo_root: Path) -> Path:
    metrics_path = summary.get("metrics_path")
    if metrics_path:
        candidate = repo_root / str(metrics_path).replace("\\", "/")
        if candidate.exists():
            return candidate
    if summary_path.name == "summary.json":
        return summary_path.with_name("metrics.csv")
    return summary_path.with_name(summary_path.name.replace("_summary.json", ".csv"))


def experiment_id(summary_path: Path, results_root: Path) -> str:
    if summary_path.name == "summary.json":
        rel = summary_path.parent.relative_to(results_root)
        if rel.parts and rel.parts[0] == "experiments":
            rel = Path(*rel.parts[1:])
        return rel.as_posix()
    return summary_path.name.removesuffix("_summary.json")


def build_rows(results_root: Path, repo_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for summary_path in iter_summary_paths(results_root):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics_path = resolve_metrics_path(summary_path, summary, repo_root)
        best_epoch = int(summary.get("best_epoch", 0))
        best_row = read_best_epoch_row(metrics_path, best_epoch)

        train_acc = as_float(best_row.get("train_acc"))
        val_acc = as_float(summary.get("best_val_acc"))
        gap = train_acc - val_acc if train_acc is not None and val_acc is not None else None

        is_guided = summary.get("augmentation") in {"simmixup", "simcutmix"}
        has_anchor = bool(summary.get("anchor_score_path"))

        row = {
            "experiment_id": experiment_id(summary_path, results_root),
            "collection": collection_name(summary_path, results_root),
            "dataset": str(summary.get("dataset", "")),
            "model": str(summary.get("model", "")),
            "k": str(summary.get("k", "")),
            "subset_seed": str(summary.get("subset_seed", "")),
            "train_seed": str(summary.get("train_seed", "")),
            "augmentation": str(summary.get("augmentation", "")),
            "epochs": str(summary.get("epochs", "")),
            "best_epoch": str(summary.get("best_epoch", "")),
            "train_acc_at_best": fmt_float(train_acc),
            "best_val_acc": fmt_float(summary.get("best_val_acc")),
            "test_acc": fmt_float(summary.get("test_acc_best_checkpoint")),
            "gap_train_val": fmt_float(gap),
            "test_loss": fmt_float(summary.get("test_loss_best_checkpoint")),
            "guided_mode": optional_str(summary.get("guided_mode")) if is_guided else "",
            "neighbor_file": neighbor_file(summary) if is_guided else "",
            "neighbor_k": optional_str(summary.get("neighbor_k")) if is_guided else "",
            "neighbor_rank_window": rank_window(summary) if is_guided else "",
            "pair_sampling": optional_str(summary.get("pair_sampling")) if is_guided else "",
            "mixup_alpha": fmt_float(summary.get("mixup_alpha")),
            "mix_prob": fmt_float(summary.get("mix_prob")) if is_guided else "",
            "mix_warmup_epochs": optional_str(summary.get("mix_warmup_epochs")) if is_guided else "",
            "anchor_selection": optional_str(summary.get("anchor_selection")) if has_anchor else "",
            "anchor_top_pct": fmt_float(summary.get("anchor_top_pct")) if has_anchor else "",
            "anchor_score_power": fmt_float(summary.get("anchor_score_power")) if has_anchor else "",
            "summary_path": repo_relative(summary_path, repo_root),
            "metrics_path": repo_relative(metrics_path, repo_root),
            "best_model_path": str(summary.get("best_model_path") or "").replace("\\", "/"),
        }
        rows.append(row)

    return rows


def write_outputs(rows: list[dict[str, str]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_csv = output_dir / "manifest.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the central experiment manifest.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/experiments"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    rows = build_rows(args.results_root, repo_root)
    write_outputs(rows, args.output_dir)
    print(f"Wrote {len(rows)} experiments to {args.output_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
