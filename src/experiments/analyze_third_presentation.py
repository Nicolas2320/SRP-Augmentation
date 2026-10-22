"""Build the final three-seed CIFAR-100 tables and figures.

Seed-0 runs are selected through the audited Phase-1 manifest. Seeds 1 and 2
come from the final confirmation tree. This avoids accidentally aggregating
historical duplicates or non-frozen screening configurations.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INITIALIZATIONS = ("scratch", "pretrained")
K_VALUES = (20, 50, 100, 200, 300, 450)
METHODS = ("none", "cutmix", "simcutmix")
DISPLAY_NAMES = {
    "none": "Crop+Flip",
    "cutmix": "CutMix",
    "simcutmix": "SimCutMix",
}
COLORS = {"none": "#5F6368", "cutmix": "#F58518", "simcutmix": "#E45756"}
MARKERS = {"none": "o", "cutmix": "^", "simcutmix": "D"}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def expected_subset_seed(k: int, train_seed: int) -> int:
    return 0 if k == 450 else train_seed


def validate_summary(
    summary: dict[str, Any], initialization: str, k: int, method: str, train_seed: int
) -> None:
    epochs = 50 if k == 450 else 100
    expected = {
        "dataset": "cifar100",
        "model": "resnet50",
        "pretrained": initialization == "pretrained",
        "k": k,
        "subset_seed": expected_subset_seed(k, train_seed),
        "train_seed": train_seed,
        "augmentation": method,
        "epochs": epochs,
        "batch_size": 32,
        "optimizer": "sgd",
        "lr": 0.1 if k == 450 else 0.01,
        "momentum": 0.9,
        "nesterov": True,
        "weight_decay": 0.0005,
        "lr_milestones": [15, 30, 40] if k == 450 else [30, 55, 75],
        "lr_gamma": 0.2 if k == 450 else 0.1,
        "test_evaluated": True,
    }
    if method in {"cutmix", "simcutmix"}:
        expected.update({"cutmix_alpha": 1.0, "cutmix_prob": 0.5})
    if method == "simcutmix":
        expected.update(
            {
                "guided_mode": "class_agnostic",
                "neighbor_k": 20,
                "neighbor_rank_start": 21,
                "pair_sampling": "uniform",
                "mix_prob": 1.0,
                "mix_warmup_epochs": 0,
            }
        )

    mismatches = [
        f"{key}={summary.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if summary.get(key) != value
    ]
    for key in ("best_val_acc", "test_acc_best_checkpoint", "test_loss_best_checkpoint"):
        if not isinstance(summary.get(key), (int, float)):
            mismatches.append(f"{key} is missing or non-numeric")
    if mismatches:
        raise ValueError("; ".join(mismatches))


def seed_zero_paths(manifest_path: Path) -> dict[tuple[str, int, str, int], Path]:
    selected: dict[tuple[str, int, str, int], Path] = {}
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["method"] not in METHODS:
                continue
            if row["candidate_classification"] != "reusable":
                raise ValueError(f"Seed-0 cell is not reusable: {row['job_key']}")
            key = (row["initialization"], int(row["k"]), row["method"], 0)
            selected[key] = Path(row["candidate_summary"])
    return selected


def find_final_summary(
    final_root: Path, initialization: str, k: int, method: str, train_seed: int
) -> Path:
    subset_seed = expected_subset_seed(k, train_seed)
    base = final_root / initialization / "cifar100" / "resnet50" / f"k{k}" / method
    matches = []
    for path in base.rglob("summary.json"):
        summary = load_json(path)
        if (
            summary.get("subset_seed") == subset_seed
            and summary.get("train_seed") == train_seed
        ):
            matches.append(path)
    if len(matches) != 1:
        raise ValueError(
            f"Expected one final summary for {initialization} k={k} {method} "
            f"seed={train_seed}; found {len(matches)}"
        )
    return matches[0]


def load_confirmation_runs(manifest_path: Path, final_root: Path) -> pd.DataFrame:
    seed_zero = seed_zero_paths(manifest_path)
    records = []
    for initialization in INITIALIZATIONS:
        for k in K_VALUES:
            for method in METHODS:
                for train_seed in (0, 1, 2):
                    key = (initialization, k, method, train_seed)
                    path = (
                        seed_zero[key]
                        if train_seed == 0
                        else find_final_summary(
                            final_root, initialization, k, method, train_seed
                        )
                    )
                    summary = load_json(path)
                    try:
                        validate_summary(summary, initialization, k, method, train_seed)
                    except ValueError as error:
                        raise ValueError(f"{path}: {error}") from error
                    records.append(
                        {
                            "initialization": initialization,
                            "k": k,
                            "method": method,
                            "subset_seed": summary["subset_seed"],
                            "train_seed": train_seed,
                            "best_val_acc": summary["best_val_acc"],
                            "test_acc": summary["test_acc_best_checkpoint"],
                            "best_epoch": summary["best_epoch"],
                            "summary_path": path.as_posix(),
                        }
                    )
    runs = pd.DataFrame.from_records(records)
    if len(runs) != 108:
        raise ValueError(f"Expected 108 confirmation cells; found {len(runs)}")
    return runs


def aggregate_runs(runs: pd.DataFrame) -> pd.DataFrame:
    return (
        runs.groupby(["initialization", "k", "method"], sort=False)
        .agg(
            n=("test_acc", "size"),
            mean_val_acc=("best_val_acc", "mean"),
            std_val_acc=("best_val_acc", "std"),
            mean_test_acc=("test_acc", "mean"),
            std_test_acc=("test_acc", "std"),
        )
        .reset_index()
    )


def paired_deltas(runs: pd.DataFrame) -> pd.DataFrame:
    wide = runs.pivot(
        index=["initialization", "k", "train_seed"], columns="method", values="test_acc"
    ).reset_index()
    records = []
    for comparison, proposal, baseline in (
        ("simcutmix_vs_none", "simcutmix", "none"),
        ("simcutmix_vs_cutmix", "simcutmix", "cutmix"),
        ("cutmix_vs_none", "cutmix", "none"),
    ):
        for row in wide.itertuples(index=False):
            records.append(
                {
                    "initialization": row.initialization,
                    "k": row.k,
                    "train_seed": row.train_seed,
                    "comparison": comparison,
                    "delta_test_pp": 100.0 * (getattr(row, proposal) - getattr(row, baseline)),
                }
            )
    paired = pd.DataFrame.from_records(records)
    aggregate = (
        paired.groupby(["initialization", "k", "comparison"], sort=False)
        .agg(
            n=("delta_test_pp", "size"),
            mean_delta_test_pp=("delta_test_pp", "mean"),
            std_delta_test_pp=("delta_test_pp", "std"),
            seed_wins=("delta_test_pp", lambda values: int((values > 0).sum())),
        )
        .reset_index()
    )
    return paired.merge(
        aggregate,
        on=["initialization", "k", "comparison"],
        how="left",
        validate="many_to_one",
    )


def plot_accuracy(summary: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, initialization in zip(axes, INITIALIZATIONS, strict=True):
        panel = summary[summary["initialization"] == initialization]
        for method in METHODS:
            curve = panel[panel["method"] == method].sort_values("k")
            ax.errorbar(
                curve["k"],
                100.0 * curve["mean_test_acc"],
                yerr=100.0 * curve["std_test_acc"],
                label=DISPLAY_NAMES[method],
                color=COLORS[method],
                marker=MARKERS[method],
                linewidth=2,
                markersize=6,
                capsize=3,
            )
        ax.set_title("From scratch" if initialization == "scratch" else "ImageNet pretrained")
        ax.set_xlabel("Training examples per class (k)")
        ax.set_xticks(K_VALUES)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Test accuracy (%)")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("CIFAR-100 confirmation: mean ± sample SD across three seeds")
    fig.text(
        0.5,
        0.01,
        "k=450 uses the predeclared 50-epoch large-data recipe; other points use 100 epochs.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_deltas(paired: pd.DataFrame, output_path: Path) -> None:
    aggregate = paired.drop_duplicates(
        ["initialization", "k", "comparison"]
    )
    comparisons = (
        ("simcutmix_vs_none", "SimCutMix − Crop+Flip", "#E45756", "D"),
        ("simcutmix_vs_cutmix", "SimCutMix − CutMix", "#4C78A8", "s"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, initialization in zip(axes, INITIALIZATIONS, strict=True):
        panel = aggregate[aggregate["initialization"] == initialization]
        for comparison, label, color, marker in comparisons:
            curve = panel[panel["comparison"] == comparison].sort_values("k")
            ax.errorbar(
                curve["k"],
                curve["mean_delta_test_pp"],
                yerr=curve["std_delta_test_pp"],
                label=label,
                color=color,
                marker=marker,
                linewidth=2,
                markersize=6,
                capsize=3,
            )
        ax.axhline(0, color="#333333", linewidth=1, linestyle="--")
        ax.set_title("From scratch" if initialization == "scratch" else "ImageNet pretrained")
        ax.set_xlabel("Training examples per class (k)")
        ax.set_xticks(K_VALUES)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Paired test-accuracy difference (pp)")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Frozen SimCutMix: paired differences across three seeds")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def markdown_report(summary: pd.DataFrame, paired: pd.DataFrame) -> str:
    lines = [
        "# Third-presentation CIFAR-100 confirmation results",
        "",
        "Test accuracy is reported as mean ± sample standard deviation across three paired seeds.",
        "The frozen SimCutMix configuration uses class-agnostic ranks 21–40, uniform sampling,",
        "CutMix alpha 1, mixing probability 1, and no warmup.",
        "",
    ]
    for initialization in INITIALIZATIONS:
        title = "Scratch" if initialization == "scratch" else "ImageNet pretrained"
        lines.extend(
            [
                f"## {title}",
                "",
                "| k | Crop+Flip | CutMix | SimCutMix |",
                "|---:|---:|---:|---:|",
            ]
        )
        panel = summary[summary["initialization"] == initialization]
        for k in K_VALUES:
            cells = []
            for method in METHODS:
                row = panel[(panel["k"] == k) & (panel["method"] == method)].iloc[0]
                cells.append(
                    f"{100 * row.mean_test_acc:.2f} ± {100 * row.std_test_acc:.2f}"
                )
            lines.append(f"| {k} | " + " | ".join(cells) + " |")
        lines.append("")

    lines.extend(
        [
            "## Paired conclusions",
            "",
            "Across all 36 initialization/k/seed comparisons:",
            "",
        ]
    )
    for comparison, label in (
        ("simcutmix_vs_none", "SimCutMix versus Crop+Flip"),
        ("simcutmix_vs_cutmix", "SimCutMix versus CutMix"),
        ("cutmix_vs_none", "CutMix versus Crop+Flip"),
    ):
        rows = paired[paired["comparison"] == comparison]
        mean = rows["delta_test_pp"].mean()
        wins = int((rows["delta_test_pp"] > 0).sum())
        lines.append(f"- {label}: mean delta {mean:+.2f} pp; wins {wins}/36 paired cells.")
    lines.extend(
        [
            "",
            "The k=450 cells use subset seed 0 with three training seeds and therefore measure",
            "optimization variability only. They also use the predeclared 50-epoch recipe, so the",
            "drop from k=300 to k=450 must not be interpreted as a pure data-size effect.",
            "",
            "## Provenance notes",
            "",
            "- The reused pretrained seed-0 results at k=20/50/100 had historical test exposure.",
            "  Configuration selection used validation evidence only, but the prior exposure must",
            "  remain disclosed.",
            "- Eighteen concurrent seed-0 evaluation jobs recorded `git_dirty=true` because they",
            "  observed other tracked result summaries being updated. Commit `5fd2e14` changes only",
            "  the 27 evaluated summaries; no source or scientific configuration changed.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/manifests/third_presentation_phase1.csv"),
    )
    parser.add_argument(
        "--final-root",
        type=Path,
        default=Path("results/third_presentation/final"),
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=Path("results/third_presentation/analysis"),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("docs/figures/third_presentation"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/third_presentation_results.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = load_confirmation_runs(args.manifest, args.final_root)
    summary = aggregate_runs(runs)
    paired = paired_deltas(runs)

    args.table_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.table_dir / "confirmation_runs.csv", index=False)
    summary.to_csv(args.table_dir / "confirmation_summary.csv", index=False)
    paired.to_csv(args.table_dir / "paired_deltas.csv", index=False)
    plot_accuracy(summary, args.figure_dir / "test_accuracy_vs_k.png")
    plot_deltas(paired, args.figure_dir / "paired_test_deltas_vs_k.png")
    args.report.write_text(markdown_report(summary, paired), encoding="utf-8")

    print(f"Validated {len(runs)} confirmation cells.")
    print(f"Wrote tables to {args.table_dir}")
    print(f"Wrote figures to {args.figure_dir}")
    print(f"Wrote report to {args.report}")


if __name__ == "__main__":
    main()
