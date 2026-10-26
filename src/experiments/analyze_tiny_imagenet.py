"""Validate and summarize the frozen Tiny ImageNet confirmation experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

try:
    from .analyze_third_presentation import aggregate_runs, paired_deltas
except ImportError:  # Support `python src/experiments/analyze_tiny_imagenet.py`.
    from analyze_third_presentation import aggregate_runs, paired_deltas


INITIALIZATIONS = ("scratch", "pretrained")
K_VALUES = (20, 450)
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
    expected = {
        "dataset": "tinyimagenet",
        "model": "resnet50",
        "pretrained": initialization == "pretrained",
        "k": k,
        "subset_seed": expected_subset_seed(k, train_seed),
        "train_seed": train_seed,
        "augmentation": method,
        "epochs": 50 if k == 450 else 100,
        "batch_size": 32,
        "optimizer": "sgd",
        "lr": 0.1 if k == 450 else 0.01,
        "momentum": 0.9,
        "nesterov": True,
        "weight_decay": 0.0005,
        "lr_milestones": [15, 30, 40] if k == 450 else [30, 55, 75],
        "lr_gamma": 0.2 if k == 450 else 0.1,
        "num_train": 200 * k,
        "num_val": 10_000,
        "num_test": 10_000,
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
    for key in (
        "best_val_acc",
        "validation_acc_recomputed",
        "test_acc_best_checkpoint",
        "test_loss_best_checkpoint",
    ):
        if not isinstance(summary.get(key), (int, float)):
            mismatches.append(f"{key} is missing or non-numeric")
    if summary.get("validation_acc_recomputed") != summary.get("best_val_acc"):
        mismatches.append("recomputed validation accuracy differs from the selected value")
    if mismatches:
        raise ValueError("; ".join(mismatches))


def load_confirmation_runs(screening_root: Path) -> pd.DataFrame:
    summaries = [(path, load_json(path)) for path in screening_root.rglob("summary.json")]
    leaked = [
        path
        for path, summary in summaries
        if summary.get("test_evaluated") and summary.get("k") not in K_VALUES
    ]
    if leaked:
        raise ValueError(f"Test evaluation found outside the frozen k values: {leaked}")

    records = []
    for initialization in INITIALIZATIONS:
        for k in K_VALUES:
            for method in METHODS:
                for train_seed in (0, 1, 2):
                    subset_seed = expected_subset_seed(k, train_seed)
                    matches = [
                        (path, summary)
                        for path, summary in summaries
                        if summary.get("pretrained") == (initialization == "pretrained")
                        and summary.get("k") == k
                        and summary.get("augmentation") == method
                        and summary.get("subset_seed") == subset_seed
                        and summary.get("train_seed") == train_seed
                    ]
                    if len(matches) != 1:
                        raise ValueError(
                            f"Expected one summary for {initialization} k={k} {method} "
                            f"seed={train_seed}; found {len(matches)}"
                        )
                    path, summary = matches[0]
                    try:
                        validate_summary(summary, initialization, k, method, train_seed)
                    except ValueError as error:
                        raise ValueError(f"{path}: {error}") from error
                    records.append(
                        {
                            "initialization": initialization,
                            "k": k,
                            "method": method,
                            "subset_seed": subset_seed,
                            "train_seed": train_seed,
                            "best_val_acc": summary["best_val_acc"],
                            "test_acc": summary["test_acc_best_checkpoint"],
                            "best_epoch": summary["best_epoch"],
                            "evaluation_git_commit": summary["evaluation_runtime"]["git_commit"],
                            "evaluation_git_dirty": summary["evaluation_runtime"]["git_dirty"],
                            "summary_path": path.as_posix(),
                        }
                    )
    runs = pd.DataFrame.from_records(records)
    if len(runs) != 36:
        raise ValueError(f"Expected 36 frozen confirmation cells; found {len(runs)}")
    return runs


def plot_accuracy(summary: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharey=False)
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
                markersize=7,
                capsize=4,
            )
        ax.set_title("From scratch" if initialization == "scratch" else "ImageNet pretrained")
        ax.set_xlabel("Training examples per class (k)")
        ax.set_xticks(K_VALUES)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Final test accuracy (%)")
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle("Tiny ImageNet: mean ± sample SD across three paired seeds")
    fig.text(
        0.5,
        0.01,
        "k=450 uses the predeclared 50-epoch recipe; k=20 uses 100 epochs.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_deltas(paired: pd.DataFrame, output_path: Path) -> None:
    aggregate = paired.drop_duplicates(["initialization", "k", "comparison"])
    comparisons = (
        ("simcutmix_vs_none", "SimCutMix − Crop+Flip", "#E45756", "D"),
        ("simcutmix_vs_cutmix", "SimCutMix − CutMix", "#4C78A8", "s"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharey=True)
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
                markersize=7,
                capsize=4,
            )
        ax.axhline(0, color="#333333", linewidth=1, linestyle="--")
        ax.set_title("From scratch" if initialization == "scratch" else "ImageNet pretrained")
        ax.set_xlabel("Training examples per class (k)")
        ax.set_xticks(K_VALUES)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Paired final-test difference (pp)")
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle("Tiny ImageNet: frozen SimCutMix paired effects")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def markdown_report(summary: pd.DataFrame, paired: pd.DataFrame, runs: pd.DataFrame) -> str:
    lines = [
        "# Tiny ImageNet confirmation results",
        "",
        "Final-test accuracy is reported as mean ± sample standard deviation across three paired seeds.",
        "The test set is the labeled official Tiny ImageNet validation partition; the official",
        "unlabeled test partition was never used.",
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
                cells.append(f"{100 * row.mean_test_acc:.2f} ± {100 * row.std_test_acc:.2f}")
            lines.append(f"| {k} | " + " | ".join(cells) + " |")
        lines.append("")

    lines.extend(["## Paired effects", ""])
    aggregate = paired.drop_duplicates(["initialization", "k", "comparison"])
    for initialization in INITIALIZATIONS:
        title = "Scratch" if initialization == "scratch" else "Pretrained"
        for k in K_VALUES:
            row = aggregate[
                (aggregate["initialization"] == initialization)
                & (aggregate["k"] == k)
                & (aggregate["comparison"] == "simcutmix_vs_none")
            ].iloc[0]
            lines.append(
                f"- {title}, k={k}: SimCutMix − Crop+Flip "
                f"{row.mean_delta_test_pp:+.2f} ± {row.std_delta_test_pp:.2f} pp; "
                f"wins {int(row.seed_wins)}/3 seeds."
            )

    dirty_count = int(runs["evaluation_git_dirty"].sum())
    commits = ", ".join(sorted(runs["evaluation_git_commit"].unique()))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The positive result is regime-specific: SimCutMix improves low-data training from",
            "scratch, but it does not improve the pretrained k=20 condition or either k=450",
            "condition. The k=450 recipe differs in learning rate and epoch count, so changes",
            "between k=20 and k=450 are not a pure data-size ablation.",
            "",
            "The six k=100 screening runs remain validation-only and are excluded from final-test",
            "claims because they were not repeated across three seeds.",
            "",
            "## Provenance",
            "",
            f"- Final evaluation used Git commit `{commits}`.",
            f"- `{dirty_count}/36` evaluation records report `git_dirty=true` because concurrent",
            "  jobs observed other tracked summary files being updated; source code and scientific",
            "  configurations were unchanged.",
            "- Validation accuracy recomputed from every saved checkpoint exactly matched the",
            "  validation accuracy used to select that checkpoint.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--screening-root",
        type=Path,
        default=Path("results/third_presentation/tiny_imagenet/screening"),
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=Path("results/third_presentation/tiny_imagenet/analysis"),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("docs/figures/third_presentation"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/tiny_imagenet_results.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = load_confirmation_runs(args.screening_root)
    summary = aggregate_runs(runs)
    paired = paired_deltas(runs)

    args.table_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.table_dir / "confirmation_runs.csv", index=False)
    summary.to_csv(args.table_dir / "confirmation_summary.csv", index=False)
    paired.to_csv(args.table_dir / "paired_deltas.csv", index=False)
    plot_accuracy(summary, args.figure_dir / "tiny_imagenet_test_accuracy.png")
    plot_deltas(paired, args.figure_dir / "tiny_imagenet_paired_test_deltas.png")
    args.report.write_text(markdown_report(summary, paired, runs), encoding="utf-8")

    print(f"Validated {len(runs)} frozen Tiny ImageNet cells.")
    print(f"Wrote tables to {args.table_dir}")
    print(f"Wrote figures to {args.figure_dir}")
    print(f"Wrote report to {args.report}")


if __name__ == "__main__":
    main()
