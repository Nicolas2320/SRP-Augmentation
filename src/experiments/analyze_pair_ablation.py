"""Validate and summarize the controlled CIFAR-100 partner-selection ablation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


SETTINGS = (
    ("scratch", 20),
    ("scratch", 100),
    ("scratch", 300),
    ("pretrained", 100),
)
STRATEGIES = (
    "random",
    "same_class",
    "most_similar",
    "middle_similarity",
    "less_similar",
    "different_label",
)
DISPLAY_NAMES = {
    "random": "Random",
    "same_class": "Same class",
    "most_similar": "Most similar\nranks 1–10",
    "middle_similarity": "Middle\nranks 11–20",
    "less_similar": "Less similar\nranks 21–40",
    "different_label": "Different label",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def strategy_from_summary(summary: dict[str, Any]) -> str:
    if summary.get("augmentation") == "cutmix" and not summary.get("neighbor_path"):
        return "random"
    if summary.get("augmentation") != "simcutmix":
        raise ValueError("Ablation summary is neither random CutMix nor guided SimCutMix")
    mode = summary.get("guided_mode")
    neighbor_k = summary.get("neighbor_k")
    rank_start = summary.get("neighbor_rank_start")
    if mode == "class_aware" and neighbor_k == 10 and rank_start == 1:
        return "same_class"
    if mode == "different_label" and neighbor_k == 10 and rank_start == 1:
        return "different_label"
    if mode == "class_agnostic" and neighbor_k == 10 and rank_start == 1:
        return "most_similar"
    if mode == "class_agnostic" and neighbor_k == 10 and rank_start == 11:
        return "middle_similarity"
    if mode == "class_agnostic" and neighbor_k == 20 and rank_start == 21:
        return "less_similar"
    raise ValueError(
        f"Unknown guided strategy: mode={mode!r}, neighbor_k={neighbor_k!r}, "
        f"neighbor_rank_start={rank_start!r}"
    )


def validate_summary(
    summary: dict[str, Any], initialization: str, k: int, strategy: str
) -> None:
    expected = {
        "dataset": "cifar100",
        "model": "resnet50",
        "pretrained": initialization == "pretrained",
        "k": k,
        "subset_seed": 0,
        "train_seed": 0,
        "epochs": 100,
        "batch_size": 32,
        "optimizer": "sgd",
        "lr": 0.01,
        "momentum": 0.9,
        "nesterov": True,
        "weight_decay": 0.0005,
        "lr_milestones": [30, 55, 75],
        "lr_gamma": 0.1,
        "cutmix_alpha": 1.0,
        "cutmix_prob": 0.5,
    }
    if strategy == "random":
        expected["augmentation"] = "cutmix"
    else:
        expected.update(
            {
                "augmentation": "simcutmix",
                "pair_sampling": "uniform",
                "mix_prob": 0.5,
                "mix_warmup_epochs": 0,
                "test_evaluated": False,
            }
        )

    mismatches = [
        f"{key}={summary.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if summary.get(key) != value
    ]
    if strategy_from_summary(summary) != strategy:
        mismatches.append("summary fields map to a different strategy")
    if not isinstance(summary.get("best_val_acc"), (int, float)):
        mismatches.append("best_val_acc is missing or non-numeric")
    if mismatches:
        raise ValueError("; ".join(mismatches))


def random_paths(manifest_path: Path) -> dict[tuple[str, int], Path]:
    selected: dict[tuple[str, int], Path] = {}
    required = set(SETTINGS)
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["initialization"], int(row["k"]))
            if key not in required or row["method"] != "cutmix":
                continue
            if row["candidate_classification"] != "reusable":
                raise ValueError(f"Random control is not reusable: {row['job_key']}")
            selected[key] = Path(row["candidate_summary"])
    missing = required.difference(selected)
    if missing:
        raise ValueError(f"Manifest is missing random controls for: {sorted(missing)}")
    return selected


def load_ablation_runs(manifest_path: Path, ablation_root: Path) -> pd.DataFrame:
    random = random_paths(manifest_path)
    guided = [(path, load_json(path)) for path in ablation_root.rglob("summary.json")]
    records = []
    for initialization, k in SETTINGS:
        for strategy in STRATEGIES:
            if strategy == "random":
                path = random[(initialization, k)]
                summary = load_json(path)
            else:
                matches = [
                    (path, summary)
                    for path, summary in guided
                    if summary.get("pretrained") == (initialization == "pretrained")
                    and summary.get("k") == k
                    and strategy_from_summary(summary) == strategy
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"Expected one {initialization} k={k} {strategy} summary; "
                        f"found {len(matches)}"
                    )
                path, summary = matches[0]
            try:
                validate_summary(summary, initialization, k, strategy)
            except ValueError as error:
                raise ValueError(f"{path}: {error}") from error
            records.append(
                {
                    "initialization": initialization,
                    "k": k,
                    "strategy": strategy,
                    "best_val_acc": summary["best_val_acc"],
                    "best_epoch": summary["best_epoch"],
                    "test_evaluated": summary["test_evaluated"],
                    "summary_path": path.as_posix(),
                }
            )
    runs = pd.DataFrame.from_records(records)
    if len(runs) != 24:
        raise ValueError(f"Expected 24 random/guided cells; found {len(runs)}")
    random_values = runs[runs["strategy"] == "random"].set_index(
        ["initialization", "k"]
    )["best_val_acc"]
    runs["delta_vs_random_pp"] = [
        100.0 * (row.best_val_acc - random_values.loc[(row.initialization, row.k)])
        for row in runs.itertuples(index=False)
    ]
    return runs


def summary_matrix(runs: pd.DataFrame, value: str) -> pd.DataFrame:
    matrix = runs.pivot(index=["initialization", "k"], columns="strategy", values=value)
    return matrix.reindex(index=pd.MultiIndex.from_tuples(SETTINGS), columns=STRATEGIES)


def plot_ablation(runs: pd.DataFrame, output_path: Path) -> None:
    accuracy = 100.0 * summary_matrix(runs, "best_val_acc")
    delta = summary_matrix(runs, "delta_vs_random_pp")
    row_labels = [
        f"{'Scratch' if initialization == 'scratch' else 'Pretrained'}, k={k}"
        for initialization, k in SETTINGS
    ]
    column_labels = [DISPLAY_NAMES[strategy] for strategy in STRATEGIES]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.2), constrained_layout=True)
    panels = (
        (axes[0], accuracy, "Best validation accuracy (%)", "viridis", None),
        (
            axes[1],
            delta,
            "Difference from random CutMix (pp)",
            "RdBu_r",
            TwoSlopeNorm(vmin=min(-2.2, float(delta.min().min())), vcenter=0, vmax=max(2.2, float(delta.max().max()))),
        ),
    )
    for ax, values, title, cmap, norm in panels:
        image = ax.imshow(values.to_numpy(), aspect="auto", cmap=cmap, norm=norm)
        ax.set_title(title)
        ax.set_xticks(range(len(column_labels)), column_labels, rotation=25, ha="right")
        ax.set_yticks(range(len(row_labels)), row_labels)
        threshold = float(np.nanmean(values.to_numpy()))
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = float(values.iloc[row, column])
                label = f"{value:+.2f}" if title.startswith("Difference") else f"{value:.2f}"
                color = "white" if (norm is None and value < threshold) else "black"
                ax.text(column, row, label, ha="center", va="center", color=color, fontsize=9)
        fig.colorbar(image, ax=ax, shrink=0.82)
    fig.suptitle("CIFAR-100 controlled partner-selection ablation (seed 0, validation only)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def markdown_report(runs: pd.DataFrame) -> str:
    accuracy = 100.0 * summary_matrix(runs, "best_val_acc")
    delta = summary_matrix(runs, "delta_vs_random_pp")
    lines = [
        "# Pair-selection ablation results",
        "",
        "This controlled CIFAR-100 ablation holds the CutMix operator, alpha (1.0), mixing",
        "probability (0.5), optimizer, schedule, subset seed, and training seed fixed. Only",
        "the partner-selection rule changes. Values are best validation accuracy from seed 0;",
        "they are selection evidence, not final multi-seed test estimates.",
        "",
        "| Initialization | k | Random | Same class | Most similar | Middle | Less similar | Different label |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row_index, (initialization, k) in enumerate(SETTINGS):
        cells = [f"{accuracy.iloc[row_index, column]:.2f}" for column in range(len(STRATEGIES))]
        title = "Scratch" if initialization == "scratch" else "Pretrained"
        lines.append(f"| {title} | {k} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Differences from random CutMix",
            "",
            "| Initialization | k | Same class | Most similar | Middle | Less similar | Different label |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row_index, (initialization, k) in enumerate(SETTINGS):
        cells = [
            f"{delta.iloc[row_index, column]:+.2f}"
            for column in range(1, len(STRATEGIES))
        ]
        title = "Scratch" if initialization == "scratch" else "Pretrained"
        lines.append(f"| {title} | {k} | " + " | ".join(cells) + " |")

    strategy_means = (
        runs[runs["strategy"] != "random"]
        .groupby("strategy", sort=False)["delta_vs_random_pp"]
        .mean()
        .sort_values(ascending=False)
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"The ranks 21–40 condition has the largest mean validation difference across the four",
            f"settings ({strategy_means['less_similar']:+.2f} pp versus random) and wins two settings.",
            "It was therefore frozen before the three-seed confirmation runs. The result is not",
            "universal: middle ranks win at scratch k=20, while random CutMix wins at scratch k=300.",
            "Same-class pairing is weakest on average, especially for pretrained k=100.",
            "",
            "`different_label` means the nearest eligible image with a different label; it is not",
            "a truly most-distant or lowest-cosine-similarity condition.",
            "",
            "## Test-boundary note",
            "",
            "The 20 guided ablation runs were validation-only. Some reused random controls were",
            "later test-evaluated as part of the frozen confirmation matrix, but configuration",
            "selection used only the validation values recorded here.",
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
        "--ablation-root",
        type=Path,
        default=Path("results/third_presentation/ablation"),
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=Path("results/third_presentation/ablation_analysis"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("docs/figures/third_presentation/pair_selection_ablation.png"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/pair_selection_ablation_results.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = load_ablation_runs(args.manifest, args.ablation_root)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(
        args.table_dir / "pair_selection_runs.csv", index=False, float_format="%.6f"
    )
    summary_matrix(runs, "best_val_acc").to_csv(
        args.table_dir / "pair_selection_validation_accuracy.csv", float_format="%.6f"
    )
    summary_matrix(runs, "delta_vs_random_pp").to_csv(
        args.table_dir / "pair_selection_deltas_vs_random.csv", float_format="%.6f"
    )
    plot_ablation(runs, args.figure)
    args.report.write_text(markdown_report(runs), encoding="utf-8")
    print(f"Validated {len(runs)} matched ablation cells (20 guided + 4 random).")
    print(f"Wrote tables to {args.table_dir}")
    print(f"Wrote figure to {args.figure}")
    print(f"Wrote report to {args.report}")


if __name__ == "__main__":
    main()
