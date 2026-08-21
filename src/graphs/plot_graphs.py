"""Generate honest, publication-ready views of the active experiment evidence.

The active experiment matrix is intentionally incomplete.  This script therefore
separates:

1. direct proposal-versus-baseline comparisons with the same training recipe;
2. validation curves for only those direct comparisons;
3. all available test results, where missing points remain visibly missing; and
4. experiment coverage, including the number of runs in every cell.

Single-run results are shown as points without fake zero-width error bars.
When repeated runs become available, aggregate plots add sample-standard-
deviation error bars or bands automatically.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


DEFAULT_EXPERIMENTS_DIR = Path("results/experiments")
# These figures are intentionally versioned so GitHub presents the current
# research evidence without requiring a reviewer to run the plotting script.
# Use --output-dir for local, throwaway figure generation instead.
DEFAULT_FIGURES_DIR = Path("docs/figures")

METHOD_ORDER = ["none", "mixup", "cutmix", "augmix", "simmixup", "simcutmix"]
BASELINE_AUGMENTATIONS = ("none", "mixup", "cutmix", "augmix")
PROPOSAL_TO_BASELINE = {
    "simmixup": "mixup",
    "simcutmix": "cutmix",
}

DISPLAY_NAMES = {
    "none": "No augmentation",
    "mixup": "MixUp",
    "cutmix": "CutMix",
    "augmix": "AugMix",
    "simmixup": "SimMixUp",
    "simcutmix": "SimCutMix",
}

COLORS = {
    "none": "#5F6368",
    "mixup": "#4C78A8",
    "cutmix": "#F58518",
    "augmix": "#54A24B",
    "simmixup": "#B279A2",
    "simcutmix": "#E45756",
}

MARKERS = {
    "none": "o",
    "mixup": "s",
    "cutmix": "^",
    "augmix": "P",
    "simmixup": "D",
    "simcutmix": "D",
}

# Fields that should be identical before a proposal result is called a direct
# comparison with a baseline. Augmentation-specific parameters are deliberately
# excluded: they define the methods being compared.
MATCHED_RECIPE_FIELDS = [
    "dataset",
    "model",
    "k",
    "subset_seed",
    "train_seed",
    "epochs",
    "batch_size",
    "lr",
    "weight_decay",
    "optimizer",
    "momentum",
    "nesterov",
    "lr_milestones",
    "lr_gamma",
    "num_train",
    "num_val",
    "num_test",
]


def configure_plot_style() -> None:
    """Apply a restrained style shared by all generated figures."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#A0A0A0",
            "axes.labelcolor": "#222222",
            "axes.titleweight": "semibold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#D8D8D8",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.7,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "legend.frameon": False,
            "savefig.facecolor": "white",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
        }
    )


def format_number(value: Any) -> str:
    """Return compact, stable numeric text for labels and configuration keys."""
    if value is None:
        return "?"
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value)


def friendly_dataset(value: str) -> str:
    value = str(value)
    if value.lower().startswith("cifar") and value[5:].isdigit():
        return f"CIFAR-{value[5:]}"
    return value


def friendly_model(value: str) -> str:
    names = {
        "resnet50": "ResNet50",
        "vit": "ViT",
    }
    return names.get(str(value).lower(), str(value))


def method_sort_key(method: str) -> tuple[int, str]:
    try:
        return METHOD_ORDER.index(str(method)), str(method)
    except ValueError:
        return len(METHOD_ORDER), str(method)


def normalized_value(value: Any) -> Any:
    """Make nested JSON values deterministic and hashable through serialization."""
    if isinstance(value, dict):
        return {key: normalized_value(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [normalized_value(item) for item in value]
    return value


def recipe_key(data: dict[str, Any]) -> str:
    recipe = {
        field: normalized_value(data.get(field, "__missing__"))
        for field in MATCHED_RECIPE_FIELDS
    }
    return json.dumps(recipe, sort_keys=True, separators=(",", ":"))


def augmentation_configuration(data: dict[str, Any]) -> dict[str, Any]:
    """Return only parameters that identify a plotted augmentation series."""
    method = str(data["augmentation"])
    if method == "none":
        return {}
    if method == "mixup":
        return {"alpha": data.get("mixup_alpha")}
    if method == "cutmix":
        return {
            "alpha": data.get("mixup_alpha"),
            "probability": data.get("cutmix_prob"),
        }
    if method == "augmix":
        # Current summaries do not expose AugMix-specific settings.
        return {}
    if method in PROPOSAL_TO_BASELINE:
        return {
            "guided_mode": data.get("guided_mode"),
            "neighbor_k": data.get("neighbor_k"),
            "neighbor_rank_start": data.get("neighbor_rank_start", 1),
            "pair_sampling": data.get("pair_sampling"),
            "alpha": data.get("mixup_alpha"),
            "mix_probability": data.get("mix_prob"),
            "warmup_epochs": data.get("mix_warmup_epochs"),
            "anchor_selection": data.get("anchor_selection"),
            "anchor_top_pct": data.get("anchor_top_pct"),
            "anchor_score_power": data.get("anchor_score_power"),
            "dynamic_neighbor_pool": data.get("dynamic_neighbor_pool"),
        }
    return {}


def series_key(data: dict[str, Any]) -> str:
    config = augmentation_configuration(data)
    return f"{data['augmentation']}|{json.dumps(config, sort_keys=True, separators=(',', ':'))}"


def series_label(data: dict[str, Any]) -> str:
    """Build a readable label that prevents guided variants being averaged."""
    method = str(data["augmentation"])
    base = DISPLAY_NAMES.get(method, method)
    if method not in PROPOSAL_TO_BASELINE:
        return base

    mode = {
        "class_aware": "class-aware",
        "class_agnostic": "class-agnostic",
        "different_label": "different-label",
    }.get(str(data.get("guided_mode")), str(data.get("guided_mode", "unknown")))
    neighbor_k = int(data.get("neighbor_k", 0))
    rank_start = int(data.get("neighbor_rank_start", 1))
    rank_end = rank_start + neighbor_k - 1
    return f"{base} ({mode}, ranks {rank_start}–{rank_end})"


def configured_series_label(
    base_label: str,
    augmentation: str,
    series_key: str,
) -> str:
    """Add augmentation parameters when one method has multiple series."""
    try:
        config_text = series_key.split("|", 1)[1]
        config = json.loads(config_text)
    except (IndexError, json.JSONDecodeError):
        return base_label

    if augmentation == "mixup" and config.get("alpha") is not None:
        return f"{base_label} (alpha={format_number(config['alpha'])})"
    if augmentation == "cutmix":
        details: list[str] = []
        if config.get("alpha") is not None:
            details.append(f"alpha={format_number(config['alpha'])}")
        if config.get("probability") is not None:
            details.append(f"p={format_number(config['probability'])}")
        if details:
            return f"{base_label} ({', '.join(details)})"
    return base_label


def resolve_metrics_path(
    summary_path: Path,
    recorded_path: str | None,
    experiments_dir: Path,
) -> Path:
    """Resolve a metric CSV robustly across Windows/Unix and moved repositories."""
    candidates = [summary_path.parent / "metrics.csv"]

    if recorded_path:
        normalized = Path(str(recorded_path).replace("\\", "/"))
        if normalized.is_absolute():
            candidates.append(normalized)
        else:
            candidates.append(Path.cwd() / normalized)
            try:
                repo_root = experiments_dir.resolve().parents[1]
                candidates.append(repo_root / normalized)
            except IndexError:
                pass

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    # Returning the canonical sibling path gives callers the most useful warning.
    return candidates[0].resolve()


def load_summary_metrics(experiments_dir: Path) -> pd.DataFrame:
    """Load canonical summaries while retaining configuration and provenance."""
    experiments_dir = Path(experiments_dir)
    rows: list[dict[str, Any]] = []

    for path in sorted(experiments_dir.rglob("summary.json")):
        if "legacy" in {part.lower() for part in path.parts}:
            continue

        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        required = {
            "dataset",
            "model",
            "k",
            "subset_seed",
            "augmentation",
            "epochs",
            "best_epoch",
            "best_val_acc",
            "test_acc_best_checkpoint",
        }
        missing = required.difference(data)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"{path} is missing required fields: {missing_text}")

        metrics_path = resolve_metrics_path(
            summary_path=path,
            recorded_path=data.get("metrics_path"),
            experiments_dir=experiments_dir,
        )
        method = str(data["augmentation"])
        run_dir = path.parent
        try:
            run_id = run_dir.relative_to(experiments_dir).as_posix()
        except ValueError:
            run_id = run_dir.as_posix()

        rows.append(
            {
                "run_id": run_id,
                "summary_path": str(path.resolve()),
                "metrics_path": str(metrics_path),
                "metrics_exists": metrics_path.exists(),
                "dataset": str(data["dataset"]),
                "model": str(data["model"]),
                "k": int(data["k"]),
                "subset_seed": int(data["subset_seed"]),
                "train_seed": int(data.get("train_seed", 0)),
                "augmentation": method,
                "method_name": DISPLAY_NAMES.get(method, method),
                "series_key": series_key(data),
                "series_label": series_label(data),
                "recipe_key": recipe_key(data),
                "epochs": int(data["epochs"]),
                "best_epoch": int(data["best_epoch"]),
                "best_val_acc": float(data["best_val_acc"]),
                "test_acc": float(data["test_acc_best_checkpoint"]),
                "test_loss": float(data.get("test_loss_best_checkpoint", np.nan)),
            }
        )

    if not rows:
        raise FileNotFoundError(f"No summary JSON files found in {experiments_dir}")

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["dataset", "model", "k", "augmentation", "series_key", "subset_seed", "train_seed"]
        )
        .reset_index(drop=True)
    )


def aggregate_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated runs without pretending a single run has uncertainty."""
    grouped = (
        runs.groupby(
            [
                "dataset",
                "model",
                "k",
                "augmentation",
                "method_name",
                "series_key",
                "series_label",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            mean_test_acc=("test_acc", "mean"),
            std_test_acc=("test_acc", "std"),
            mean_best_val_acc=("best_val_acc", "mean"),
            std_best_val_acc=("best_val_acc", "std"),
            runs=("run_id", "nunique"),
            min_epochs=("epochs", "min"),
            max_epochs=("epochs", "max"),
        )
    )
    return grouped.sort_values(
        ["dataset", "model", "k", "augmentation", "series_key"]
    ).reset_index(drop=True)


def build_all_baseline_comparisons(runs: pd.DataFrame) -> pd.DataFrame:
    """Pair each proposal run with every available baseline using its exact recipe.

    The recipe fields must match so that the comparison remains controlled, but
    the baseline augmentation is not restricted to the proposal's paired
    counterpart. This produces one comparison row for each available
    ``proposal × baseline`` pair.
    """
    rows: list[dict[str, Any]] = []
    for proposal in runs[runs["augmentation"].isin(PROPOSAL_TO_BASELINE)].itertuples(
        index=False
    ):
        candidates = runs[
            (runs["augmentation"].isin(BASELINE_AUGMENTATIONS))
            & (runs["recipe_key"] == proposal.recipe_key)
        ]

        for baseline in candidates.itertuples(index=False):
            rows.append(
                {
                    "dataset": proposal.dataset,
                    "model": proposal.model,
                    "k": proposal.k,
                    "subset_seed": proposal.subset_seed,
                    "train_seed": proposal.train_seed,
                    "epochs": proposal.epochs,
                    "baseline_augmentation": baseline.augmentation,
                    "baseline_label": baseline.method_name,
                    "baseline_series_key": baseline.series_key,
                    "baseline_run_id": baseline.run_id,
                    "baseline_test_acc": baseline.test_acc,
                    "baseline_best_val_acc": baseline.best_val_acc,
                    "baseline_best_epoch": baseline.best_epoch,
                    "proposal_augmentation": proposal.augmentation,
                    "proposal_label": proposal.method_name,
                    "proposal_series_label": proposal.series_label,
                    "proposal_series_key": proposal.series_key,
                    "proposal_run_id": proposal.run_id,
                    "proposal_test_acc": proposal.test_acc,
                    "proposal_best_val_acc": proposal.best_val_acc,
                    "proposal_best_epoch": proposal.best_epoch,
                    "delta_test_pp": 100.0 * (proposal.test_acc - baseline.test_acc),
                }
            )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        [
            "dataset",
            "model",
            "k",
            "proposal_augmentation",
            "proposal_series_key",
            "subset_seed",
            "train_seed",
        ]
    ).reset_index(drop=True)


def build_matched_comparisons(runs: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible alias for all recipe-matched baseline comparisons."""
    return build_all_baseline_comparisons(runs)


def load_epoch_metrics(runs: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run in runs.itertuples(index=False):
        path = Path(run.metrics_path)
        if not path.exists():
            print(f"Skipping missing metric CSV: {path}")
            continue

        metrics = pd.read_csv(path)
        required = {"epoch", "train_acc", "val_acc"}
        missing = required.difference(metrics.columns)
        if missing:
            missing_text = ", ".join(sorted(missing))
            print(f"Skipping {path}; missing metric columns: {missing_text}")
            continue

        metrics["run_id"] = run.run_id
        metrics["dataset"] = run.dataset
        metrics["model"] = run.model
        metrics["k"] = run.k
        metrics["augmentation"] = run.augmentation
        metrics["method_name"] = run.method_name
        metrics["series_key"] = run.series_key
        frames.append(metrics)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def aggregate_matched_comparisons(comparisons: pd.DataFrame) -> pd.DataFrame:
    """Aggregate paired comparisons for scalar test-accuracy figures."""
    if comparisons.empty:
        return pd.DataFrame()

    group_columns = [
        "dataset",
        "model",
        "k",
        "baseline_augmentation",
        "baseline_label",
        "baseline_series_key",
        "proposal_augmentation",
        "proposal_label",
        "proposal_series_label",
        "proposal_series_key",
    ]
    return (
        comparisons.groupby(group_columns, as_index=False)
        .agg(
            baseline_mean=("baseline_test_acc", "mean"),
            baseline_std=("baseline_test_acc", "std"),
            proposal_mean=("proposal_test_acc", "mean"),
            proposal_std=("proposal_test_acc", "std"),
            delta_mean_pp=("delta_test_pp", "mean"),
            delta_std_pp=("delta_test_pp", "std"),
            pairs=("proposal_run_id", "nunique"),
        )
        .sort_values(["dataset", "model", "k", "proposal_series_key"])
        .reset_index(drop=True)
    )


def plot_matched_test_accuracy(
    comparisons: pd.DataFrame,
    output_dir: Path,
) -> bool:
    """Draw connected baseline/proposal points with an explicit delta."""
    if comparisons.empty:
        return False

    grouped = aggregate_matched_comparisons(comparisons)
    grouped = grouped.sort_values(
        [
            "dataset",
            "model",
            "k",
            "proposal_series_key",
            "baseline_augmentation",
            "baseline_series_key",
        ]
    )

    height = max(6.0, 0.64 * len(grouped) + 2.4)
    fig, ax = plt.subplots(figsize=(13.5, height))
    y_positions = np.arange(len(grouped))[::-1]
    observed_max = 0.0
    baseline_series_counts = grouped.groupby("baseline_augmentation")[
        "baseline_series_key"
    ].nunique()

    for y, row in zip(y_positions, grouped.itertuples(index=False)):
        baseline_pct = 100.0 * row.baseline_mean
        proposal_pct = 100.0 * row.proposal_mean
        observed_max = max(observed_max, baseline_pct, proposal_pct)

        ax.plot(
            [baseline_pct, proposal_pct],
            [y, y],
            color="#A7A7A7",
            linewidth=2.2,
            zorder=1,
        )
        ax.scatter(
            baseline_pct,
            y,
            s=90,
            marker=MARKERS[row.baseline_augmentation],
            color=COLORS[row.baseline_augmentation],
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )
        ax.scatter(
            proposal_pct,
            y,
            s=100,
            marker=MARKERS[row.proposal_augmentation],
            color=COLORS[row.proposal_augmentation],
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )

        if row.pairs > 1:
            ax.errorbar(
                baseline_pct,
                y,
                xerr=100.0 * row.baseline_std,
                fmt="none",
                color=COLORS[row.baseline_augmentation],
                capsize=3,
                zorder=2,
            )
            ax.errorbar(
                proposal_pct,
                y,
                xerr=100.0 * row.proposal_std,
                fmt="none",
                color=COLORS[row.proposal_augmentation],
                capsize=3,
                zorder=2,
            )

        ax.annotate(
            f"{baseline_pct:.2f}%",
            (baseline_pct, y),
            xytext=(0, -12),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8.5,
            color="#333333",
        )
        ax.annotate(
            f"{proposal_pct:.2f}%",
            (proposal_pct, y),
            xytext=(0, 11),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#333333",
        )

    right_edge = min(100.0, max(20.0, math.ceil((observed_max + 13.0) / 5.0) * 5.0))
    for y, row in zip(y_positions, grouped.itertuples(index=False)):
        delta_text = f"Δ {row.delta_mean_pp:+.2f} pp"
        if row.pairs > 1:
            delta_text += f" ± {row.delta_std_pp:.2f}"
        ax.text(
            right_edge - 0.7,
            y,
            delta_text,
            ha="right",
            va="center",
            fontweight="semibold",
            color="#222222",
        )

    labels: list[str] = []
    for row in grouped.itertuples(index=False):
        baseline_label = row.baseline_label
        if baseline_series_counts[row.baseline_augmentation] > 1:
            baseline_label = configured_series_label(
                baseline_label,
                row.baseline_augmentation,
                row.baseline_series_key,
            )
        labels.append(
            f"{friendly_model(row.model)} · k={row.k}\n"
            f"{row.proposal_series_label} vs {baseline_label}"
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlim(0.0, right_edge)
    ax.set_ylim(-0.65, len(grouped) - 0.35)
    ax.set_xlabel("Test accuracy (%) at the best-validation checkpoint")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    fig.suptitle(
        "Proposal versus all recipe-matched baselines",
        y=0.994,
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.978,
        "Connected points show baseline → proposal; Δ = proposal − baseline",
        ha="center",
        va="top",
        color="#555555",
        fontsize=9.5,
    )

    legend_items: list[Line2D] = []
    seen: set[tuple[str, str]] = set()
    for row in grouped.itertuples(index=False):
        for role, method, series_key, label in (
            (
                "baseline",
                row.baseline_augmentation,
                row.baseline_series_key,
                row.baseline_label,
            ),
            (
                "proposal",
                row.proposal_augmentation,
                row.proposal_series_key,
                row.proposal_series_label,
            ),
        ):
            key = (method, series_key)
            if key in seen:
                continue
            seen.add(key)
            if role == "baseline" and baseline_series_counts[method] > 1:
                label = configured_series_label(label, method, series_key)
            legend_items.append(
                Line2D(
                    [0],
                    [0],
                    marker=MARKERS[method],
                    color="none",
                    markerfacecolor=COLORS[method],
                    markeredgecolor="white",
                    markersize=8,
                    label=label,
                )
            )
    fig.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.952),
        ncol=min(4, max(1, len(legend_items))),
        fontsize=8.5,
        frameon=False,
        columnspacing=1.4,
        handletextpad=0.4,
    )

    if grouped["pairs"].max() == 1:
        fig.text(
            0.5,
            0.01,
            "All displayed comparisons currently contain one run; no uncertainty interval is estimable.",
            ha="center",
            color="#555555",
            fontsize=9,
        )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.90))
    save_figure(fig, output_dir / "matched_test_accuracy.png")
    return True


def plot_matched_test_accuracy_bars(
    comparisons: pd.DataFrame,
    output_dir: Path,
) -> bool:
    """Draw a grouped-bar alternative for the matched test comparisons."""
    grouped = aggregate_matched_comparisons(comparisons)
    if grouped.empty:
        return False

    x = np.arange(len(grouped), dtype=float)
    width = 0.34
    baseline_values = 100.0 * grouped["baseline_mean"].to_numpy()
    proposal_values = 100.0 * grouped["proposal_mean"].to_numpy()
    baseline_colors = [
        COLORS.get(method, "#777777") for method in grouped["baseline_augmentation"]
    ]
    proposal_colors = [
        COLORS.get(method, "#777777") for method in grouped["proposal_augmentation"]
    ]

    fig_width = max(7.2, 2.4 * len(grouped) + 3.2)
    fig, ax = plt.subplots(figsize=(fig_width, 5.2))
    baseline_bars = ax.bar(
        x - width / 2,
        baseline_values,
        width,
        color=baseline_colors,
        edgecolor="white",
        linewidth=0.9,
        zorder=3,
    )
    proposal_bars = ax.bar(
        x + width / 2,
        proposal_values,
        width,
        color=proposal_colors,
        edgecolor="white",
        linewidth=0.9,
        zorder=3,
    )

    for index, row in enumerate(grouped.itertuples(index=False)):
        if row.pairs > 1:
            ax.errorbar(
                x[index] - width / 2,
                baseline_values[index],
                yerr=100.0 * row.baseline_std,
                fmt="none",
                color="#333333",
                capsize=3,
                linewidth=1.0,
                zorder=4,
            )
            ax.errorbar(
                x[index] + width / 2,
                proposal_values[index],
                yerr=100.0 * row.proposal_std,
                fmt="none",
                color="#333333",
                capsize=3,
                linewidth=1.0,
                zorder=4,
            )

        ax.text(
            x[index],
            max(baseline_values[index], proposal_values[index]) + 5.1,
            f"Δ {row.delta_mean_pp:+.2f} pp",
            ha="center",
            va="bottom",
            fontweight="semibold",
            color="#222222",
        )

    ax.bar_label(
        baseline_bars,
        labels=[f"{value:.2f}%" for value in baseline_values],
        padding=3,
        fontsize=9,
    )
    ax.bar_label(
        proposal_bars,
        labels=[f"{value:.2f}%" for value in proposal_values],
        padding=3,
        fontsize=9,
    )

    labels: list[str] = []
    for row in grouped.itertuples(index=False):
        label = f"{friendly_model(row.model)}\nk={row.k}"
        duplicates = grouped[
            (grouped["dataset"] == row.dataset)
            & (grouped["model"] == row.model)
            & (grouped["k"] == row.k)
        ]
        if len(duplicates) > 1:
            label += f"\n{row.proposal_series_label}"
        labels.append(label)

    observed_max = max(baseline_values.max(), proposal_values.max())
    upper = min(100.0, max(20.0, math.ceil((observed_max + 13.0) / 5.0) * 5.0))
    ax.set_ylim(0.0, upper)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test accuracy (%) at the best-validation checkpoint")
    ax.set_title(
        "Pairwise proposal-versus-baseline test accuracy",
        fontsize=15,
        pad=28,
    )
    ax.text(
        0.5,
        1.015,
        "All available baselines with the same recorded training recipe",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color="#555555",
        fontsize=9.5,
    )
    ax.grid(axis="y")
    ax.set_axisbelow(True)

    legend_items: list[Patch] = []
    seen_methods: set[str] = set()
    for method, label in zip(
        grouped["baseline_augmentation"], grouped["baseline_label"]
    ):
        if method not in seen_methods:
            legend_items.append(
                Patch(facecolor=COLORS.get(method, "#777777"), label=label)
            )
            seen_methods.add(method)
    for method, label in zip(
        grouped["proposal_augmentation"], grouped["proposal_label"]
    ):
        if method not in seen_methods:
            legend_items.append(
                Patch(facecolor=COLORS.get(method, "#777777"), label=label)
            )
            seen_methods.add(method)
    ax.legend(handles=legend_items, loc="upper left", ncol=min(3, len(legend_items)))

    if grouped["pairs"].max() == 1:
        fig.text(
            0.5,
            0.01,
            "Each bar currently represents one run; no uncertainty interval is estimable.",
            ha="center",
            color="#555555",
            fontsize=9,
        )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    save_figure(fig, output_dir / "matched_test_accuracy_bars.png")
    return True


def plot_matched_validation_curves(
    comparisons: pd.DataFrame,
    epoch_metrics: pd.DataFrame,
    output_dir: Path,
) -> bool:
    """Plot each proposal trajectory against every recipe-matched baseline."""
    if comparisons.empty or epoch_metrics.empty:
        return False

    panel_columns = [
        "dataset",
        "model",
        "k",
        "proposal_augmentation",
        "proposal_label",
        "proposal_series_label",
        "proposal_series_key",
    ]
    panels = comparisons[panel_columns].drop_duplicates().sort_values(
        ["dataset", "model", "k", "proposal_series_key"]
    )
    n_panels = len(panels)
    n_cols = min(2, n_panels)
    n_rows = math.ceil(n_panels / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.3 * n_cols, 4.2 * n_rows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for ax, panel in zip(axes_flat, panels.itertuples(index=False)):
        panel_pairs = comparisons[
            (comparisons["dataset"] == panel.dataset)
            & (comparisons["model"] == panel.model)
            & (comparisons["k"] == panel.k)
            & (comparisons["proposal_series_key"] == panel.proposal_series_key)
        ]
        proposal_ids = panel_pairs["proposal_run_id"].unique()
        best_baseline = panel_pairs["baseline_test_acc"].max()
        proposal_test = panel_pairs["proposal_test_acc"].iloc[0]
        delta = 100.0 * (proposal_test - best_baseline)

        series_specs: list[tuple[Any, ...]] = []
        baseline_specs = (
            panel_pairs[
                ["baseline_augmentation", "baseline_label", "baseline_series_key"]
            ]
            .drop_duplicates()
            .sort_values(
                ["baseline_augmentation", "baseline_series_key"],
                key=lambda column: column.map(method_sort_key)
                if column.name == "baseline_augmentation"
                else column,
            )
        )
        for baseline in baseline_specs.itertuples(index=False):
            baseline_ids = panel_pairs[
                (panel_pairs["baseline_augmentation"] == baseline.baseline_augmentation)
                & (panel_pairs["baseline_series_key"] == baseline.baseline_series_key)
            ]["baseline_run_id"].unique()
            series_specs.append(
                (
                    baseline.baseline_label,
                    baseline.baseline_augmentation,
                    baseline_ids,
                    "--",
                    "baseline_best_epoch",
                    "baseline_best_val_acc",
                    -20,
                    baseline.baseline_series_key,
                    "baseline",
                )
            )
        series_specs.append(
            (
                panel.proposal_series_label,
                panel.proposal_augmentation,
                proposal_ids,
                "-",
                "proposal_best_epoch",
                "proposal_best_val_acc",
                15,
                panel.proposal_series_key,
                "proposal",
            )
        )
        panel_curve_max = 0.0
        for (
            label,
            method,
            run_ids,
            line_style,
            checkpoint_epoch_column,
            checkpoint_value_column,
            label_y_offset,
            comparison_series_key,
            role,
        ) in series_specs:
            subset = epoch_metrics[epoch_metrics["run_id"].isin(run_ids)]
            curve = (
                subset.groupby("epoch", as_index=False)
                .agg(
                    mean_val_acc=("val_acc", "mean"),
                    std_val_acc=("val_acc", "std"),
                    runs=("run_id", "nunique"),
                )
                .sort_values("epoch")
            )
            if curve.empty:
                continue

            x = curve["epoch"].to_numpy()
            y = 100.0 * curve["mean_val_acc"].to_numpy()
            panel_curve_max = max(panel_curve_max, float(np.nanmax(y)))
            run_count = len(run_ids)
            ax.plot(
                x,
                y,
                color=COLORS[method],
                linewidth=2.0,
                linestyle=line_style,
                label=f"{label} (n={run_count})",
            )
            if run_count > 1:
                std = 100.0 * curve["std_val_acc"].fillna(0.0).to_numpy()
                ax.fill_between(
                    x,
                    y - std,
                    y + std,
                    color=COLORS[method],
                    alpha=0.14,
                    linewidth=0,
                )

            if role == "baseline":
                checkpoint_source = panel_pairs[
                    panel_pairs["baseline_series_key"] == comparison_series_key
                ]
            else:
                checkpoint_source = panel_pairs
            checkpoint_rows = checkpoint_source[
                [checkpoint_epoch_column, checkpoint_value_column]
            ].drop_duplicates()
            for checkpoint in checkpoint_rows.itertuples(index=False, name=None):
                checkpoint_epoch = int(checkpoint[0])
                checkpoint_value = 100.0 * float(checkpoint[1])
                ax.scatter(
                    checkpoint_epoch,
                    checkpoint_value,
                    s=115,
                    marker="*",
                    color=COLORS[method],
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=4,
                )

            annotate_checkpoint = role == "proposal"
            if role == "baseline":
                baseline_test_values = panel_pairs[
                    panel_pairs["baseline_series_key"] == comparison_series_key
                ]["baseline_test_acc"]
                annotate_checkpoint = (
                    not baseline_test_values.empty
                    and np.isclose(float(baseline_test_values.iloc[0]), best_baseline)
                )
            if len(checkpoint_rows) == 1 and annotate_checkpoint:
                checkpoint_epoch = int(checkpoint_rows.iloc[0, 0])
                checkpoint_value = 100.0 * float(checkpoint_rows.iloc[0, 1])
                max_epoch = float(curve["epoch"].max())
                label_x_offset = -8 if checkpoint_epoch >= 0.8 * max_epoch else 8
                ax.annotate(
                    f"best checkpoint: epoch {checkpoint_epoch}, {checkpoint_value:.2f}%",
                    (checkpoint_epoch, checkpoint_value),
                    xytext=(label_x_offset, label_y_offset),
                    textcoords="offset points",
                    ha="right" if label_x_offset < 0 else "left",
                    va="bottom" if label_y_offset > 0 else "top",
                    color=COLORS[method],
                    fontsize=8.5,
                    fontweight="semibold",
                    arrowprops={
                        "arrowstyle": "-",
                        "color": COLORS[method],
                        "linewidth": 0.8,
                    },
                )
            elif len(checkpoint_rows) > 1:
                min_epoch = int(checkpoint_rows[checkpoint_epoch_column].min())
                max_epoch = int(checkpoint_rows[checkpoint_epoch_column].max())
                ax.text(
                    0.02,
                    0.96 if method == panel.proposal_augmentation else 0.90,
                    f"{label} best-checkpoint epochs: {min_epoch}–{max_epoch}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    color=COLORS[method],
                    fontsize=8.5,
                )

        ax.set_title(
            f"{friendly_model(panel.model)} · {friendly_dataset(panel.dataset)} · k={panel.k}\n"
            f"Difference versus best baseline: {delta:+.2f} pp",
            fontsize=11,
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation accuracy (%)")
        ax.set_ylim(0.0, min(100.0, math.ceil((panel_curve_max + 8.0) / 5.0) * 5.0))
        ax.grid()
        ax.set_axisbelow(True)
        ax.legend(loc="lower right")

    for ax in axes_flat[n_panels:]:
        ax.axis("off")

    fig.suptitle(
        "Validation trajectories and saved best checkpoints",
        fontsize=15,
        y=1.01,
    )
    fig.tight_layout()
    save_figure(fig, output_dir / "matched_validation_curves.png")
    return True


def panel_series_label(panel: pd.DataFrame, row: pd.Series) -> str:
    """Use detailed labels only if a method has multiple configurations."""
    method_configs = panel.loc[
        panel["augmentation"] == row["augmentation"], "series_key"
    ].nunique()
    if method_configs > 1:
        return configured_series_label(
            row["method_name"],
            row["augmentation"],
            row["series_key"],
        )
    return row["method_name"]


def spread_label_positions(
    values: list[float],
    minimum_gap: float,
    lower: float,
    upper: float,
) -> list[float]:
    """Spread nearby vertical labels while preserving their value order."""
    if not values:
        return []

    order = np.argsort(values)
    sorted_values = np.array([values[index] for index in order], dtype=float)
    positions = sorted_values.copy()

    for index in range(1, len(positions)):
        positions[index] = max(positions[index], positions[index - 1] + minimum_gap)

    overflow = positions[-1] - upper
    if overflow > 0:
        positions -= overflow

    for index in range(len(positions) - 2, -1, -1):
        positions[index] = min(positions[index], positions[index + 1] - minimum_gap)

    underflow = lower - positions[0]
    if underflow > 0:
        positions += underflow

    restored = np.zeros(len(values), dtype=float)
    for sorted_index, original_index in enumerate(order):
        restored[original_index] = positions[sorted_index]
    return restored.tolist()


def plot_available_test_accuracy(
    aggregated: pd.DataFrame,
    output_dir: Path,
) -> bool:
    """Draw grouped bars for all active test results without imputing gaps."""
    if aggregated.empty:
        return False

    combos = (
        aggregated[["dataset", "model"]]
        .drop_duplicates()
        .sort_values(["dataset", "model"])
    )
    n_panels = len(combos)
    n_cols = min(2, n_panels)
    n_rows = math.ceil(n_panels / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7.2 * n_cols, 5.4 * n_rows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for ax, combo in zip(axes_flat, combos.itertuples(index=False)):
        panel = aggregated[
            (aggregated["dataset"] == combo.dataset)
            & (aggregated["model"] == combo.model)
        ].copy()
        k_values = sorted(panel["k"].unique())
        x_lookup = {k: index for index, k in enumerate(k_values)}

        series_order = (
            panel[["augmentation", "series_key", "series_label"]]
            .drop_duplicates()
            .assign(
                sort_key=lambda frame: frame["augmentation"].map(
                    lambda value: method_sort_key(value)[0]
                )
            )
            .sort_values(["sort_key", "series_key"])
        )
        x = np.arange(len(k_values), dtype=float)
        n_series = len(series_order)
        width = min(0.18, 0.82 / max(1, n_series))
        offsets = (np.arange(n_series) - (n_series - 1) / 2) * width

        for offset, series in zip(offsets, series_order.itertuples(index=False)):
            subset = panel[
                (panel["augmentation"] == series.augmentation)
                & (panel["series_key"] == series.series_key)
            ].sort_values("k")
            row0 = subset.iloc[0]
            label = panel_series_label(panel, row0)
            by_k = subset.set_index("k")
            values = np.asarray(
                [
                    100.0 * float(by_k.loc[k, "mean_test_acc"])
                    if k in by_k.index
                    else np.nan
                    for k in k_values
                ],
                dtype=float,
            )
            runs = np.asarray(
                [
                    float(by_k.loc[k, "runs"]) if k in by_k.index else np.nan
                    for k in k_values
                ],
                dtype=float,
            )
            bars = ax.bar(
                x + offset,
                values,
                width,
                color=COLORS.get(series.augmentation, "#777777"),
                edgecolor="white",
                linewidth=0.8,
                label=label,
                zorder=3,
            )
            errors = np.asarray(
                [
                    100.0 * float(by_k.loc[k, "std_test_acc"])
                    if k in by_k.index and pd.notna(by_k.loc[k, "std_test_acc"])
                    else np.nan
                    for k in k_values
                ],
                dtype=float,
            )
            if np.isfinite(errors).any():
                ax.errorbar(
                    x + offset,
                    values,
                    yerr=errors,
                    fmt="none",
                    ecolor=COLORS.get(series.augmentation, "#777777"),
                    capsize=3,
                    linewidth=1.2,
                    zorder=2,
                )
            for bar, value, run_count in zip(bars, values, runs):
                if np.isfinite(value):
                    suffix = f"\n(n={int(run_count)})" if run_count > 1 else ""
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        value + 0.8,
                        f"{value:.1f}{suffix}",
                        ha="center",
                        va="bottom",
                        fontsize=7.5,
                        rotation=90 if n_series > 4 else 0,
                    )

        panel_max = 100.0 * panel["mean_test_acc"].max()
        upper = min(100.0, max(20.0, math.ceil((panel_max + 12.0) / 10.0) * 10.0))

        ax.set_ylim(0.0, upper)
        ax.set_xticks(x)
        ax.set_xticklabels([f"k={value}" for value in k_values])
        ax.set_xlabel("Training examples per class (k)")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title(f"{friendly_model(combo.model)} · {friendly_dataset(combo.dataset)}")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", fontsize=8.0)

    for ax in axes_flat[n_panels:]:
        ax.axis("off")

    fig.suptitle(
        "All available test accuracy by method and data budget",
        fontsize=15,
        y=1.02,
    )
    fig.text(
        0.5,
        0.005,
        "Missing method–k combinations are not imputed; labels show test accuracy at the best-validation checkpoint.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 1.0))
    save_figure(fig, output_dir / "available_test_accuracy_vs_k.png")
    return True


def plot_experiment_coverage(runs: pd.DataFrame, output_dir: Path) -> bool:
    """Show how many complete summary/metrics pairs exist in each matrix cell."""
    if runs.empty:
        return False

    datasets = sorted(runs["dataset"].unique())
    panel_keys = [
        (dataset, model)
        for dataset in datasets
        for model in sorted(runs.loc[runs["dataset"] == dataset, "model"].unique())
    ]
    k_values = sorted(runs["k"].unique())
    observed_methods = sorted(runs["augmentation"].unique(), key=method_sort_key)
    methods = METHOD_ORDER + [
        method for method in observed_methods if method not in METHOD_ORDER
    ]

    n_panels = len(panel_keys)
    n_cols = min(2, n_panels)
    n_rows = math.ceil(n_panels / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.0 * n_cols, max(3.6, 0.62 * len(methods) + 2.1) * n_rows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    complete = runs[runs["metrics_exists"]].copy()
    counts = (
        complete.groupby(["dataset", "model", "k", "augmentation"])["run_id"]
        .nunique()
        .to_dict()
    )

    missing_color = np.array([0.92, 0.93, 0.94, 1.0])
    baseline_color = np.array([76 / 255, 120 / 255, 168 / 255, 0.82])
    proposal_color = np.array([228 / 255, 87 / 255, 86 / 255, 0.82])

    for ax, (dataset, model) in zip(axes_flat, panel_keys):
        rgba = np.zeros((len(methods), len(k_values), 4), dtype=float)
        rgba[:] = missing_color

        for row_index, method in enumerate(methods):
            for column_index, k in enumerate(k_values):
                count = counts.get((dataset, model, k, method), 0)
                if count:
                    rgba[row_index, column_index] = (
                        proposal_color if method in PROPOSAL_TO_BASELINE else baseline_color
                    )

        ax.imshow(rgba, aspect="auto", interpolation="none")
        ax.set_xticks(range(len(k_values)))
        ax.set_xticklabels([f"k={value}" for value in k_values])
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([DISPLAY_NAMES.get(method, method) for method in methods])
        ax.set_title(f"{friendly_model(model)} · {friendly_dataset(dataset)}")
        ax.tick_params(length=0)

        for row_index, method in enumerate(methods):
            for column_index, k in enumerate(k_values):
                count = counts.get((dataset, model, k, method), 0)
                ax.text(
                    column_index,
                    row_index,
                    f"n={count}" if count else "—",
                    ha="center",
                    va="center",
                    color="white" if count else "#666666",
                    fontweight="semibold" if count else "normal",
                    fontsize=9,
                )

        for x in np.arange(-0.5, len(k_values), 1.0):
            ax.axvline(x, color="white", linewidth=2)
        for y in np.arange(-0.5, len(methods), 1.0):
            ax.axhline(y, color="white", linewidth=2)

    for ax in axes_flat[n_panels:]:
        ax.axis("off")

    legend = [
        Patch(facecolor=baseline_color, edgecolor="none", label="Baseline run"),
        Patch(facecolor=proposal_color, edgecolor="none", label="Proposal run"),
        Patch(facecolor=missing_color, edgecolor="none", label="Missing"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(
        "Active experiment coverage",
        fontsize=15,
        y=1.01,
    )
    fig.text(
        0.5,
        0.06,
        "Counts include runs with both a summary and an epoch-metrics file.",
        ha="center",
        color="#555555",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.14, 1.0, 1.0))
    save_figure(fig, output_dir / "experiment_coverage.png")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate comparison figures from canonical experiment records."
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=DEFAULT_EXPERIMENTS_DIR,
        help=f"Canonical experiment root (default: {DEFAULT_EXPERIMENTS_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Figure output directory (default: docs/figures).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiments_dir = args.experiments_dir
    output_dir = args.output_dir or DEFAULT_FIGURES_DIR

    configure_plot_style()
    runs = load_summary_metrics(experiments_dir)
    aggregated = aggregate_runs(runs)
    comparisons = build_all_baseline_comparisons(runs)
    epoch_metrics = load_epoch_metrics(runs)

    generated: list[str] = []
    if plot_matched_test_accuracy(comparisons, output_dir):
        generated.append("matched_test_accuracy.png")
    if plot_matched_validation_curves(comparisons, epoch_metrics, output_dir):
        generated.append("matched_validation_curves.png")
    if plot_available_test_accuracy(aggregated, output_dir):
        generated.append("available_test_accuracy_vs_k.png")
    if plot_experiment_coverage(runs, output_dir):
        generated.append("experiment_coverage.png")

    print(f"Loaded {len(runs)} active experiment summaries.")
    print(f"Found {len(comparisons)} proposal-baseline run pairs.")
    print(f"Generated {len(generated)} figures in {output_dir}:")
    for filename in generated:
        print(f"  - {filename}")
    if comparisons.empty:
        print(
            "No direct proposal-baseline figure was generated because no proposal "
            "run had a baseline with the same recorded training recipe."
        )


if __name__ == "__main__":
    main()
