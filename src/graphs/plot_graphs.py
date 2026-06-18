"""Generate comparison plots for SRP augmentation experiments.

The script reads:
- summary JSON files for final/best-checkpoint metrics
- CSV files for epoch-level training curves

It produces six high-value figures in results/figures:
1. test_accuracy_by_augmentation.png
2. test_accuracy_vs_k.png
3. val_accuracy_curves_by_augmentation.png
4. train_accuracy_curves_by_augmentation.png
5. baseline_improvement.png
6. test_accuracy_heatmap.png
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRICS_DIR = Path("results/metrics_v2")
FIGURES_DIR = Path("results/figures")

AUGMENTATION_ORDER = ["none", "mixup", "cutmix", "augmix", "simmixup"]
MODEL_ORDER = ["resnet50", "vit"]
COLORS = {
    "none": "#4C566A",
    "mixup": "#5E81AC",
    "cutmix": "#A3BE8C",
    "augmix": "#D08770",
    "simmixup": "#B48EAD",
}

DISPLAY_NAMES = {
    "none": "None",
    "mixup": "MixUp",
    "cutmix": "CutMix",
    "augmix": "AugMix",
}


def save_current_figure(path: Path, tight_layout: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def format_float_label(value) -> str:
    return f"{float(value):g}".replace(".", "p")


def neighbor_rank_label(data: dict) -> str:
    neighbor_k = data.get("neighbor_k", "na")
    rank_start = int(data.get("neighbor_rank_start", 1))
    try:
        rank_end = rank_start + int(neighbor_k) - 1
    except (TypeError, ValueError):
        rank_end = "?"
    return f"r{rank_start}-{rank_end}"


def augmentation_label(data: dict) -> str:
    """Return a plot label that keeps SimMixUp variants separate."""
    augmentation = data["augmentation"]
    if augmentation != "simmixup":
        return augmentation

    guided_mode = data.get("guided_mode", "unknown")
    neighbor_k = data.get("neighbor_k", "na")
    rank_label = neighbor_rank_label(data)
    pair_sampling = data.get("pair_sampling", "uniform")
    mix_prob = format_float_label(data.get("mix_prob", 1.0))
    warmup = int(data.get("mix_warmup_epochs", 0))
    guided_short = {
        "class_aware": "ca",
        "class_agnostic": "cg",
    }.get(guided_mode, guided_mode)

    label = f"simmixup_{guided_short}_nk{neighbor_k}_{rank_label}_{pair_sampling}_mp{mix_prob}"
    if warmup > 0:
        label = f"{label}_warm{warmup}"
    return label


def augmentation_base(label: str) -> str:
    for augmentation in AUGMENTATION_ORDER:
        if label == augmentation or label.startswith(f"{augmentation}_"):
            return augmentation
    return label


def augmentation_sort_key(label: str) -> tuple[int, str]:
    base = augmentation_base(label)
    try:
        base_index = AUGMENTATION_ORDER.index(base)
    except ValueError:
        base_index = len(AUGMENTATION_ORDER)
    return base_index, label


def ordered_augmentations(df: pd.DataFrame) -> list[str]:
    labels = [str(label) for label in df["augmentation"].dropna().unique()]
    return sorted(labels, key=augmentation_sort_key)


def augmentation_color(label: str) -> str:
    return COLORS.get(augmentation_base(label), "#888888")


def augmentation_line_style(label: str) -> str:
    label = str(label)
    if label.startswith("simmixup_cg_"):
        return "--"
    if label.startswith("simmixup_ca_"):
        return "-"
    return "-"


def augmentation_display_label(label: str) -> str:
    """Return compact labels for legends and axes."""
    label = str(label)
    if label in DISPLAY_NAMES:
        return DISPLAY_NAMES[label]
    if not label.startswith("simmixup_"):
        return label

    parts = label.split("_")
    mode = parts[1] if len(parts) > 1 else "?"
    mode_name = {
        "ca": "CA",
        "cg": "CG",
    }.get(mode, mode.upper())

    neighbor_k = next((part[2:] for part in parts if part.startswith("nk")), "?")
    rank_window = next((part[1:] for part in parts if part.startswith("r") and "-" in part), None)
    mix_prob = next((part[2:] for part in parts if part.startswith("mp")), "1")
    display = f"SMU-{mode_name} k{neighbor_k}"
    if rank_window is not None:
        display = f"{display} r{rank_window}"
    if mix_prob != "1":
        display = f"{display} p{mix_prob}"
    return display


def load_summary_metrics(metrics_dir: Path) -> pd.DataFrame:
    rows = []

    for path in sorted(metrics_dir.glob("*_summary.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        rows.append(
            {
                "dataset": data["dataset"],
                "model": data["model"],
                "k": int(data["k"]),
                "subset_seed": int(data["subset_seed"]),
                "augmentation": augmentation_label(data),
                "base_augmentation": data["augmentation"],
                "best_epoch": int(data["best_epoch"]),
                "best_val_acc": float(data["best_val_acc"]),
                "test_acc": float(data["test_acc_best_checkpoint"]),
                "test_loss": float(data["test_loss_best_checkpoint"]),
                "metrics_path": data["metrics_path"],
            }
        )

    if not rows:
        raise FileNotFoundError(f"No summary JSON files found in {metrics_dir}")

    df = pd.DataFrame(rows)
    augmentation_categories = ordered_augmentations(df)
    df["augmentation"] = pd.Categorical(
        df["augmentation"],
        categories=augmentation_categories,
        ordered=True,
    )
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    return df.sort_values(["dataset", "model", "k", "augmentation", "subset_seed"])


def aggregate_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(
            ["dataset", "model", "k", "augmentation"],
            observed=True,
            as_index=False,
        )
        .agg(
            mean_test_acc=("test_acc", "mean"),
            std_test_acc=("test_acc", "std"),
            mean_val_acc=("best_val_acc", "mean"),
            mean_best_epoch=("best_epoch", "mean"),
            runs=("test_acc", "count"),
        )
        .sort_values(["dataset", "model", "k", "augmentation"])
    )
    grouped["std_test_acc"] = grouped["std_test_acc"].fillna(0.0)
    return grouped


def resolve_metrics_path(path_text: str, metrics_dir: Path) -> Path:
    path = Path(path_text)
    if path.exists():
        return path

    fallback = metrics_dir / Path(path_text.replace("\\", "/")).name
    return fallback


def load_epoch_metrics(summary_df: pd.DataFrame, metrics_dir: Path) -> pd.DataFrame:
    frames = []

    for row in summary_df.itertuples(index=False):
        metrics_path = resolve_metrics_path(row.metrics_path, metrics_dir)
        if not metrics_path.exists():
            print(f"Skipping missing metric CSV: {metrics_path}")
            continue

        metrics = pd.read_csv(metrics_path)
        metrics["dataset"] = row.dataset
        metrics["model"] = row.model
        metrics["k"] = row.k
        metrics["subset_seed"] = row.subset_seed
        metrics["augmentation"] = row.augmentation
        metrics["source_csv"] = str(metrics_path)
        frames.append(metrics)

    if not frames:
        raise FileNotFoundError(f"No metric CSV files found from summaries in {metrics_dir}")

    return pd.concat(frames, ignore_index=True)


def plot_test_accuracy_by_augmentation(summary: pd.DataFrame) -> None:
    combos = summary[["dataset", "model", "k"]].drop_duplicates()
    n_plots = len(combos)
    n_cols = min(3, n_plots)
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, combo in zip(axes_flat, combos.itertuples(index=False)):
        subset = summary[
            (summary["dataset"] == combo.dataset)
            & (summary["model"] == combo.model)
            & (summary["k"] == combo.k)
        ].sort_values("augmentation")

        labels = subset["augmentation"].astype(str).tolist()
        display_labels = [augmentation_display_label(label) for label in labels]
        values = subset["mean_test_acc"].tolist()
        errors = subset["std_test_acc"].tolist()
        colors = [augmentation_color(label) for label in labels]

        ax.bar(display_labels, values, yerr=errors, color=colors, edgecolor="#2E3440", linewidth=0.8)
        ax.set_title(f"{combo.dataset} | {combo.model} | k={combo.k}")
        ax.set_ylabel("Test accuracy")
        ax.set_ylim(0, max(values + [0.05]) * 1.2)
        ax.grid(axis="y", alpha=0.25)

        for i, value in enumerate(values):
            ax.text(i, value + 0.005, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    for ax in axes_flat[n_plots:]:
        ax.axis("off")

    fig.suptitle("Test Accuracy by Augmentation", fontsize=16, y=1.02)
    save_current_figure(FIGURES_DIR / "test_accuracy_by_augmentation.png")


def plot_test_accuracy_vs_k(summary: pd.DataFrame) -> None:
    models = [m for m in MODEL_ORDER if m in set(summary["model"].astype(str))]
    n_cols = len(models)

    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 4.5), squeeze=False)

    for ax, model in zip(axes.flatten(), models):
        model_df = summary[summary["model"].astype(str) == model]

        for augmentation in ordered_augmentations(model_df):
            subset = model_df[model_df["augmentation"].astype(str) == augmentation].sort_values("k")
            if subset.empty:
                continue

            ax.plot(
                subset["k"],
                subset["mean_test_acc"],
                marker="o",
                linewidth=2,
                label=augmentation_display_label(augmentation),
                color=augmentation_color(augmentation),
                linestyle=augmentation_line_style(augmentation),
            )

        ax.set_title(model)
        ax.set_xlabel("Training examples per class (k)")
        ax.set_ylabel("Test accuracy")
        ax.grid(alpha=0.25)
        ax.legend(title="Augmentation")

    fig.suptitle("Test Accuracy vs k-shot Size", fontsize=16, y=1.02)
    save_current_figure(FIGURES_DIR / "test_accuracy_vs_k.png")


def plot_accuracy_curves(
    epoch_metrics: pd.DataFrame,
    metric_column: str,
    aggregate_column: str,
    ylabel: str,
    title: str,
    output_filename: str,
) -> None:
    """Plot train or validation accuracy curves by augmentation."""
    curve_df = (
        epoch_metrics.groupby(
            ["dataset", "model", "k", "augmentation", "epoch"],
            observed=True,
            as_index=False,
        )
        .agg(**{aggregate_column: (metric_column, "mean")})
        .sort_values(["dataset", "model", "k", "augmentation", "epoch"])
    )

    combos = curve_df[["dataset", "model", "k"]].drop_duplicates()
    n_plots = len(combos)
    n_cols = min(3, n_plots)
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, combo in zip(axes_flat, combos.itertuples(index=False)):
        subset = curve_df[
            (curve_df["dataset"] == combo.dataset)
            & (curve_df["model"] == combo.model)
            & (curve_df["k"] == combo.k)
        ]

        for augmentation in ordered_augmentations(subset):
            aug_df = subset[subset["augmentation"].astype(str) == augmentation]
            if aug_df.empty:
                continue

            ax.plot(
                aug_df["epoch"],
                aug_df[aggregate_column],
                linewidth=2,
                label=augmentation_display_label(augmentation),
                color=augmentation_color(augmentation),
                linestyle=augmentation_line_style(augmentation),
            )

        ax.set_title(f"{combo.dataset} | {combo.model} | k={combo.k}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    for ax in axes_flat[n_plots:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=16, y=1.02)
    save_current_figure(FIGURES_DIR / output_filename)


def plot_val_accuracy_curves(epoch_metrics: pd.DataFrame) -> None:
    plot_accuracy_curves(
        epoch_metrics=epoch_metrics,
        metric_column="val_acc",
        aggregate_column="mean_val_acc",
        ylabel="Validation accuracy",
        title="Validation Accuracy Curves by Augmentation",
        output_filename="val_accuracy_curves_by_augmentation.png",
    )


def plot_train_accuracy_curves(epoch_metrics: pd.DataFrame) -> None:
    plot_accuracy_curves(
        epoch_metrics=epoch_metrics,
        metric_column="train_acc",
        aggregate_column="mean_train_acc",
        ylabel="Training accuracy",
        title="Training Accuracy Curves by Augmentation",
        output_filename="train_accuracy_curves_by_augmentation.png",
    )


def plot_baseline_improvement(summary: pd.DataFrame) -> None:
    baseline = summary[summary["augmentation"].astype(str) == "none"][
        ["dataset", "model", "k", "mean_test_acc"]
    ].rename(columns={"mean_test_acc": "baseline_test_acc"})

    improved = summary[summary["augmentation"].astype(str) != "none"].merge(
        baseline,
        on=["dataset", "model", "k"],
        how="inner",
    )
    improved["improvement"] = improved["mean_test_acc"] - improved["baseline_test_acc"]
    improved["experiment"] = (
        improved["model"].astype(str) + "\n" + improved["dataset"].astype(str) + " k=" + improved["k"].astype(str)
    )

    experiments = improved["experiment"].drop_duplicates().tolist()
    augmentations = [aug for aug in ordered_augmentations(improved) if aug != "none"]
    x = range(len(experiments))
    width = min(0.8 / max(len(augmentations), 1), 0.24)

    fig, ax = plt.subplots(figsize=(max(9, len(experiments) * 1.3), 5))

    for offset, augmentation in enumerate(augmentations):
        subset = improved[improved["augmentation"].astype(str) == augmentation]
        values = [
            subset.loc[subset["experiment"] == experiment, "improvement"].iloc[0]
            if experiment in set(subset["experiment"])
            else 0.0
            for experiment in experiments
        ]
        positions = [i + (offset - (len(augmentations) - 1) / 2) * width for i in x]
        ax.bar(
            positions,
            values,
            width=width,
            label=augmentation_display_label(augmentation),
            color=augmentation_color(augmentation),
            edgecolor="#2E3440",
            linewidth=0.7,
        )

    ax.axhline(0, color="#2E3440", linewidth=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(experiments)
    ax.set_ylabel("Test accuracy improvement over no augmentation")
    ax.set_title("Augmentation Improvement over Baseline")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Augmentation")

    save_current_figure(FIGURES_DIR / "baseline_improvement.png")


def plot_test_accuracy_heatmap(summary: pd.DataFrame) -> None:
    models = [m for m in MODEL_ORDER if m in set(summary["model"].astype(str))]
    n_cols = len(models)

    fig, axes = plt.subplots(1, n_cols, figsize=(5.5 * n_cols, 4.5), squeeze=False)
    max_acc = summary["mean_test_acc"].max()

    for ax, model in zip(axes.flatten(), models):
        model_df = summary[summary["model"].astype(str) == model]
        heatmap_data = model_df.pivot_table(
            index="augmentation",
            columns="k",
            values="mean_test_acc",
            observed=True,
        ).reindex(ordered_augmentations(model_df))

        image = ax.imshow(heatmap_data, cmap="YlGnBu", vmin=0, vmax=max_acc)
        ax.set_title(model)
        ax.set_xlabel("k")
        ax.set_ylabel("Augmentation")
        ax.set_xticks(range(len(heatmap_data.columns)))
        ax.set_xticklabels(heatmap_data.columns)
        ax.set_yticks(range(len(heatmap_data.index)))
        ax.set_yticklabels([augmentation_display_label(label) for label in heatmap_data.index.astype(str)])

        for y in range(heatmap_data.shape[0]):
            for x in range(heatmap_data.shape[1]):
                value = heatmap_data.iloc[y, x]
                if pd.notna(value):
                    ax.text(x, y, f"{value:.3f}", ha="center", va="center", fontsize=9)

    fig.colorbar(image, ax=axes.ravel().tolist(), label="Test accuracy", shrink=0.85)
    fig.suptitle("Test Accuracy Heatmap", fontsize=16, y=1.02)
    save_current_figure(FIGURES_DIR / "test_accuracy_heatmap.png", tight_layout=False)


def main() -> None:
    summary_df = load_summary_metrics(METRICS_DIR)
    summary = aggregate_summary(summary_df)
    epoch_metrics = load_epoch_metrics(summary_df, METRICS_DIR)

    print(f"Loaded {len(summary_df)} experiment summaries.")
    print(f"Loaded epoch metrics for {epoch_metrics['source_csv'].nunique()} runs.")

    plot_test_accuracy_by_augmentation(summary)
    plot_test_accuracy_vs_k(summary)
    plot_val_accuracy_curves(epoch_metrics)
    plot_train_accuracy_curves(epoch_metrics)
    plot_baseline_improvement(summary)
    plot_test_accuracy_heatmap(summary)

    print(f"Saved 6 comparison figures to {FIGURES_DIR}.")


if __name__ == "__main__":
    main()
