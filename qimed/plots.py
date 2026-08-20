"""Render the complete QIMED V2 comparison from outer-fold results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS = ("AUC-ROC", "AUC-PR", "Accuracy", "F1", "Precision", "Recall")
REPRESENTATIONS = ("classical", "hybrid")
LABELS = {"classical": "Classical baseline", "hybrid": "18-qubit XYZ hybrid"}
COLORS = {"classical": "#1F77B4", "hybrid": "#FF7F0E"}
PLOT_DPI = 360


def _means(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics.groupby(["representation", "model"], as_index=False)[list(METRICS)].mean()


def _values(means: pd.DataFrame, models: tuple[str, ...], representation: str, metric: str) -> np.ndarray:
    indexed = means.set_index(["representation", "model"])
    return np.asarray([
        indexed.loc[(representation, model), metric]
        if (representation, model) in indexed.index else np.nan
        for model in models
    ])


def render_metric_comparison(
    means: pd.DataFrame, models: tuple[str, ...], output: Path, metric: str = "AUC-PR"
) -> None:
    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 6))
    for offset, representation in enumerate(REPRESENTATIONS):
        bars = ax.bar(
            x + (offset - (len(REPRESENTATIONS) - 1) / 2) * width,
            _values(means, models, representation, metric),
            width,
            label=LABELS[representation],
            color=COLORS[representation],
        )
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8, rotation=90)
    ax.set_xticks(x, models, rotation=20, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"QIMED V2 outer-fold {metric} comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=PLOT_DPI)
    plt.close(fig)


def render_all_metrics(means: pd.DataFrame, models: tuple[str, ...], output: Path) -> None:
    if len(models) == 5:
        fig = plt.figure(figsize=(15, 10))
        grid = fig.add_gridspec(2, 6)
        plot_axes = [
            fig.add_subplot(grid[0, 0:2]),
            fig.add_subplot(grid[0, 2:4]),
            fig.add_subplot(grid[0, 4:6]),
            fig.add_subplot(grid[1, 1:3]),
            fig.add_subplot(grid[1, 3:5]),
        ]
        ylabel_panels = (0, 3)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(13, 11), sharey=True)
        plot_axes = list(np.asarray(axes).flat)
        ylabel_panels = (0, 2)
    x = np.arange(len(METRICS))
    width = 0.38
    indexed = means.set_index(["representation", "model"])
    for panel, (ax, model) in enumerate(zip(plot_axes, models)):
        for offset, representation in enumerate(REPRESENTATIONS):
            values = np.asarray([
                indexed.loc[(representation, model), metric]
                for metric in METRICS
            ])
            bars = ax.bar(
                x + (offset - (len(REPRESENTATIONS) - 1) / 2) * width,
                values,
                width,
                label=LABELS[representation],
                color=COLORS[representation],
            )
            ax.bar_label(
                bars,
                fmt="%.3f",
                padding=2,
                fontsize=9,
                fontweight="bold",
                rotation=90,
            )
        ax.text(
            -0.10,
            1.03,
            f"{chr(ord('a') + panel)})",
            transform=ax.transAxes,
            fontsize=17,
            fontweight="bold",
        )
        ax.set_title(model, fontsize=15, fontweight="bold", pad=10)
        ax.set_xticks(x, METRICS, rotation=30, ha="right", fontsize=11)
        ax.tick_params(axis="y", labelsize=11)
        for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
            label.set_fontweight("bold")
        ax.set_ylim(0.5, 1.0)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)
        if panel == len(models) - 1:
            ax.legend(
                loc="lower right",
                frameon=True,
                framealpha=0.9,
                prop={"size": 11, "weight": "bold"},
            )
        ax.set_box_aspect(0.82)
    for panel in ylabel_panels:
        plot_axes[panel].set_ylabel(
            "Score (axis starts at 0.5)", fontsize=12, fontweight="bold"
        )
    fig.suptitle(
        "Classical baseline vs 18-qubit XYZ hybrid — outer-fold performance",
        y=0.995,
        fontsize=17,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.2, w_pad=1.0)
    fig.savefig(output, dpi=PLOT_DPI)
    plt.close(fig)


def render_delta_bars(
    metrics: pd.DataFrame,
    models: tuple[str, ...],
    output: Path,
    representation: str,
) -> None:
    delta = _paired_deltas(metrics, models, representation)
    limit = max(float(np.nanmax(np.abs(delta.to_numpy()))) * 1.2, 0.01)
    fig, axes = plt.subplots(2, 2, figsize=(17, 10), sharex=True)
    colors = np.where(
        delta.to_numpy() >= 0, COLORS["hybrid"], COLORS["classical"]
    )
    for index, (ax, model) in enumerate(zip(axes.flat, models)):
        values = delta.loc[model].to_numpy(float)
        bars = ax.barh(METRICS, values, color=colors[index])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.bar_label(
            bars,
            labels=[f"{value:+.3f}" for value in values],
            padding=3,
            fontsize=9,
        )
        ax.set_title(model, fontsize=14)
        ax.set_xlim(-limit, limit)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle(
        f"Paired outer-fold delta: {LABELS[representation]} − Classical",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=PLOT_DPI)
    plt.close(fig)


def _paired_deltas(
    metrics: pd.DataFrame, models: tuple[str, ...], representation: str
) -> pd.DataFrame:
    baseline = metrics[metrics["representation"] == "classical"].set_index(
        ["outer_fold", "model"]
    )[list(METRICS)]
    selected = metrics[metrics["representation"] == representation].set_index(
        ["outer_fold", "model"]
    )[list(METRICS)]
    return selected.subtract(baseline).groupby("model").mean().reindex(models)


def render_delta_heatmap(
    metrics: pd.DataFrame, models: tuple[str, ...], output: Path, representation: str
) -> None:
    delta = _paired_deltas(metrics, models, representation).T
    limit = max(float(np.nanmax(np.abs(delta.to_numpy()))), 1e-6)
    fig, ax = plt.subplots(figsize=(12, 6))
    image = ax.imshow(delta, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(delta.columns)), delta.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(delta.index)), delta.index)
    for row in range(delta.shape[0]):
        for column in range(delta.shape[1]):
            ax.text(column, row, f"{delta.iloc[row, column]:+.3f}", ha="center", va="center")
    ax.set_title(f"Mean paired outer-fold delta: {LABELS[representation]} − Classical")
    fig.colorbar(image, ax=ax, label="Metric delta")
    fig.tight_layout()
    fig.savefig(output, dpi=PLOT_DPI)
    plt.close(fig)


def render_absolute_heatmap(
    metrics: pd.DataFrame,
    models: tuple[str, ...],
    output: Path,
    representation: str,
) -> None:
    """Show the absolute values behind one side of the paired-delta view."""
    values = (
        metrics[metrics["representation"] == representation]
        .groupby("model")[list(METRICS)]
        .mean()
        .reindex(models)
        .T
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    image = ax.imshow(values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(
        range(len(values.columns)), values.columns, rotation=25, ha="right"
    )
    ax.set_yticks(range(len(values.index)), values.index)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(
                column,
                row,
                f"{values.iloc[row, column]:.3f}",
                ha="center", va="center",
            )
    ax.set_title(f"{LABELS[representation]} outer-fold metric values")
    fig.colorbar(image, ax=ax, label=f"{LABELS[representation]} metric value")
    fig.tight_layout()
    fig.subplots_adjust(left=0.16)
    fig.savefig(output, dpi=PLOT_DPI)
    plt.close(fig)


def render_absolute_bars(
    metrics: pd.DataFrame,
    models: tuple[str, ...],
    output: Path,
    representation: str,
) -> None:
    values = (
        metrics[metrics["representation"] == representation]
        .groupby("model")[list(METRICS)]
        .mean()
        .reindex(models)
    )
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    x = np.arange(len(models))
    for ax, metric in zip(axes.flat, METRICS):
        bars = ax.bar(x, values[metric], color=COLORS[representation])
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
        ax.set_title(metric)
        ax.set_xticks(
            x, [name.replace(" ", "\n") for name in models], fontsize=8
        )
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"{LABELS[representation]} outer-fold metric values", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=PLOT_DPI)
    plt.close(fig)


def render_selection_frequencies(metrics: pd.DataFrame, output: Path) -> None:
    frequencies = (
        metrics[metrics["qubits"].notna()]
        .groupby(["qubits", "model"])
        .size()
        .unstack(fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(14, 5))
    frequencies.plot.bar(ax=ax)
    ax.set_ylabel("Selections across model × outer-fold decisions")
    ax.set_xlabel("Qubits")
    ax.set_title("Selected XYZ qubit budgets")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=PLOT_DPI)
    plt.close(fig)


def render_results(results_dir: Path) -> list[Path]:
    metrics = pd.read_parquet(results_dir / "outer_metrics.parquet")
    if metrics.empty:
        return []
    metrics["representation"] = np.where(
        metrics["qubits"].isna(), "classical", "hybrid"
    )
    models = tuple(dict.fromkeys(metrics["model"]))
    output_dir = results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_dir / "01_all_metrics.png",
        output_dir / "02_auc_pr_comparison.png",
        output_dir / "03_hybrid_paired_deltas.png",
        output_dir / "05_classical_metric_values.png",
        output_dir / "06_hybrid_metric_values.png",
        output_dir / "07_classical_metric_bars.png",
        output_dir / "08_hybrid_metric_bars.png",
        output_dir / "09_hybrid_paired_delta_bars.png",
    ]
    means = _means(metrics)
    render_all_metrics(means, models, outputs[0])
    render_metric_comparison(means, models, outputs[1])
    render_delta_heatmap(metrics, models, outputs[2], "hybrid")
    render_absolute_heatmap(metrics, models, outputs[3], "classical")
    render_absolute_heatmap(metrics, models, outputs[4], "hybrid")
    render_absolute_bars(metrics, models, outputs[5], "classical")
    render_absolute_bars(metrics, models, outputs[6], "hybrid")
    render_delta_bars(metrics, models, outputs[7], "hybrid")
    outputs.append(output_dir / "04_selection_frequencies.png")
    render_selection_frequencies(metrics, outputs[-1])
    return outputs
