"""Visualization utilities for thesis figures.

All plots are saved as both PDF (for LaTeX) and PNG (for quick preview).
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .config import FIGURES_DIR
from .metrics import ClassificationMetrics, RegressionMetrics
from .benchmarking import aggregate_fold_metrics

logger = logging.getLogger(__name__)

THESIS_STYLE = {
    "figure.figsize": (8, 5),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


def apply_thesis_style() -> None:
    """Apply consistent plot style for thesis figures."""
    plt.rcParams.update(THESIS_STYLE)
    sns.set_palette("colorblind")


def save_figure(fig: plt.Figure, name: str, subdir: str = "") -> None:
    """Save figure as PDF and PNG."""
    out_dir = FIGURES_DIR / subdir if subdir else FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    for ext in ["pdf", "png"]:
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path)
        logger.info("Saved %s", path)
    plt.close(fig)


def plot_model_comparison_bar(
    results: dict[str, list[ClassificationMetrics]],
    metric_name: str = "accuracy",
    title: str = "Model Comparison",
    filename: str = "model_comparison",
    subdir: str = "",
) -> None:
    """Bar chart comparing models across folds with error bars.

    Args:
        results: Dict of {model_name: list of fold metrics}.
        metric_name: Which metric to plot (must be in ClassificationMetrics).
        title: Plot title.
        filename: Output filename (without extension).
        subdir: Subdirectory under figures/.
    """
    apply_thesis_style()

    model_names = list(results.keys())
    means, stds = [], []

    for name in model_names:
        values = [getattr(m, metric_name) for m in results[name] if getattr(m, metric_name) is not None]
        means.append(np.mean(values))
        stds.append(np.std(values))

    fig, ax = plt.subplots()
    x = np.arange(len(model_names))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=sns.color_palette("colorblind"))

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=30, ha="right")
    ax.set_ylabel(metric_name.replace("_", " ").title())
    ax.set_title(title)
    ax.set_ylim(0, max(means) * 1.2 if means else 1)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.01,
            f"{mean:.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    save_figure(fig, filename, subdir)


def plot_learning_curves(
    curve_results: dict[str, dict[float, list]],
    metric_name: str = "accuracy",
    title: str = "Learning Curves: Classical vs Quantum",
    filename: str = "learning_curves",
    subdir: str = "",
) -> None:
    """Learning curve plot showing performance vs training set size.

    This is the key plot for demonstrating QML advantages with limited data.

    Args:
        curve_results: Dict of {model_name: {fraction: [fold_metrics]}}.
        metric_name: Metric to plot on y-axis.
        title: Plot title.
        filename: Output filename.
        subdir: Subdirectory under figures/.
    """
    apply_thesis_style()
    fig, ax = plt.subplots()
    palette = sns.color_palette("colorblind", n_colors=len(curve_results))

    for idx, (model_name, frac_metrics) in enumerate(curve_results.items()):
        fracs = sorted(frac_metrics.keys())
        means, stds = [], []

        for frac in fracs:
            values = [
                getattr(m, metric_name)
                for m in frac_metrics[frac]
                if getattr(m, metric_name) is not None
            ]
            means.append(np.mean(values))
            stds.append(np.std(values))

        means_arr = np.array(means)
        stds_arr = np.array(stds)
        fracs_pct = [f * 100 for f in fracs]

        ax.plot(fracs_pct, means_arr, "o-", label=model_name, color=palette[idx])
        ax.fill_between(
            fracs_pct,
            means_arr - stds_arr,
            means_arr + stds_arr,
            alpha=0.15,
            color=palette[idx],
        )

    ax.set_xlabel("Training Set Size (%)")
    ax.set_ylabel(metric_name.replace("_", " ").title())
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_figure(fig, filename, subdir)


def plot_training_time_comparison(
    results: dict[str, list],
    title: str = "Training Time Comparison",
    filename: str = "training_time",
    subdir: str = "",
) -> None:
    """Bar chart comparing training times (log scale)."""
    apply_thesis_style()

    model_names = list(results.keys())
    means, stds = [], []

    for name in model_names:
        times = [m.training_time_seconds for m in results[name]]
        means.append(np.mean(times))
        stds.append(np.std(times))

    fig, ax = plt.subplots()
    x = np.arange(len(model_names))
    ax.bar(x, means, yerr=stds, capsize=5, color=sns.color_palette("colorblind"))

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=30, ha="right")
    ax.set_ylabel("Training Time (seconds)")
    ax.set_title(title)
    ax.set_yscale("log")

    save_figure(fig, filename, subdir)


def generate_comparison_table(
    results: dict[str, list],
    task_type: str = "classification",
) -> str:
    """Generate a LaTeX-ready comparison table.

    Returns:
        String containing LaTeX table code.
    """
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")

    if task_type == "classification":
        lines.append(r"\begin{tabular}{lccccc}")
        lines.append(r"\toprule")
        lines.append(r"Model & Accuracy & F1 (macro) & AUC-ROC & Train Time (s) & Parameters \\")
        lines.append(r"\midrule")

        for name, fold_metrics in results.items():
            agg = aggregate_fold_metrics(fold_metrics)
            acc = agg.get("accuracy", {"mean": 0, "std": 0})
            f1 = agg.get("f1_macro", {"mean": 0, "std": 0})
            auc = agg.get("auc_roc", {"mean": 0, "std": 0})
            tt = agg.get("training_time_s", {"mean": 0, "std": 0})
            n_params = fold_metrics[0].n_parameters or "N/A"

            lines.append(
                f"{name} & {acc['mean']:.4f}$\\pm${acc['std']:.4f} "
                f"& {f1['mean']:.4f}$\\pm${f1['std']:.4f} "
                f"& {auc['mean']:.4f}$\\pm${auc['std']:.4f} "
                f"& {tt['mean']:.2f} & {n_params} \\\\"
            )

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
    else:
        lines.append(r"\begin{tabular}{lcccc}")
        lines.append(r"\toprule")
        lines.append(r"Model & RMSE & MAE & $R^2$ & Train Time (s) \\")
        lines.append(r"\midrule")

        for name, fold_metrics in results.items():
            agg = aggregate_fold_metrics(fold_metrics)
            rmse = agg.get("rmse", {"mean": 0, "std": 0})
            mae = agg.get("mae", {"mean": 0, "std": 0})
            r2 = agg.get("r2", {"mean": 0, "std": 0})
            tt = agg.get("training_time_s", {"mean": 0, "std": 0})

            lines.append(
                f"{name} & {rmse['mean']:.4f}$\\pm${rmse['std']:.4f} "
                f"& {mae['mean']:.4f}$\\pm${mae['std']:.4f} "
                f"& {r2['mean']:.4f}$\\pm${r2['std']:.4f} "
                f"& {tt['mean']:.2f} \\\\"
            )

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")

    lines.append(r"\caption{Classical vs Quantum Model Comparison}")
    lines.append(r"\label{tab:model_comparison}")
    lines.append(r"\end{table}")

    return "\n".join(lines)
