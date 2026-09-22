"""Statistical analysis and visualization of all benchmark results.

Produces thesis-ready figures (PDF + PNG) in results/figures/ and
LaTeX tables in results/tables/.

Usage:
    python experiments/analyze_results.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qml_life_sciences.utils.config import LOGS_DIR, FIGURES_DIR, TABLES_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

THESIS_STYLE = {
    "figure.figsize": (9, 5.5),
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

PALETTE = sns.color_palette("colorblind")
MODEL_ORDER_DD = ["Random Forest", "XGBoost", "SVM (RBF)", "VQC", "QSVM"]
MODEL_ORDER_GE = ["Random Forest", "XGBoost", "SVM (RBF)", "MLP", "VQC", "QSVM"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_latest(pattern: str) -> pd.DataFrame:
    files = sorted(LOGS_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching: {LOGS_DIR / pattern}")
    path = files[-1]
    logger.info("Loading %s", path.name)
    return pd.read_csv(path)


def load_lc_checkpoints(use_case: str, dataset: str, model_type: str) -> pd.DataFrame:
    """Load all per-fraction checkpoint CSVs for a use case + model type."""
    exp_id = f"{use_case}__all_{model_type}__{dataset}"
    files = sorted(LOGS_DIR.glob(f"{exp_id}__lc_*__frac*.csv"))
    if not files:
        raise FileNotFoundError(f"No LC checkpoints for {exp_id}")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        if "fraction" not in df.columns:
            frac_str = [p for p in f.stem.split("__") if p.startswith("frac")][0]
            df["fraction"] = int(frac_str[4:]) / 100
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    logger.info("LC checkpoints (%s %s): %d rows from %d files",
                model_type, use_case, len(combined), len(files))
    return combined


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_figure(fig: plt.Figure, name: str, subdir: str = "") -> None:
    out_dir = FIGURES_DIR / subdir if subdir else FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        fig.savefig(out_dir / f"{name}.{ext}")
        logger.info("Saved %s/%s.%s", subdir or "figures", name, ext)
    plt.close(fig)


def mean_std(df: pd.DataFrame, col: str) -> tuple[float, float]:
    vals = df[col].dropna()
    return float(vals.mean()), float(vals.std())


def order_models(df: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    present = [m for m in order if m in df["model_name"].unique()]
    df = df.copy()
    df["model_name"] = pd.Categorical(df["model_name"], categories=present, ordered=True)
    return df.sort_values("model_name")


def wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Wilcoxon signed-rank test on paired fold scores (exact method).

    Paired because both models are evaluated on the same CV folds.
    Exact method is required for n=5 (asymptotic approximation is inaccurate).
    Returns (statistic, p_value).
    """
    diffs = a - b
    if np.all(diffs == 0):
        return 0.0, 1.0
    result = stats.wilcoxon(diffs, alternative="two-sided", method="exact")
    return float(result.statistic), float(result.pvalue)


# ---------------------------------------------------------------------------
# Figure 1-3: Model comparison bar charts
# ---------------------------------------------------------------------------

def plot_comparison_bars(df_classical: pd.DataFrame, df_quantum: pd.DataFrame,
                         model_order: list[str], use_case: str, dataset: str) -> None:
    plt.rcParams.update(THESIS_STYLE)
    combined = order_models(pd.concat([df_classical, df_quantum], ignore_index=True), model_order)
    n_classical = len([m for m in model_order if m in combined["model_name"].values
                       and m not in ["VQC", "QSVM"]])

    for metric, ylabel, fname in [
        ("accuracy", "Accuracy", f"{use_case}_accuracy_comparison"),
        ("f1_macro", "F1 Score (macro)", f"{use_case}_f1_comparison"),
        ("auc_roc", "AUC-ROC", f"{use_case}_aucroc_comparison"),
    ]:
        if combined[metric].isna().all():
            continue
        agg = (combined.groupby("model_name", observed=True)[metric]
               .agg(["mean", "std"]).reset_index())
        fig, ax = plt.subplots()
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(agg))]
        bars = ax.bar(agg["model_name"], agg["mean"], yerr=agg["std"],
                      capsize=5, color=colors, error_kw={"elinewidth": 1.5})

        for bar, row in zip(bars, agg.itertuples()):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (row.std or 0) + 0.005,
                    f"{row.mean:.3f}", ha="center", va="bottom", fontsize=9)

        ax.axvspan(-0.5, n_classical - 0.5, alpha=0.04, color="steelblue", label="Classical")
        ax.axvspan(n_classical - 0.5, len(agg) - 0.5, alpha=0.04, color="darkorange", label="Quantum")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Model")
        ax.set_title(f"{dataset.upper()} — {ylabel}: Classical vs Quantum")
        ax.set_ylim(0, min(1.0, agg["mean"].max() * 1.25))
        ax.legend(loc="lower right")
        ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=25, ha="right")
        save_figure(fig, fname, use_case)


# ---------------------------------------------------------------------------
# Figure 4: Training time (log scale)
# ---------------------------------------------------------------------------

def plot_training_time(df_classical: pd.DataFrame, df_quantum: pd.DataFrame,
                       model_order: list[str], use_case: str, dataset: str) -> None:
    plt.rcParams.update(THESIS_STYLE)
    combined = order_models(pd.concat([df_classical, df_quantum], ignore_index=True), model_order)
    agg = (combined.groupby("model_name", observed=True)["training_time_s"]
           .agg(["mean", "std"]).reset_index())

    fig, ax = plt.subplots()
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(agg))]
    ax.bar(agg["model_name"], agg["mean"], yerr=agg["std"],
           capsize=5, color=colors, error_kw={"elinewidth": 1.5})
    ax.set_yscale("log")
    ax.set_ylabel("Training Time (seconds, log scale)")
    ax.set_xlabel("Model")
    ax.set_title(f"{dataset.upper()} — Training Time Comparison")
    ax.grid(axis="y", alpha=0.3, which="both")
    plt.xticks(rotation=25, ha="right")
    save_figure(fig, f"{use_case}_training_time", use_case)


# ---------------------------------------------------------------------------
# Figure 5-6: Learning curves (real classical + quantum curves)
# ---------------------------------------------------------------------------

def plot_learning_curves(df_classical_lc: pd.DataFrame, df_quantum_lc: pd.DataFrame,
                         model_order: list[str], use_case: str, dataset: str,
                         metric: str = "accuracy") -> None:
    """Real learning curves for both classical and quantum models.

    Classical: % of full dataset. Quantum: % of 200-sample subsample.
    Both shown on same x-axis (fraction 10–100%); note this in the caption.
    """
    plt.rcParams.update(THESIS_STYLE)
    color_map = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(model_order)}
    classical_models = [m for m in model_order if m not in ["VQC", "QSVM"]]
    quantum_models = ["VQC", "QSVM"]

    fig, ax = plt.subplots()

    for model in classical_models:
        sub = df_classical_lc[df_classical_lc["model_name"] == model]
        if sub.empty:
            continue
        sub_agg = (sub.groupby("fraction")[metric]
                   .agg(["mean", "std"]).reset_index()
                   .sort_values("fraction"))
        fracs_pct = sub_agg["fraction"] * 100
        color = color_map.get(model, "gray")
        ax.plot(fracs_pct, sub_agg["mean"], "s--", color=color,
                label=model, linewidth=1.5, markersize=5, alpha=0.85)
        ax.fill_between(fracs_pct,
                        sub_agg["mean"] - sub_agg["std"],
                        sub_agg["mean"] + sub_agg["std"],
                        alpha=0.08, color=color)

    for model in quantum_models:
        sub = df_quantum_lc[df_quantum_lc["model_name"] == model]
        if sub.empty:
            continue
        sub_agg = (sub.groupby("fraction")[metric]
                   .agg(["mean", "std"]).reset_index()
                   .sort_values("fraction"))
        fracs_pct = sub_agg["fraction"] * 100
        color = color_map.get(model, "black")
        ax.plot(fracs_pct, sub_agg["mean"], "o-", color=color,
                label=model, linewidth=2, markersize=6)
        ax.fill_between(fracs_pct,
                        sub_agg["mean"] - sub_agg["std"],
                        sub_agg["mean"] + sub_agg["std"],
                        alpha=0.15, color=color)

    ax.set_xlabel("Training Set Size (%)")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(
        f"{dataset.upper()} — Learning Curves: Classical vs Quantum\n"
        r"(Classical: % of full dataset; Quantum: % of 200-sample subsample)"
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    save_figure(fig, f"{use_case}_learning_curves_{metric}", use_case)


# ---------------------------------------------------------------------------
# Figure 7: Radar chart
# ---------------------------------------------------------------------------

def plot_radar(df_classical: pd.DataFrame, df_quantum: pd.DataFrame,
               model_order: list[str], use_case: str, dataset: str) -> None:
    plt.rcParams.update(THESIS_STYLE)
    combined = pd.concat([df_classical, df_quantum], ignore_index=True)
    metrics = ["accuracy", "f1_macro", "auc_roc"]
    metric_labels = ["Accuracy", "F1 (macro)", "AUC-ROC"]
    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    models_present = [m for m in model_order if m in combined["model_name"].values]

    for i, model in enumerate(models_present):
        sub = combined[combined["model_name"] == model]
        values = [sub[m].mean() for m in metrics]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2,
                color=PALETTE[i % len(PALETTE)], label=model)
        ax.fill(angles, values, alpha=0.07, color=PALETTE[i % len(PALETTE)])

    ax.set_thetagrids(np.degrees(angles[:-1]), metric_labels)
    ax.set_ylim(0, 1)
    ax.set_title(f"{dataset.upper()} — Multi-Metric Radar", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    save_figure(fig, f"{use_case}_radar", use_case)


# ---------------------------------------------------------------------------
# Statistical tests — Wilcoxon signed-rank (paired, exact)
# ---------------------------------------------------------------------------

def run_statistical_tests(df_classical: pd.DataFrame, df_quantum: pd.DataFrame,
                           use_case: str) -> pd.DataFrame:
    """Paired Wilcoxon signed-rank test (exact) on fold-level scores.

    Paired because both models are evaluated on the same CV folds.
    Uses F1-macro to select best classical (not accuracy, which is misleading
    on imbalanced datasets). Tests all three primary metrics.
    """
    combined = pd.concat([df_classical, df_quantum], ignore_index=True)
    best_classical = (combined[~combined["model_name"].isin(["VQC", "QSVM"])]
                      .groupby("model_name")["f1_macro"].mean().idxmax())

    rows = []
    for quantum_model in ["VQC", "QSVM"]:
        for metric in ["accuracy", "f1_macro", "auc_roc"]:
            q = (combined[combined["model_name"] == quantum_model]
                 .sort_values("fold")[metric].values)
            c = (combined[combined["model_name"] == best_classical]
                 .sort_values("fold")[metric].values)
            if len(q) == 0 or len(c) == 0 or len(q) != len(c):
                continue
            stat, p = wilcoxon_paired(c, q)
            rows.append({
                "use_case": use_case,
                "classical_model": best_classical,
                "quantum_model": quantum_model,
                "metric": metric,
                "classical_mean": round(float(np.mean(c)), 4),
                "quantum_mean": round(float(np.mean(q)), 4),
                "wilcoxon_statistic": round(stat, 2),
                "p_value": round(p, 4),
                "significant_p05": p < 0.05,
                "test": "wilcoxon_signed_rank_exact",
            })
            logger.info(
                "[%s] %s vs %s | %s | classical=%.4f quantum=%.4f | p=%.4f %s",
                use_case, best_classical, quantum_model, metric,
                np.mean(c), np.mean(q), p,
                "(*)" if p < 0.05 else "",
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def generate_latex_table(df_classical: pd.DataFrame, df_quantum: pd.DataFrame,
                          model_order: list[str], use_case: str, dataset: str) -> str:
    combined = pd.concat([df_classical, df_quantum], ignore_index=True)
    models_present = [m for m in model_order if m in combined["model_name"].values]
    n_classical = len([m for m in models_present if m not in ["VQC", "QSVM"]])

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Model & Type & Accuracy & F1 (macro) & AUC-ROC & Train Time (s) & \#Params \\",
        r"\midrule",
    ]

    for i, model in enumerate(models_present):
        if i == n_classical:
            lines.append(r"\midrule")
        sub = combined[combined["model_name"] == model]
        mtype = sub["model_type"].iloc[0] if "model_type" in sub.columns else "—"
        acc_m, acc_s = mean_std(sub, "accuracy")
        f1_m, f1_s = mean_std(sub, "f1_macro")
        auc_m, auc_s = mean_std(sub, "auc_roc")
        tt_m, _ = mean_std(sub, "training_time_s")
        n_params = sub["n_parameters"].iloc[0] if "n_parameters" in sub.columns else "N/A"
        lines.append(
            f"{model} & {mtype} "
            f"& ${acc_m:.3f}\\pm{acc_s:.3f}$ "
            f"& ${f1_m:.3f}\\pm{f1_s:.3f}$ "
            f"& ${auc_m:.3f}\\pm{auc_s:.3f}$ "
            f"& {tt_m:.2f} "
            f"& {n_params} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{Classical vs Quantum Model Comparison — {dataset.upper()} "
        r"(mean $\pm$ std over 5-fold stratified CV)}}",
        rf"\label{{tab:{use_case}_comparison}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_use_case(use_case: str, dataset: str, classical_pattern: str,
                 quantum_pattern: str, model_order: list[str]) -> pd.DataFrame:
    logger.info("\n%s\nAnalysing: %s\n%s", "=" * 60, use_case.upper(), "=" * 60)

    df_classical = load_latest(classical_pattern)
    df_quantum = load_latest(quantum_pattern)
    df_classical_lc = load_lc_checkpoints(use_case, dataset, "classical")
    df_quantum_lc = load_lc_checkpoints(use_case, dataset, "quantum")

    plot_comparison_bars(df_classical, df_quantum, model_order, use_case, dataset)
    plot_training_time(df_classical, df_quantum, model_order, use_case, dataset)
    plot_learning_curves(df_classical_lc, df_quantum_lc, model_order, use_case, dataset, "accuracy")
    plot_learning_curves(df_classical_lc, df_quantum_lc, model_order, use_case, dataset, "f1_macro")
    plot_radar(df_classical, df_quantum, model_order, use_case, dataset)

    stats_df = run_statistical_tests(df_classical, df_quantum, use_case)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    latex = generate_latex_table(df_classical, df_quantum, model_order, use_case, dataset)
    (TABLES_DIR / f"{use_case}_comparison_table.tex").write_text(latex)
    logger.info("LaTeX table → results/tables/%s_comparison_table.tex", use_case)

    return stats_df


def main():
    plt.rcParams.update(THESIS_STYLE)
    sns.set_palette("colorblind")

    all_stats = []

    all_stats.append(run_use_case(
        use_case="drug_discovery",
        dataset="bbbp",
        classical_pattern="drug_discovery__all_classical__bbbp__*.csv",
        quantum_pattern="drug_discovery__all_quantum__bbbp__2*.csv",
        model_order=MODEL_ORDER_DD,
    ))

    all_stats.append(run_use_case(
        use_case="genomics",
        dataset="gene_expression",
        classical_pattern="genomics__all_classical__gene_expression__*.csv",
        quantum_pattern="genomics__all_quantum__gene_expression__2*.csv",
        model_order=MODEL_ORDER_GE,
    ))

    stats_df = pd.concat(all_stats, ignore_index=True)
    stats_path = TABLES_DIR / "statistical_tests.csv"
    stats_df.to_csv(stats_path, index=False)
    logger.info("Statistical tests → %s", stats_path)

    logger.info("\n=== STATISTICAL TEST SUMMARY ===")
    print(stats_df.to_string(index=False))

    logger.info("\nAll figures → results/figures/")
    logger.info("All tables  → results/tables/")


if __name__ == "__main__":
    main()
