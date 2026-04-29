"""Plotting helpers for the main EM figures used in the report."""

from __future__ import annotations

from pathlib import Path
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "athlete_recovery_matplotlib"))
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .mixture import EMResult


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Save a figure with tight layout defaults suitable for the report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_em_model_selection(result: EMResult) -> plt.Figure:
    """Recreate the main EM model-selection figure."""

    selection_plot = result.selection_table.copy()
    selection_plot["bic_lower"] = selection_plot["median_bic"] - selection_plot["q25_bic"]
    selection_plot["bic_upper"] = selection_plot["q75_bic"] - selection_plot["median_bic"]

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3), gridspec_kw={"width_ratios": [1.2, 0.9]})

    axes[0].errorbar(
        selection_plot["n_components"],
        selection_plot["median_bic"],
        yerr=[selection_plot["bic_lower"], selection_plot["bic_upper"]],
        fmt="o-",
        color="#1d3557",
        ecolor="#8d99ae",
        elinewidth=1.2,
        capsize=3,
        linewidth=2,
        label="Median BIC (IQR)",
    )
    axes[0].scatter(
        selection_plot["n_components"],
        selection_plot["best_bic"],
        color="#e76f51",
        s=35,
        label="Best BIC across starts",
        zorder=3,
    )
    axes[0].axvline(result.primary_k, color="#2a9d8f", linestyle="--", linewidth=1.5)
    axes[0].set_title("Primary EM model selection", pad=6)
    axes[0].set_xlabel("Number of profiles")
    axes[0].set_ylabel("BIC")
    axes[0].set_xticks(selection_plot["n_components"])
    axes[0].legend(frameon=False, loc="best")

    ari_plot = selection_plot.loc[selection_plot["n_components"] > 1].copy()
    colors = ["#2a9d8f" if k == result.primary_k else "#adb5bd" for k in ari_plot["n_components"]]
    axes[1].bar(ari_plot["n_components"], ari_plot["mean_ari"], color=colors, edgecolor="none")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("Repeated-fit stability", pad=6)
    axes[1].set_xlabel("Number of profiles")
    axes[1].set_ylabel("Mean pairwise agreement")
    axes[1].set_xticks(ari_plot["n_components"])
    for _, row in ari_plot.iterrows():
        axes[1].annotate(
            f"{row['mean_ari']:.2f}",
            (row["n_components"], row["mean_ari"]),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout(pad=0.5, w_pad=1.0)
    return fig


def plot_em_structure(result: EMResult) -> plt.Figure:
    """Plot the PCA map and the standardized cluster feature heatmap."""

    fig, axes = plt.subplots(1, 2, figsize=(9.1, 3.6), gridspec_kw={"width_ratios": [1.0, 1.15]})

    for cluster_label in result.cluster_label_order:
        cluster_frame = result.assignments.loc[result.assignments["cluster_label"] == cluster_label]
        axes[0].scatter(
            cluster_frame["pc1"],
            cluster_frame["pc2"],
            s=35 + 35 * cluster_frame["cluster_prob"],
            alpha=0.85,
            color=result.cluster_palette[cluster_label],
            label=cluster_label,
            edgecolor="white",
            linewidth=0.4,
        )
    axes[0].set_title("Primary EM profiles in PCA space", pad=6)
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend(frameon=False, loc="best")

    sns.heatmap(
        result.profile_heatmap_z,
        cmap="RdBu_r",
        center=0,
        linewidths=0.4,
        cbar_kws={"shrink": 0.75, "label": "Cluster mean (z-score)"},
        ax=axes[1],
    )
    axes[1].set_title("Cluster feature profile heatmap", pad=6)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")

    fig.tight_layout(pad=0.5, w_pad=0.8)
    return fig


def plot_onset_profiles(result: EMResult) -> plt.Figure:
    """Plot profile-specific fatigue, recovery, and workload trajectories around onset."""

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.1), sharex=True)
    specs = [
        ("y_t", "Fatigue around onset", "Fatigue"),
        ("recovery_score", "Recovery around onset", "Recovery score"),
        ("u_t", "Workload around onset", "Workload"),
    ]
    for ax, (variable, title, ylabel) in zip(axes, specs):
        sns.lineplot(
            data=result.cluster_onset_summary,
            x="rel_session",
            y=variable,
            hue="cluster_label",
            hue_order=result.cluster_label_order,
            palette=result.cluster_palette,
            linewidth=2,
            marker="o",
            ax=ax,
        )
        ax.set_title(title, pad=6)
        ax.set_xlabel("Sessions rel. onset")
        ax.set_ylabel(ylabel if ax is axes[0] else "")
        if ax is not axes[2]:
            ax.legend_.remove()
    axes[2].legend(frameon=False, loc="best")

    fig.tight_layout(pad=0.5, w_pad=0.8)
    return fig
