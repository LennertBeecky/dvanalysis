"""Bland-Altman plots for within-session cycle-to-cycle variability.

Extracted from statistics/bland_altman.py — only the reusable plotting logic.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_bland_altman_data(
    df: pd.DataFrame,
    biomarker_col: str,
    subject_col: str = "subject_id",
) -> Dict[str, object]:
    """Compute Bland-Altman data from a per-cycle parameter DataFrame.

    Parameters
    ----------
    df : DataFrame with one row per cycle per subject.
    biomarker_col : column name for the biomarker values.
    subject_col : column identifying subjects.

    Returns
    -------
    dict with keys: "means", "deviations", "loa_lower", "loa_upper",
        "mean_dev", "sd_dev".
    """
    sub = df[[subject_col, biomarker_col]].dropna(subset=[biomarker_col]).copy()
    subj_means = sub.groupby(subject_col)[biomarker_col].mean()
    sub = sub.merge(subj_means.rename("subj_mean"), on=subject_col)
    sub["deviation"] = sub[biomarker_col] - sub["subj_mean"]

    mean_dev = float(sub["deviation"].mean())
    sd_dev = float(sub["deviation"].std())
    loa_upper = mean_dev + 1.96 * sd_dev
    loa_lower = mean_dev - 1.96 * sd_dev

    return {
        "means": sub["subj_mean"].values,
        "deviations": sub["deviation"].values,
        "loa_lower": loa_lower,
        "loa_upper": loa_upper,
        "mean_dev": mean_dev,
        "sd_dev": sd_dev,
    }


def plot_bland_altman(
    means: np.ndarray,
    deviations: np.ndarray,
    loa_lower: float,
    loa_upper: float,
    *,
    colour: str = "#2166AC",
    ylabel: str = "",
    xlabel: str = "Within-subject mean",
    title: str = "",
    ax: Optional[plt.Axes] = None,
    show_loa_labels: bool = True,
) -> plt.Axes:
    """Plot a single Bland-Altman panel.

    Parameters
    ----------
    means : within-subject mean values (x-axis).
    deviations : cycle-to-mean deviations (y-axis).
    loa_lower, loa_upper : limits of agreement.
    colour : scatter point colour.
    ylabel, xlabel : axis labels.
    title : panel title.
    ax : existing Axes (creates new if None).
    show_loa_labels : annotate LoA values on the plot.

    Returns
    -------
    matplotlib Axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 3.5))

    ax.axhspan(loa_lower, loa_upper, alpha=0.05, color="#888888", zorder=0)
    ax.scatter(means, deviations, s=14, alpha=0.45, color=colour,
               edgecolors="none", zorder=3)
    ax.axhline(0, color="#333333", linewidth=0.8, zorder=2)
    ax.axhline(loa_upper, color="#999999", linewidth=0.7, linestyle="--", zorder=2)
    ax.axhline(loa_lower, color="#999999", linewidth=0.7, linestyle="--", zorder=2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=6)

    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8)
    if title:
        ax.set_title(title, fontsize=10, loc="left")

    if show_loa_labels:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        pad = 0.02 * (ylim[1] - ylim[0])
        ax.text(xlim[1], loa_upper + pad, f"+{loa_upper:.1f}",
                fontsize=6, color="#777777", va="bottom", ha="right")
        ax.text(xlim[1], loa_lower - pad, f"{loa_lower:.1f}",
                fontsize=6, color="#777777", va="top", ha="right")

    return ax


def plot_bland_altman_grid(
    ba_data: Dict[Tuple[str, str], dict],
    biomarkers: Sequence[str],
    biomarker_labels: Sequence[str],
    vessels: Sequence[str] = ("artery", "vein"),
    vessel_colours: Optional[Dict[str, str]] = None,
    vessel_panel_labels: Optional[Dict[str, str]] = None,
    figsize: Tuple[float, float] = (9, 7.5),
) -> plt.Figure:
    """Plot a grid of Bland-Altman panels (rows=biomarkers, cols=vessels).

    Parameters
    ----------
    ba_data : dict mapping (vessel, biomarker) -> output of compute_bland_altman_data.
    biomarkers : ordered list of biomarker keys.
    biomarker_labels : display labels for each biomarker.
    vessels : vessel types for columns.
    vessel_colours : colour per vessel type.
    vessel_panel_labels : display label per vessel type.
    figsize : figure size.

    Returns
    -------
    matplotlib Figure.
    """
    if vessel_colours is None:
        vessel_colours = {"artery": "#B2182B", "vein": "#2166AC"}
    if vessel_panel_labels is None:
        vessel_panel_labels = {"artery": "Artery", "vein": "Vein"}

    n_rows = len(biomarkers)
    n_cols = len(vessels)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for row, (bio, bio_label) in enumerate(zip(biomarkers, biomarker_labels)):
        for col, vessel in enumerate(vessels):
            ax = axes[row, col]
            data = ba_data.get((vessel, bio))
            if data is None:
                ax.set_visible(False)
                continue

            plot_bland_altman(
                data["means"], data["deviations"],
                data["loa_lower"], data["loa_upper"],
                colour=vessel_colours.get(vessel, "#333333"), ax=ax,
                ylabel=bio_label if col == 0 else "",
                xlabel="Within-subject mean" if row == n_rows - 1 else "",
                title=vessel_panel_labels.get(vessel, vessel) if row == 0 else "",
                show_loa_labels=True,
            )

    plt.tight_layout(h_pad=1.2, w_pad=1.0)
    return fig
