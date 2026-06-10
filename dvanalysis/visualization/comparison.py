"""Method comparison scatter plots (e.g. SMS-avg vs RPCA).

Extracted from scripts/fig_kotliar_vs_rpca.py — only the reusable plotting logic.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


def plot_method_scatter(
    x: np.ndarray,
    y: np.ndarray,
    *,
    colour: str = "#2166AC",
    label: str = "",
    ax: Optional[plt.Axes] = None,
    xlabel: str = "Method A",
    ylabel: str = "Method B",
    title: str = "",
    show_identity: bool = True,
    show_pearson: bool = True,
    point_size: float = 18,
    alpha: float = 0.7,
) -> plt.Axes:
    """Scatter plot comparing two methods on the same biomarker.

    Parameters
    ----------
    x, y : matched arrays of biomarker values from two methods.
    colour : scatter point colour.
    label : legend label.
    ax : existing Axes (creates new if None).
    xlabel, ylabel : axis labels.
    title : panel title.
    show_identity : draw y=x line.
    show_pearson : annotate Pearson r.
    point_size : marker size.
    alpha : marker transparency.

    Returns
    -------
    matplotlib Axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    valid = np.isfinite(x) & np.isfinite(y)
    xv, yv = x[valid], y[valid]

    ax.scatter(xv, yv, s=point_size, alpha=alpha, color=colour,
               edgecolors="none", zorder=3, label=label)

    if show_identity and len(xv) > 0:
        lo = min(xv.min(), yv.min())
        hi = max(xv.max(), yv.max())
        margin = (hi - lo) * 0.05 if hi > lo else 1.0
        lims = [lo - margin, hi + margin]
        ax.plot(lims, lims, color="#999999", ls="--", lw=0.8, zorder=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

    if show_pearson and len(xv) >= 3:
        from scipy.stats import pearsonr
        r, p = pearsonr(xv, yv)
        ax.text(0.03, 0.95, f"r={r:.2f} (n={len(xv)})",
                transform=ax.transAxes, fontsize=7, color=colour, va="top")

    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title, fontsize=9, color="#333333")

    return ax


def plot_method_comparison_grid(
    pairs: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
    *,
    panel_titles: Optional[Dict[str, str]] = None,
    panel_order: Optional[Sequence[str]] = None,
    vessel_configs: Optional[Dict[str, dict]] = None,
    xlabel_template: str = "RPCA {bio}",
    ylabel_template: str = "SMS-avg {bio}",
    biomarker_label: str = "",
    suptitle: str = "",
    figsize: Tuple[float, float] = (8, 7.5),
) -> plt.Figure:
    """Plot a 2x2 grid of method comparison scatter panels.

    Parameters
    ----------
    pairs : dict mapping panel_key -> {vessel_label -> (x_values, y_values)}.
    panel_titles : display title per panel_key.
    panel_order : ordering of panels in the grid.
    vessel_configs : dict mapping vessel_label -> {"color": str, "label": str}.
    xlabel_template, ylabel_template : axis label templates ({bio} is replaced).
    biomarker_label : human-readable biomarker name.
    suptitle : figure title.
    figsize : figure size.

    Returns
    -------
    matplotlib Figure.
    """
    if panel_titles is None:
        panel_titles = {"best": "Best cycle", "cycle0": "Cycle 1",
                        "cycle1": "Cycle 2", "cycle2": "Cycle 3"}
    if panel_order is None:
        panel_order = list(pairs.keys())[:4]
    if vessel_configs is None:
        vessel_configs = {
            "artery": dict(color="#2166AC", label="Artery"),
            "vein": dict(color="#C0392B", label="Vein"),
        }

    n_panels = len(panel_order)
    n_cols = min(n_panels, 2)
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()

    xlabel = xlabel_template.replace("{bio}", biomarker_label)
    ylabel = ylabel_template.replace("{bio}", biomarker_label)

    for idx, panel_key in enumerate(panel_order):
        ax = axes_flat[idx]
        vessel_pairs = pairs.get(panel_key, {})

        for vessel_label, vcfg in vessel_configs.items():
            xy = vessel_pairs.get(vessel_label)
            if xy is None:
                continue
            x_vals, y_vals = xy
            if len(x_vals) == 0:
                continue

            plot_method_scatter(
                x_vals, y_vals,
                colour=vcfg["color"], label=vcfg["label"],
                ax=ax, xlabel=xlabel, ylabel=ylabel,
                title=panel_titles.get(panel_key, panel_key),
                show_identity=True, show_pearson=True,
            )

    for idx in range(len(panel_order), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(handles),
                   fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))

    if suptitle:
        fig.suptitle(suptitle, fontsize=11, color="#333333", y=0.98)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig
