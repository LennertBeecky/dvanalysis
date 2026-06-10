"""Per-cycle overlay plots aligned to flicker onset, with quality ranking.

Extracted from scripts/fig_cycle_quality.py — only the reusable plotting logic.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import plotly.graph_objects as go

from dvanalysis.domain import SegmentSignal, StimulusProtocol
from dvanalysis.preprocessing.result import PreprocessResult


def extract_cycle_trace(
    s_hat: np.ndarray,
    t_abs: np.ndarray,
    flicker_start_sec: float,
    pre_sec: float = 15.0,
    post_sec: float = 50.0,
    t_common: Optional[np.ndarray] = None,
    fs: float = 25.0,
) -> Optional[np.ndarray]:
    """Cut a single cycle from a trace, baseline-normalise, and resample.

    Parameters
    ----------
    s_hat : (T,) preprocessed trace
    t_abs : (T,) absolute time vector in seconds
    flicker_start_sec : flicker onset time in seconds
    pre_sec : seconds before flicker onset to include
    post_sec : seconds after flicker onset to include
    t_common : common time grid to resample onto (relative to flicker onset).
        If None, one is generated from -pre_sec to post_sec at *fs* Hz.
    fs : sampling rate (used only if t_common is None)

    Returns
    -------
    (len(t_common),) resampled trace, or None if not enough valid data.
    """
    if t_common is None:
        t_common = np.arange(-pre_sec, post_sec, 1.0 / fs)

    t_start = flicker_start_sec - pre_sec
    t_end = flicker_start_sec + post_sec

    mask = (t_abs >= t_start) & (t_abs < t_end)
    t_rel = t_abs[mask] - flicker_start_sec
    y = np.asarray(s_hat[mask], dtype=float)

    # Baseline normalise: subtract median of pre-flicker window
    bl_mask = t_rel < 0
    bl_vals = y[bl_mask]
    bl_vals = bl_vals[np.isfinite(bl_vals)]
    if len(bl_vals) == 0:
        return None
    y -= np.median(bl_vals)

    # Interpolate over NaN gaps
    valid = np.isfinite(y)
    if valid.sum() < 3:
        return None
    if (~valid).any():
        f = interp1d(t_rel[valid], y[valid], kind="linear",
                     bounds_error=False, fill_value=np.nan)
        y = f(t_rel)

    # Resample onto common time grid
    valid2 = np.isfinite(y)
    if valid2.sum() < 3:
        return None
    f2 = interp1d(t_rel[valid2], y[valid2], kind="linear",
                  bounds_error=False, fill_value=np.nan)
    return f2(t_common)


# Default colour schemes for artery/vein rank plots
RANK_STYLES = {
    "artery": {
        "best":   dict(color="#B2182B", ls="-",  label="Best cycle"),
        "middle": dict(color="#E08080", ls="--", label="Middle cycle"),
        "worst":  dict(color="#D4B5B5", ls=":",  label="Worst cycle"),
    },
    "vein": {
        "best":   dict(color="#2166AC", ls="-",  label="Best cycle"),
        "middle": dict(color="#67A9CF", ls="--", label="Middle cycle"),
        "worst":  dict(color="#B0C4DE", ls=":",  label="Worst cycle"),
    },
}


def plot_cycle_quality_overlay(
    traces_by_rank: Dict[str, List[np.ndarray]],
    t_common: np.ndarray,
    *,
    vessel_type: Literal["artery", "vein"] = "artery",
    flicker_dur_sec: float = 20.0,
    title: str = "",
    ylim: Optional[Tuple[float, float]] = (-3.5, 5.5),
    rank_styles: Optional[Dict[str, dict]] = None,
    figsize: Tuple[float, float] = (7, 4),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot population-level cycle overlays stratified by quality rank.

    Parameters
    ----------
    traces_by_rank : dict mapping rank label ("best", "middle", "worst")
        to a list of 1-D arrays on *t_common*.
    t_common : (N,) time grid relative to flicker onset (seconds).
    vessel_type : "artery" or "vein" (selects default colour scheme).
    flicker_dur_sec : duration of flicker for shading.
    title : plot title.
    ylim : y-axis limits, or None for auto.
    rank_styles : override default colour/linestyle per rank.
    figsize : figure size.
    ax : existing Axes to plot on (creates new figure if None).

    Returns
    -------
    matplotlib Figure.
    """
    if rank_styles is None:
        rank_styles = RANK_STYLES.get(vessel_type, RANK_STYLES["artery"])

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Flicker shading
    ax.axvspan(0, flicker_dur_sec, alpha=0.08, color="#888888", zorder=0)
    ax.axhline(0, color="#cccccc", linewidth=0.5, linestyle=":", zorder=1)
    ax.axvline(0, color="#999999", linewidth=0.6, linestyle="--", zorder=2)

    for rank in ("best", "middle", "worst"):
        arr_list = traces_by_rank.get(rank, [])
        if len(arr_list) == 0:
            continue

        mat = np.vstack(arr_list)
        n = mat.shape[0]
        mean = np.nanmean(mat, axis=0)
        sem = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(n)

        style = rank_styles.get(rank, dict(color="gray", ls="-", label=rank))
        ax.plot(t_common, mean,
                color=style["color"], linestyle=style["ls"], linewidth=1.4,
                label=f'{style["label"]} (n={n})', zorder=4)
        ax.fill_between(t_common, mean - sem, mean + sem,
                        color=style["color"], alpha=0.15, zorder=3)

    pre_sec = float(-t_common[0])
    post_sec = float(t_common[-1])
    ax.set_xlim(-pre_sec, post_sec)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xlabel("Time relative to flicker onset (s)", fontsize=9)
    ax.set_ylabel("% change from baseline", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.95,
              edgecolor="#dddddd", fancybox=False)

    if title:
        ax.set_title(title, fontsize=10, loc="left")

    return fig


# ---------------------------------------------------------------------------
# Three-cycle overlay comparing preprocessing methods (Plotly)
# ---------------------------------------------------------------------------

def _agg_median_masked(sig: SegmentSignal) -> np.ndarray:
    """Median across loci respecting mask."""
    X = np.asarray(sig.x, float).copy()
    M = np.asarray(sig.m, bool)
    X[~M] = np.nan
    return np.nanmedian(X, axis=1)


def _baseline_idx(sig: SegmentSignal, baseline_source: str = "protocol_global") -> np.ndarray:
    """Return sample indices for the baseline window."""
    prot = sig.protocol
    if baseline_source == "protocol_global" and getattr(prot, "global_baseline", None) is not None:
        b = prot.global_baseline
    else:
        b = prot.cycles[0].baseline
    t = np.asarray(sig.t, float)
    idx = np.where((t >= float(b.start_sec)) & (t < float(b.end_sec)))[0]
    if idx.size == 0:
        raise ValueError("Empty baseline window.")
    return idx


def _to_percent_1d(y_abs: np.ndarray, b0: float, mode: str = "delta_over_baseline") -> np.ndarray:
    if (not np.isfinite(b0)) or b0 == 0.0:
        return np.full_like(y_abs, np.nan, dtype=float)
    if mode == "delta_over_baseline":
        return 100.0 * ((y_abs - b0) / b0)
    if mode == "ratio":
        return 100.0 * (y_abs / b0)
    raise ValueError("mode must be 'delta_over_baseline' or 'ratio'.")


def _raw_percent_trace(
    sig: SegmentSignal,
    baseline_source: str = "protocol_global",
    percent_mode: str = "delta_over_baseline",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute raw median trace in percent change from baseline."""
    t = np.asarray(sig.t, float)
    y_abs = _agg_median_masked(sig)
    b_idx = _baseline_idx(sig, baseline_source=baseline_source)
    b0 = float(np.nanmedian(y_abs[b_idx]))
    y = _to_percent_1d(y_abs, b0, mode=percent_mode)
    return t, y


def _cycle_window_idx(sig: SegmentSignal, cycle_i: int) -> np.ndarray:
    """Return indices spanning baseline→recovery for one cycle."""
    t = np.asarray(sig.t, float)
    cyc = sig.protocol.cycles[cycle_i]
    start = float(cyc.baseline.start_sec) if cyc.baseline is not None else float(cyc.flicker.start_sec)
    end = float(cyc.recovery.end_sec)
    return np.where((t >= start) & (t < end))[0]


def _stack_cycles(
    sig: SegmentSignal,
    t: np.ndarray,
    y: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Slice and align cycles to flicker onset. Returns lists of (t_rel, y_cycle)."""
    out_t, out_y = [], []
    for i in range(len(sig.protocol.cycles)):
        idx = _cycle_window_idx(sig, i)
        if idx.size == 0:
            continue
        t0 = float(sig.protocol.cycles[i].flicker.start_sec)
        out_t.append(t[idx] - t0)
        out_y.append(y[idx])
    return out_t, out_y


def _median_cycle_curve(
    t_list: List[np.ndarray],
    y_list: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Median across cycles on a common grid (trim to shortest cycle)."""
    if len(y_list) == 0:
        return np.array([]), np.array([])
    L = min(len(y) for y in y_list)
    t_ref = t_list[0][:L]
    Y = np.stack([y[:L] for y in y_list], axis=0)
    y_med = np.nanmedian(Y, axis=0)
    return t_ref, y_med


# Default colours per method
METHOD_COLOURS = {
    "raw":            "#888888",
    "kotliar":        "#2166AC",
    "kotliar_smooth": "#67A9CF",
    "rpca":           "#B2182B",
}


def plot_three_cycles_method_comparison(
    sig: SegmentSignal,
    methods: Dict[str, Union[PreprocessResult, np.ndarray]],
    *,
    percent_mode: str = "delta_over_baseline",
    baseline_source: str = "protocol_global",
    include_raw: bool = True,
    method_colours: Optional[Dict[str, str]] = None,
    title: Optional[str] = None,
    flicker_dur_sec: float = 20.0,
    height: int = 600,
    template: str = "plotly_white",
) -> go.Figure:
    """Plot 3-cycle overlay for multiple preprocessing methods on one segment.

    For each method, individual cycles are shown faintly and the median cycle is bold.

    Parameters
    ----------
    sig : SegmentSignal
        The raw signal (used for the raw trace and protocol timing).
    methods : dict mapping method_name -> PreprocessResult or 1-D array (s_hat).
        The preprocessed outputs to overlay.
    percent_mode : "delta_over_baseline" or "ratio".
    baseline_source : "protocol_global" or "cycle0_baseline".
    include_raw : if True, also plot the raw median trace (converted to percent).
    method_colours : optional colour overrides per method name.
    title : plot title (auto-generated if None).
    flicker_dur_sec : flicker duration for vertical shading.
    height : figure height in pixels.
    template : Plotly template.

    Returns
    -------
    plotly Figure.
    """
    colours = dict(METHOD_COLOURS)
    if method_colours:
        colours.update(method_colours)

    # Collect (t, y, sig_for_protocol) per method
    series: Dict[str, Tuple[np.ndarray, np.ndarray, SegmentSignal]] = {}

    if include_raw:
        t_raw, y_raw = _raw_percent_trace(sig, baseline_source=baseline_source, percent_mode=percent_mode)
        series["raw"] = (t_raw, y_raw, sig)

    for method_name, out in methods.items():
        if isinstance(out, PreprocessResult):
            t_m = np.asarray(out.input_signal.t, float)
            y_m = np.asarray(out.s_hat, float)
            sig_m = out.input_signal
        elif isinstance(out, np.ndarray):
            t_m = np.asarray(sig.t, float)
            y_m = np.asarray(out, float)
            sig_m = sig
        else:
            continue
        series[method_name] = (t_m, y_m, sig_m)

    fig = go.Figure()

    for method_name, (t, y, sig_proto) in series.items():
        colour = colours.get(method_name, "#666666")
        t_list, y_list = _stack_cycles(sig_proto, t, y)

        # Individual cycles (faint)
        for ci, (tc, yc) in enumerate(zip(t_list, y_list)):
            fig.add_trace(go.Scatter(
                x=tc, y=yc, mode="lines",
                name=f"{method_name} cycle {ci+1}",
                line=dict(color=colour, width=1),
                opacity=0.25,
                showlegend=(ci == 0),
                legendgroup=method_name,
            ))

        # Median cycle (bold)
        t_med, y_med = _median_cycle_curve(t_list, y_list)
        if t_med.size > 0:
            fig.add_trace(go.Scatter(
                x=t_med, y=y_med, mode="lines",
                name=f"{method_name} (median)",
                line=dict(color=colour, width=3.5),
                opacity=1.0,
                legendgroup=method_name,
            ))

    # Flicker shading and zero line
    fig.add_vrect(x0=0, x1=flicker_dur_sec, fillcolor="rgba(255,215,0,0.12)", line_width=0)
    fig.add_vline(x=0.0, line_dash="dash", line_color="#999999", line_width=0.8)
    fig.add_hline(y=0.0, line_dash="dot", line_color="#666666", line_width=0.5)

    if title is None:
        meta = sig.meta if isinstance(sig.meta, dict) else {}
        sid = meta.get("subject_id", "?")
        sl = meta.get("segment_label", "?")
        title = f"{sid} — {sl} | 3-cycle overlay (aligned to flicker onset)"

    fig.update_layout(
        title=title,
        xaxis_title="Time relative to flicker onset (s)",
        yaxis_title="% change from baseline",
        hovermode="x unified",
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        template=template,
    )
    return fig
