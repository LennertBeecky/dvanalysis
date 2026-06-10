"""Pipeline figure: fundus → raw signal → denoised + artefacts (3-column layout).

Publication-quality matplotlib figure for the methods section.
Style matches the per-cycle overlay figures.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from scipy.interpolate import interp1d

from dvanalysis.domain import SegmentSignal, StimulusProtocol, TimeWindow
from dvanalysis.preprocessing.result import PreprocessResult


def _interpolate_gaps(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Linear interpolation over NaN gaps so flicker traces look continuous."""
    valid = np.isfinite(y)
    if valid.sum() < 3 or valid.all():
        return y
    f = interp1d(t[valid], y[valid], kind="linear", bounds_error=False, fill_value=np.nan)
    return f(t)


def _locus_median_and_iqr(
    sig: SegmentSignal,
    t_start: float,
    t_end: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute locus-median trace with IQR for a time window."""
    mask = (sig.t >= t_start) & (sig.t < t_end)
    t_slice = sig.t[mask]
    x = sig.x[mask, :].copy()
    m = sig.m[mask, :]
    x[~m] = np.nan

    med = np.nanmedian(x, axis=1)
    q25 = np.nanpercentile(x, 25, axis=1)
    q75 = np.nanpercentile(x, 75, axis=1)

    return t_slice, med, q25, q75


def plot_pipeline_figure(
    sig: SegmentSignal,
    result: PreprocessResult,
    *,
    fundus_path: Optional[str] = None,
    cycle_index: int = 0,
    pre_sec: float = 15.0,
    vessel_colour: str = "#B2182B",
    artefact_colour: str = "#999999",
    figsize: Tuple[float, float] = (14, 4.5),
    save_path: Optional[str] = None,
    save_pdf: bool = False,
) -> plt.Figure:
    """Create the 3-column pipeline figure for one flicker cycle.

    Layout:
        A: Fundus  |  B: Raw signal  |  C: Denoised (top) + D: Artefact (bottom)
    """
    protocol = sig.protocol
    cyc = protocol.cycles[cycle_index]
    fl_start = float(cyc.flicker.start_sec)
    fl_end = float(cyc.flicker.end_sec)
    fl_dur = fl_end - fl_start

    bl_start = float(cyc.baseline.start_sec) if cyc.baseline is not None else fl_start - 30.0
    rec_end = float(cyc.recovery.end_sec)
    win_start = bl_start - pre_sec
    win_end = rec_end

    t_full = np.asarray(sig.t, float)
    win_mask = (t_full >= win_start) & (t_full < win_end)
    t_win = t_full[win_mask]

    # Shift all time arrays so flicker onset = 0
    t_offset = fl_start
    t_win = t_win - t_offset
    fl_start_rel = 0.0
    fl_end_rel = fl_dur
    win_start_rel = win_start - t_offset
    win_end_rel = win_end - t_offset

    # Panel B data: Raw locus-median with IQR, normalised to percent change
    t_raw, med_raw, q25_raw, q75_raw = _locus_median_and_iqr(sig, win_start, win_end)
    # Compute baseline median for percent conversion (from raw signal)
    bl_time_mask = (sig.t >= bl_start) & (sig.t < fl_start)
    x_raw = sig.x.copy()
    x_raw[~sig.m] = np.nan
    raw_bl_per_locus = np.nanmedian(x_raw[bl_time_mask, :], axis=0)
    raw_bl_safe = np.where((np.isfinite(raw_bl_per_locus)) & (raw_bl_per_locus != 0), raw_bl_per_locus, np.nan)
    # Convert raw per-locus to percent, then take median/IQR
    x_raw_pct = 100.0 * (x_raw - raw_bl_safe[np.newaxis, :]) / raw_bl_safe[np.newaxis, :]
    raw_win_mask = (sig.t >= win_start) & (sig.t < win_end)
    x_raw_pct_win = x_raw_pct[raw_win_mask, :]
    med_raw = np.nanmedian(x_raw_pct_win, axis=1)
    q25_raw = np.nanpercentile(x_raw_pct_win, 25, axis=1)
    q75_raw = np.nanpercentile(x_raw_pct_win, 75, axis=1)
    t_raw = sig.t[raw_win_mask] - t_offset
    med_raw_interp = _interpolate_gaps(t_raw, med_raw)
    q25_interp = _interpolate_gaps(t_raw, q25_raw)
    q75_interp = _interpolate_gaps(t_raw, q75_raw)

    # Panel C data: Denoised trace (median of per-locus percent change)
    if result.S_hat is not None:
        S_full = result.S_hat.copy()
        M_full = result.m_used
        S_full[~M_full] = np.nan
        bl_mask_full = (t_full >= bl_start) & (t_full < fl_start)
        bl_per_locus = np.nanmedian(S_full[bl_mask_full, :], axis=0)
        bl_safe = np.where((np.isfinite(bl_per_locus)) & (bl_per_locus != 0), bl_per_locus, np.nan)
        S_pct = 100.0 * (S_full - bl_safe[np.newaxis, :]) / bl_safe[np.newaxis, :]
        S_pct_win = S_pct[win_mask, :]
        s_med_pct = np.nanmedian(S_pct_win, axis=1)
        s_q25 = np.nanpercentile(S_pct_win, 25, axis=1)
        s_q75 = np.nanpercentile(S_pct_win, 75, axis=1)
        y_den_interp = _interpolate_gaps(t_win, s_med_pct)
        s_q25_interp = _interpolate_gaps(t_win, s_q25)
        s_q75_interp = _interpolate_gaps(t_win, s_q75)
    else:
        y_den_interp = _interpolate_gaps(t_win, result.s_hat[win_mask])
        s_q25_interp = y_den_interp
        s_q75_interp = y_den_interp

    # Panel D data: Artefact component with IQR across loci
    if result.A_hat is not None:
        A_full = result.A_hat.copy()
        A_full[~M_full] = np.nan
        A_pct = 100.0 * A_full / bl_safe[np.newaxis, :]
        A_pct_win = A_pct[win_mask, :]
        a_med = np.nanmedian(A_pct_win, axis=1)
        a_q25 = np.nanpercentile(A_pct_win, 25, axis=1)
        a_q75 = np.nanpercentile(A_pct_win, 75, axis=1)
        y_art_interp = _interpolate_gaps(t_win, a_med)
        a_q25_interp = _interpolate_gaps(t_win, a_q25)
        a_q75_interp = _interpolate_gaps(t_win, a_q75)
    else:
        y_art_interp = np.zeros_like(t_win)
        a_q25_interp = np.zeros_like(t_win)
        a_q75_interp = np.zeros_like(t_win)

    # =====================================================
    # Figure layout
    # =====================================================
    fig = plt.figure(figsize=figsize)

    outer = gridspec.GridSpec(1, 3, width_ratios=[2.2, 4, 4.5], wspace=0.30)

    ax_fundus = fig.add_subplot(outer[0, 0])
    ax_raw = fig.add_subplot(outer[0, 1])

    # C+D stacked, matching B's full height
    inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, 2],
                                             height_ratios=[1, 1], hspace=0.35)
    ax_den = fig.add_subplot(inner[0])
    ax_art = fig.add_subplot(inner[1])

    # Flicker shading colour (light grey for CVD safety)
    fl_colour = "#cccccc"
    fl_alpha = 0.25

    # ── Panel A: Fundus image (preserve aspect ratio, centre vertically) ──
    if fundus_path is not None:
        img = mpimg.imread(fundus_path)
        ax_fundus.imshow(img, aspect="equal")
        h, w = img.shape[:2]
        ax_fundus.set_xlim(0, w)
        ax_fundus.set_ylim(h, 0)
    ax_fundus.set_xticks([])
    ax_fundus.set_yticks([])
    ax_fundus.set_title(r"$\bf{A}$  DVA fundus recording", fontsize=9, loc="left")
    for spine in ax_fundus.spines.values():
        spine.set_edgecolor("#cccccc")

    # ── Panel B: Raw signal ──────────────────────────────
    ax = ax_raw

    ax.axvspan(fl_start_rel, fl_end_rel, alpha=fl_alpha, color=fl_colour, zorder=0)
    ax.axhline(0, color="#cccccc", linewidth=0.5, linestyle=":", zorder=1)
    ax.axvline(fl_start_rel, color="#aaaaaa", linewidth=0.6, linestyle="--", zorder=1)
    ax.axvline(fl_end_rel, color="#aaaaaa", linewidth=0.6, linestyle="--", zorder=1)

    valid_iqr = np.isfinite(q25_interp) & np.isfinite(q75_interp)
    ax.fill_between(t_raw, q25_interp, q75_interp, where=valid_iqr,
                    alpha=0.12, color="#333333", zorder=1, label="IQR across loci")

    valid_raw = np.isfinite(med_raw_interp)
    ax.plot(t_raw[valid_raw], med_raw_interp[valid_raw],
            color="#333333", linewidth=0.8, zorder=3, label="Median")

    ax.set_ylabel("% change from baseline", fontsize=9)
    ax.set_xlabel("Time relative to flicker onset (s)", fontsize=9)
    ax.set_xlim(win_start_rel, win_end_rel)
    ax.set_ylim(-10, 10)
    ax.legend(fontsize=6, loc="upper right", framealpha=0.95,
              edgecolor="#dddddd", fancybox=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7)

    ax.text(fl_dur / 2, ax.get_ylim()[1] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.04,
            "Flicker", ha="center", va="top", fontsize=8, color="#888888", style="italic")

    ax.set_title(r"$\bf{B}$  Raw vessel diameter trace", fontsize=9, loc="left")

    # ── Zoom lines from fundus to raw signal ────────────
    fig.canvas.draw()
    bbox_fundus = ax_fundus.get_position()
    bbox_raw = ax_raw.get_position()

    line_kwargs = dict(color="#999999", linewidth=0.6, linestyle="-", alpha=0.5,
                       transform=fig.transFigure, clip_on=False)

    fig.lines.append(plt.Line2D(
        [bbox_fundus.x1, bbox_raw.x0],
        [bbox_fundus.y1, bbox_raw.y1],
        **line_kwargs))

    fig.lines.append(plt.Line2D(
        [bbox_fundus.x1, bbox_raw.x0],
        [bbox_fundus.y0, bbox_raw.y0],
        **line_kwargs))

    # ── RPCA annotation — positioned after layout to align with "+" ────
    # Will be placed after C and D are drawn, see below

    # ── Panel C: Denoised response (with IQR) ──────────────
    ax = ax_den

    ax.axvspan(fl_start_rel, fl_end_rel, alpha=fl_alpha, color=fl_colour, zorder=0)
    ax.axhline(0, color="#cccccc", linewidth=0.5, linestyle=":", zorder=1)
    ax.axvline(fl_start_rel, color="#aaaaaa", linewidth=0.6, linestyle="--", zorder=1)
    ax.axvline(fl_end_rel, color="#aaaaaa", linewidth=0.6, linestyle="--", zorder=1)

    # IQR band across loci in percent space
    valid_s_iqr = np.isfinite(s_q25_interp) & np.isfinite(s_q75_interp)
    ax.fill_between(t_win, s_q25_interp, s_q75_interp, where=valid_s_iqr,
                    alpha=0.15, color=vessel_colour, zorder=1, label="IQR across loci")

    valid_den = np.isfinite(y_den_interp)
    ax.plot(t_win[valid_den], y_den_interp[valid_den],
            color=vessel_colour, linewidth=1.2, zorder=3, label="Median")

    ax.set_ylabel("% change", fontsize=8)
    ax.set_xlim(win_start_rel, win_end_rel)
    ax.tick_params(labelbottom=False, labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=6, loc="upper right", framealpha=0.95,
              edgecolor="#dddddd", fancybox=False)

    ax.set_title(r"$\bf{C}$  Signal ($\mathbf{S}$)", fontsize=9, loc="left")

    # ── Panel D: Sparse artefacts (with IQR) ─────────────
    ax = ax_art

    ax.axvspan(fl_start_rel, fl_end_rel, alpha=fl_alpha, color=fl_colour, zorder=0)
    ax.axhline(0, color="#cccccc", linewidth=0.5, linestyle=":", zorder=1)
    ax.axvline(fl_start_rel, color="#aaaaaa", linewidth=0.6, linestyle="--", zorder=1)
    ax.axvline(fl_end_rel, color="#aaaaaa", linewidth=0.6, linestyle="--", zorder=1)

    # IQR band
    valid_a_iqr = np.isfinite(a_q25_interp) & np.isfinite(a_q75_interp)
    ax.fill_between(t_win, a_q25_interp, a_q75_interp, where=valid_a_iqr,
                    alpha=0.15, color=artefact_colour, zorder=1, label="IQR across loci")

    valid_art = np.isfinite(y_art_interp)
    ax.plot(t_win[valid_art], y_art_interp[valid_art],
            color=artefact_colour, linewidth=0.6, alpha=0.7, zorder=3, label="Median")

    ax.set_ylabel("Artefact (%)", fontsize=8)
    ax.set_xlabel("Time relative to flicker onset (s)", fontsize=8)
    ax.set_xlim(win_start_rel, win_end_rel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=6, loc="upper right", framealpha=0.95,
              edgecolor="#dddddd", fancybox=False)

    ax.set_title(r"$\bf{D}$  Artefacts ($\mathbf{A}$)", fontsize=9, loc="left")

    # "+" between C and D, and RPCA arrow aligned at same y
    fig.canvas.draw()
    bbox_den = ax_den.get_position()
    bbox_art = ax_art.get_position()
    bbox_raw_pos = ax_raw.get_position()

    plus_y = (bbox_den.y0 + bbox_art.y1) / 2
    plus_x = (bbox_den.x0 + bbox_den.x1) / 2

    fig.text(plus_x, plus_y, "+", ha="center", va="center", fontsize=14,
             fontweight="bold", color="#555555")

    # RPCA arrow aligned with the "+" sign
    rpca_x = (bbox_raw_pos.x1 + bbox_den.x0) / 2
    fig.text(rpca_x, plus_y, r"$\longrightarrow$", ha="center", va="center",
             fontsize=16, color="#333333")
    fig.text(rpca_x, plus_y + 0.02, r"RPCA", ha="center", va="bottom",
             fontsize=9, fontweight="bold", color="#333333")

    # ── Save ─────────────────────────────────────────────
    if save_path is not None:
        fig.savefig(f"{save_path}.png", dpi=300, facecolor="white", bbox_inches="tight")
        if save_pdf:
            fig.savefig(f"{save_path}.pdf", facecolor="white", bbox_inches="tight")
        print(f"Saved: {save_path}.png" + (" and .pdf" if save_pdf else ""))

    return fig
