"""
Appendix analyses for the DVA manuscript.

Four independent analyses:
  1. plot_templates        -- arterial & venous response template gallery
  2. run_noise_stationarity -- baseline vs recovery noise comparison
  3. run_vasomotion_drift  -- RPCA/SMS robustness to slow vasomotion drift
  4. run_rank1_ablation    -- RPCA sensitivity to rank-1 violation

Usage:
    conda run -n dvanalysis python dvanalysis/examples/run_appendix_analyses.py

Output:
    dvanalysis/examples/appendix_output/
"""
import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from dvanalysis.domain import (
    Segment,
    SegmentSignal,
    TimeWindow,
    StimulusCycle,
    StimulusProtocol,
)
from dvanalysis.io import ImedosReader, DataReaderConfig
from dvanalysis.preprocessing.rpca_denoise import MyMethodRPCA, MyMethodConfig
from dvanalysis.validation.hybrid import build_hybrid, HybridBuilder
from dvanalysis.validation.metrics import compute_sdr, compute_biomarker_bias
from dvanalysis.validation.noise import extract_baseline_noise
from dvanalysis.validation.templates import (
    ArterialParams,
    VenousParams,
    arterial_template,
    venous_template,
    sample_arterial_params,
    sample_venous_params,
)

# ── Paths ───────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "Healthy_Volunteers"
OUT_DIR = Path(__file__).resolve().parent / "appendix_output"

# ── Protocol (same as run_hybrid_experiments.py) ────────────────────────────
PROTOCOL = StimulusProtocol(
    name="DVA_3cycle_custom",
    fs=25.0,
    global_baseline=TimeWindow("baseline", 0.0, 20.0),
    cycles=[
        StimulusCycle(
            index=i,
            baseline=TimeWindow("baseline", float(b), float(f)),
            flicker=TimeWindow("flicker", float(f), float(r)),
            recovery=TimeWindow("recovery", float(r), float(e)),
        )
        for i, (b, f, r, e) in enumerate([
            (20, 50, 70, 120),
            (120, 150, 170, 220),
            (220, 250, 270, 320),
        ])
    ],
)

# ── RPCA configuration ─────────────────────────────────────────────────────
SEED = 42
RPCA_LMB = 0.55
RPCA_GAMMA = 1000.0
RPCA_MAX_ITER = 70

RPCA_CFG = MyMethodConfig(
    rpca_lmb=RPCA_LMB,
    rpca_gamma=RPCA_GAMMA,
    rpca_max_iter=RPCA_MAX_ITER,
    rpca_tol_rel=1e-3,
    standardize=True,
    harmonize_output=True,
    harmonize_percent_mode="delta_over_baseline",
    harmonize_baseline_source="protocol_global",
    harmonize_aggregation_order="percent_then_aggregate",
    harmonize_baseline_per_locus=True,
    support_min_valid_frac=0.75,
    support_min_valid_abs=12,
    hampel_enable=True,
)

SEGMENT_LABELS = {"A1", "V3"}

# ── Matplotlib defaults ────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ═══════════════════════════════════════════════════════════════════════════
# 1. Template gallery
# ═══════════════════════════════════════════════════════════════════════════

def plot_templates(out_dir: Path) -> None:
    """Plot arterial and venous response templates, matching fig_cycle_quality style."""
    print("\n=== 1. Template gallery ===")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use cycle 0 only for display
    cycle0 = PROTOCOL.cycles[0]
    fl_start = cycle0.flicker.start_sec
    fl_end = cycle0.flicker.end_sec
    fl_dur = fl_end - fl_start

    # Time vector covering one cycle, aligned to flicker onset
    t_full = np.arange(0, 320, 1.0 / PROTOCOL.fs)
    cycle0_bl_start = cycle0.baseline.start_sec
    cycle0_rec_end = cycle0.recovery.end_sec
    idx_cycle = np.where((t_full >= cycle0_bl_start) & (t_full < cycle0_rec_end))[0]
    t_rel = t_full[idx_cycle] - fl_start

    # Arterial configs: highest dilation first in legend, constriction inversely
    # correlated (absolute). Streese normative: MC can be 1-3% even for small dilators.
    art_configs = [
        ArterialParams(dilation_amplitude=6.0, constriction_depth=1.0),
        ArterialParams(dilation_amplitude=4.5, constriction_depth=1.5),
        ArterialParams(dilation_amplitude=3.0, constriction_depth=2.0),
        ArterialParams(dilation_amplitude=2.0, constriction_depth=2.5),
        ArterialParams(dilation_amplitude=1.0, constriction_depth=3.0),
    ]

    # Venous configs: highest dilation first in legend
    ven_configs = [
        VenousParams(dilation_amplitude=7.0),
        VenousParams(dilation_amplitude=6.0),
        VenousParams(dilation_amplitude=5.0),
        VenousParams(dilation_amplitude=3.5),
        VenousParams(dilation_amplitude=2.0),
    ]

    Y_LIM = (-4.5, 7.5)

    # ── Arterial panel ──
    fig_art, ax = plt.subplots(figsize=(7, 4))

    ax.axvspan(0, fl_dur, alpha=0.08, color="#888888", zorder=0)
    ax.axhline(0, color="#cccccc", linewidth=0.5, linestyle=":", zorder=1)
    ax.axvline(0, color="#999999", linewidth=0.6, linestyle="--", zorder=2)
    ax.axvline(fl_dur, color="#999999", linewidth=0.6, linestyle="--", zorder=2)

    colors_art = ["#B2182B", "#D6604D", "#F4A582", "#FDDBC7", "#E0C8C8"]
    for i, p in enumerate(art_configs):
        y = arterial_template(t_full, PROTOCOL, p)
        label = f"MD={p.dilation_amplitude:.0f}%, MC={p.constriction_depth:.1f}%"
        ax.plot(t_rel, y[idx_cycle], color=colors_art[i], linewidth=1.4,
                label=label, zorder=4)

    ax.set_xlim(-15, 50)
    ax.set_ylim(Y_LIM)
    ax.set_xlabel("Time relative to flicker onset (s)", fontsize=9)
    ax.set_ylabel("% change from baseline", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.95,
              edgecolor="#dddddd", fancybox=False)
    ax.set_title("$\\bf{A}$  Artery", fontsize=10, loc="left")
    ax.text(fl_dur / 2, Y_LIM[1] - (Y_LIM[1] - Y_LIM[0]) * 0.04, "Flicker",
            ha="center", va="top", fontsize=8, color="#888888", style="italic")

    fig_art.tight_layout()
    fig_art.savefig(out_dir / "fig_templates_artery.png", dpi=200,
                    facecolor="white", bbox_inches="tight")
    plt.close(fig_art)

    # ── Venous panel ──
    fig_ven, ax = plt.subplots(figsize=(7, 4))

    ax.axvspan(0, fl_dur, alpha=0.08, color="#888888", zorder=0)
    ax.axhline(0, color="#cccccc", linewidth=0.5, linestyle=":", zorder=1)
    ax.axvline(0, color="#999999", linewidth=0.6, linestyle="--", zorder=2)
    ax.axvline(fl_dur, color="#999999", linewidth=0.6, linestyle="--", zorder=2)

    colors_ven = ["#2166AC", "#4393C3", "#92C5DE", "#D1E5F0", "#C8D8E8"]
    for i, p in enumerate(ven_configs):
        y = venous_template(t_full, PROTOCOL, p)
        label = f"MD={p.dilation_amplitude:.0f}%"
        ax.plot(t_rel, y[idx_cycle], color=colors_ven[i], linewidth=1.4,
                label=label, zorder=4)

    ax.set_xlim(-15, 50)
    ax.set_ylim(Y_LIM)
    ax.set_xlabel("Time relative to flicker onset (s)", fontsize=9)
    ax.set_ylabel("% change from baseline", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.95,
              edgecolor="#dddddd", fancybox=False)
    ax.set_title("$\\bf{B}$  Vein", fontsize=10, loc="left")
    ax.text(fl_dur / 2, Y_LIM[1] - (Y_LIM[1] - Y_LIM[0]) * 0.04, "Flicker",
            ha="center", va="top", fontsize=8, color="#888888", style="italic")

    fig_ven.tight_layout()
    fig_ven.savefig(out_dir / "fig_templates_vein.png", dpi=200,
                    facecolor="white", bbox_inches="tight")
    plt.close(fig_ven)

    # ── Combined two-panel (for manuscript appendix) ──
    fig, (ax_art, ax_ven) = plt.subplots(1, 2, figsize=(14, 4))

    for panel_ax, configs, colors, panel_letter, panel_name, make_label in [
        (ax_art, art_configs, colors_art, "A", "Artery",
         lambda p: f"MD={p.dilation_amplitude:.0f}%, MC={p.constriction_depth:.1f}%"),
        (ax_ven, ven_configs, colors_ven, "B", "Vein",
         lambda p: f"MD={p.dilation_amplitude:.0f}%"),
    ]:
        panel_ax.axvspan(0, fl_dur, alpha=0.08, color="#888888", zorder=0)
        panel_ax.axhline(0, color="#cccccc", linewidth=0.5, linestyle=":", zorder=1)
        panel_ax.axvline(0, color="#999999", linewidth=0.6, linestyle="--", zorder=2)
        panel_ax.axvline(fl_dur, color="#999999", linewidth=0.6, linestyle="--", zorder=2)

        for i, p in enumerate(configs):
            if hasattr(p, "constriction_depth"):
                y = arterial_template(t_full, PROTOCOL, p)
            else:
                y = venous_template(t_full, PROTOCOL, p)
            panel_ax.plot(t_rel, y[idx_cycle], color=colors[i], linewidth=1.4,
                          label=make_label(p), zorder=4)

        panel_ax.set_xlim(-15, 50)
        panel_ax.set_ylim(Y_LIM)
        panel_ax.set_xlabel("Time relative to flicker onset (s)", fontsize=9)
        panel_ax.set_ylabel("% change from baseline", fontsize=9)
        panel_ax.tick_params(labelsize=7)
        panel_ax.spines["top"].set_visible(False)
        panel_ax.spines["right"].set_visible(False)
        panel_ax.legend(fontsize=8, loc="upper right", framealpha=0.95,
                        edgecolor="#dddddd", fancybox=False)
        panel_ax.set_title(f"$\\bf{{{panel_letter}}}$  {panel_name}",
                           fontsize=10, loc="left")
        panel_ax.text(fl_dur / 2, Y_LIM[1] - (Y_LIM[1] - Y_LIM[0]) * 0.04,
                      "Flicker", ha="center", va="top", fontsize=8,
                      color="#888888", style="italic")

    fig.tight_layout()
    fig.savefig(out_dir / "fig_templates.png", dpi=200,
                facecolor="white", bbox_inches="tight")
    fig.savefig(out_dir / "fig_templates.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / 'fig_templates.png'}")
    print(f"  Saved: {out_dir / 'fig_templates.pdf'}")


# ═══════════════════════════════════════════════════════════════════════════
# 1b. Hybrid example decomposition
# ═══════════════════════════════════════════════════════════════════════════

def plot_hybrid_example(segments: Dict[str, Segment], out_dir: Path) -> None:
    """Plot a single hybrid sample decomposed into its components.

    Shows: (A) combined hybrid signal (locus median), (B) ground-truth response,
    (C) cardiac + noise component, (D) observation mask coverage.
    """
    print("\n=== 1b. Hybrid example ===")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pick the arterial segment with best flicker-period observation coverage
    seg = None
    seg_key = None
    best_min_cov = -1
    for k, s in segments.items():
        if s.vessel_type != "artery":
            continue
        m = np.asarray(s.signal.m, bool)
        t_seg = np.asarray(s.signal.t, float)
        min_cov = min(
            m[np.where((t_seg >= cyc.flicker.start_sec) & (t_seg < cyc.flicker.end_sec))[0]].mean()
            for cyc in PROTOCOL.cycles
        )
        if min_cov > best_min_cov:
            best_min_cov = min_cov
            seg = s
            seg_key = k
    if seg is None:
        print("  No arterial segment found, skipping.")
        return
    print(f"  Using segment {seg_key} (min flicker coverage {best_min_cov:.2f})")

    # Build one hybrid sample
    rng = np.random.default_rng(SEED)
    sig = seg.signal
    t = np.asarray(sig.t, float)

    from dvanalysis.validation.hybrid import build_hybrid

    params = ArterialParams(dilation_amplitude=3.5, constriction_depth=1.2)
    hs = build_hybrid(sig, "artery", params, cycle_jitter=True, rng=rng)

    # Locus-median of hybrid signal in percent change from baseline
    m_hybrid = np.asarray(hs.signal.m, bool)
    x_hybrid = np.asarray(hs.signal.x, float).copy()
    x_hybrid[~m_hybrid] = np.nan

    # Per-locus baseline from global baseline only (same as methods use)
    bl_global_idx = np.where(PROTOCOL.global_baseline.contains(t))[0]
    baselines = np.nanmedian(x_hybrid[bl_global_idx, :], axis=0)

    hybrid_pct = np.full(len(t), np.nan)
    for i_t in range(len(t)):
        valid = m_hybrid[i_t, :]
        if valid.sum() < 1:
            continue
        bl_v = baselines[valid]
        nonzero = bl_v != 0
        if nonzero.any():
            hybrid_pct[i_t] = np.nanmedian(100.0 * (x_hybrid[i_t, valid][nonzero] - bl_v[nonzero]) / bl_v[nonzero])

    gt = hs.ground_truth
    frac_obs = m_hybrid.sum(axis=1) / m_hybrid.shape[1]
    fl_dur = 20.0  # flicker duration

    def _shade_flicker(ax):
        for cyc in PROTOCOL.cycles:
            ax.axvspan(cyc.flicker.start_sec, cyc.flicker.end_sec,
                       alpha=0.08, color="#888888", zorder=0)
            ax.axvline(cyc.flicker.start_sec, color="#999999",
                       linewidth=0.5, linestyle="--", zorder=2)
            ax.axvline(cyc.flicker.end_sec, color="#999999",
                       linewidth=0.5, linestyle="--", zorder=2)

    # ── Full recording (330 s) ──
    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    _shade_flicker(ax)
    ax.axhline(0, color="#cccccc", linewidth=0.5, linestyle=":", zorder=1)
    ax.plot(t, hybrid_pct, color="#888888", linewidth=0.4, alpha=0.6,
            zorder=3, rasterized=True)
    ax.scatter(t[np.isfinite(hybrid_pct)], hybrid_pct[np.isfinite(hybrid_pct)],
               color="#888888", s=0.15, alpha=0.4, zorder=3, rasterized=True,
               label="Hybrid signal")
    gt_interp = gt.copy()
    v = np.isfinite(gt_interp)
    if v.sum() >= 2:
        gt_interp[~v] = np.interp(t[~v], t[v], gt[v])
    ax.plot(t, gt_interp, color="#B2182B", linewidth=1.2, label="Ground truth", zorder=4)
    # Set y-limits tightly around the ground truth
    gt_valid_full = gt_interp[np.isfinite(gt_interp)]
    if len(gt_valid_full) > 0:
        gt_margin_full = (np.nanmax(gt_valid_full) - np.nanmin(gt_valid_full)) * 0.6
        ax.set_ylim(np.nanmin(gt_valid_full) - gt_margin_full, np.nanmax(gt_valid_full) + gt_margin_full)
    ax.set_ylabel("% change from baseline", fontsize=9)
    ax.set_title("$\\bf{A}$  Hybrid signal (full recording)", fontsize=10, loc="left")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.95,
              edgecolor="#dddddd", fancybox=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7)

    ax = axes[1]
    # Binary rasterplot: loci sorted by observation rate (most-observed at top)
    locus_obs_rate_full = np.nanmean(m_hybrid, axis=0)
    sort_order_full = np.argsort(locus_obs_rate_full)[::-1]
    mask_sorted_full = m_hybrid[:, sort_order_full].T.astype(float)
    from matplotlib.colors import ListedColormap
    cmap_mask_full = ListedColormap(["#ffffff", "#444444"])
    ax.imshow(mask_sorted_full, aspect="auto", interpolation="none",
              cmap=cmap_mask_full, vmin=0, vmax=1,
              extent=[t[0], t[-1], mask_sorted_full.shape[0], 0])
    ax.set_ylabel("Locus", fontsize=9)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_title("$\\bf{B}$  Observation mask (dark = observed)", fontsize=10, loc="left")
    ax.tick_params(labelsize=7)

    for a in axes:
        a.set_xlim(0, t[-1])

    fig.tight_layout()
    fig.savefig(out_dir / "fig_hybrid_example_full.png", dpi=200,
                facecolor="white", bbox_inches="tight")
    fig.savefig(out_dir / "fig_hybrid_example_full.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / 'fig_hybrid_example_full.png'}")

    # ── Single cycle zoom (cycle 0) ──
    fl_start = PROTOCOL.cycles[0].flicker.start_sec
    show_start = fl_start - 15
    show_end = fl_start + 50
    idx = np.where((t >= show_start) & (t < show_end))[0]
    t_rel = t[idx] - fl_start

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    ax.axvspan(0, fl_dur, alpha=0.08, color="#888888", zorder=0)
    ax.axhline(0, color="#cccccc", linewidth=0.5, linestyle=":", zorder=1)
    ax.axvline(0, color="#999999", linewidth=0.6, linestyle="--", zorder=2)
    ax.axvline(fl_dur, color="#999999", linewidth=0.6, linestyle="--", zorder=2)
    ax.plot(t_rel, hybrid_pct[idx], color="#888888", linewidth=0.4, alpha=0.6,
            zorder=3, rasterized=True)
    valid_idx = np.isfinite(hybrid_pct[idx])
    ax.scatter(t_rel[valid_idx], hybrid_pct[idx][valid_idx],
               color="#888888", s=0.3, alpha=0.4, zorder=3, rasterized=True,
               label="Hybrid signal")
    def _interp_nan_local(tr, yr):
        v = np.isfinite(yr)
        if v.sum() < 2: return yr
        yo = yr.copy()
        yo[~v] = np.interp(tr[~v], tr[v], yr[v])
        return yo
    gt_local = _interp_nan_local(t_rel, gt[idx])
    ax.plot(t_rel, gt_local, color="#B2182B", linewidth=1.6, label="Ground truth",
            zorder=4)
    # Set y-limits tightly around the ground truth so the response shape is
    # clearly visible; outlier spikes from the hybrid signal remain visible
    # but are clipped, which is the desired effect.
    gt_valid = gt_local[np.isfinite(gt_local)]
    if len(gt_valid) > 0:
        gt_margin = (np.nanmax(gt_valid) - np.nanmin(gt_valid)) * 0.6
        ax.set_ylim(np.nanmin(gt_valid) - gt_margin, np.nanmax(gt_valid) + gt_margin)
    ax.set_ylabel("% change from baseline", fontsize=9)
    ax.set_title("$\\bf{A}$  Hybrid signal (cycle 1)", fontsize=10, loc="left")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.95,
              edgecolor="#dddddd", fancybox=False)
    ax.text(fl_dur / 2, ax.get_ylim()[1] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.06,
            "Flicker", ha="center", va="top", fontsize=8, color="#888888", style="italic")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7)

    ax = axes[1]
    # Binary rasterplot: loci sorted by observation rate (most-observed at top)
    mask_local = m_hybrid[idx, :]
    locus_obs_rate = np.nanmean(mask_local, axis=0)
    sort_order = np.argsort(locus_obs_rate)[::-1]  # most-observed first
    mask_sorted = mask_local[:, sort_order].T.astype(float)
    from matplotlib.colors import ListedColormap
    cmap_mask = ListedColormap(["#ffffff", "#444444"])
    ax.imshow(mask_sorted, aspect="auto", interpolation="none",
              cmap=cmap_mask, vmin=0, vmax=1,
              extent=[t_rel[0], t_rel[-1], mask_sorted.shape[0], 0])
    ax.set_ylabel("Locus", fontsize=9)
    ax.set_xlabel("Time relative to flicker onset (s)", fontsize=9)
    ax.set_title("$\\bf{B}$  Observation mask (dark = observed)", fontsize=10, loc="left")
    ax.tick_params(labelsize=7)

    for a in axes:
        a.set_xlim(-15, 50)

    fig.tight_layout()
    fig.savefig(out_dir / "fig_hybrid_example.png", dpi=200,
                facecolor="white", bbox_inches="tight")
    fig.savefig(out_dir / "fig_hybrid_example.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / 'fig_hybrid_example.png'}")

    # ── Method comparison on cycle 0 ──
    from dvanalysis.validation.runner import _sms_trace, _sms_avg_trace

    bl_window = PROTOCOL.global_baseline
    sms_trace = _sms_trace(hs.signal, bl_window)
    sms_avg_trace = _sms_avg_trace(hs.signal, bl_window)

    try:
        rpca_method = MyMethodRPCA(config=RPCA_CFG)
        rpca_result = rpca_method.run(hs.signal)
        rpca_trace = rpca_result.s_hat
    except Exception as e:
        print(f"  RPCA failed: {e}")
        rpca_trace = np.full(len(t), np.nan)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.axvspan(0, fl_dur, alpha=0.08, color="#888888", zorder=0)
    ax.axhline(0, color="#cccccc", linewidth=0.5, linestyle=":", zorder=1)
    ax.axvline(0, color="#999999", linewidth=0.6, linestyle="--", zorder=2)
    ax.axvline(fl_dur, color="#999999", linewidth=0.6, linestyle="--", zorder=2)

    # Hybrid signal (background)
    ax.plot(t_rel, hybrid_pct[idx], color="#cccccc", linewidth=0.3, alpha=0.5,
            zorder=2, rasterized=True)
    valid_h = np.isfinite(hybrid_pct[idx])
    ax.scatter(t_rel[valid_h], hybrid_pct[idx][valid_h],
               color="#cccccc", s=0.2, alpha=0.3, zorder=2, rasterized=True)

    # Interpolate NaN gaps for display (connect valid points with lines)
    def _interp_nan(t_arr, y_arr):
        valid = np.isfinite(y_arr)
        if valid.sum() < 2:
            return y_arr
        y_out = y_arr.copy()
        y_out[~valid] = np.interp(t_arr[~valid], t_arr[valid], y_arr[valid])
        return y_out

    # Ground truth (interpolate for display so it's visible during flicker)
    ax.plot(t_rel, _interp_nan(t_rel, gt[idx]), color="black", linewidth=2.0,
            label="Ground truth", zorder=7)

    # SMS
    ax.plot(t_rel, _interp_nan(t_rel, sms_trace[idx]), color="#4393C3",
            linewidth=1.3, linestyle="-", label="SMS", zorder=4)

    # SMS-avg
    ax.plot(t_rel, _interp_nan(t_rel, sms_avg_trace[idx]), color="#2166AC",
            linewidth=1.3, linestyle=":", label="SMS-avg", zorder=5)

    # RPCA
    ax.plot(t_rel, _interp_nan(t_rel, rpca_trace[idx]), color="#D6604D",
            linewidth=1.5, linestyle="--", label="RPCA", zorder=6)

    ax.set_xlim(-15, 50)
    # Set y-limits tightly around the ground truth
    gt_mc = _interp_nan(t_rel, gt[idx])
    gt_mc_valid = gt_mc[np.isfinite(gt_mc)]
    if len(gt_mc_valid) > 0:
        gt_mc_margin = (np.nanmax(gt_mc_valid) - np.nanmin(gt_mc_valid)) * 0.6
        ax.set_ylim(np.nanmin(gt_mc_valid) - gt_mc_margin, np.nanmax(gt_mc_valid) + gt_mc_margin)
    ax.set_ylabel("% change from baseline", fontsize=9)
    ax.set_xlabel("Time relative to flicker onset (s)", fontsize=9)
    ax.set_title("Method comparison on hybrid data (cycle 1)", fontsize=10, loc="left")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.95,
              edgecolor="#dddddd", fancybox=False)
    ax.text(fl_dur / 2, ax.get_ylim()[1] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.06,
            "Flicker", ha="center", va="top", fontsize=8, color="#888888", style="italic")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(out_dir / "fig_hybrid_method_comparison.png", dpi=200,
                facecolor="white", bbox_inches="tight")
    fig.savefig(out_dir / "fig_hybrid_method_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / 'fig_hybrid_method_comparison.png'}")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Noise stationarity
# ═══════════════════════════════════════════════════════════════════════════

def _lag1_autocorrelation(x: np.ndarray) -> float:
    """Lag-1 autocorrelation of a 1D array (NaN-aware)."""
    valid = np.isfinite(x)
    n = valid.sum()
    if n < 3:
        return float("nan")
    xv = x[valid]
    xv = xv - np.mean(xv)
    var = np.mean(xv ** 2)
    if var == 0:
        return float("nan")
    return float(np.mean(xv[:-1] * xv[1:]) / var)


def run_noise_stationarity(
    segments: Dict[str, Segment],
    protocol: StimulusProtocol,
    out_dir: Path,
) -> None:
    """Compare noise statistics between baseline and recovery periods."""
    print("\n=== 2. Noise stationarity ===")
    out_dir.mkdir(parents=True, exist_ok=True)

    bl_window = protocol.global_baseline  # 0-20 s
    rec_window = protocol.cycles[1].recovery  # recovery of cycle 2: 170-220 s

    rows = []
    try:
        from tqdm import tqdm
        seg_iter = tqdm(segments.items(), desc="Noise stationarity", unit="seg")
    except ImportError:
        seg_iter = segments.items()
    for seg_key, seg in seg_iter:
        sig = seg.signal
        t = np.asarray(sig.t, float)

        # Baseline noise
        noise_bl = extract_baseline_noise(sig, bl_window)  # (T_bl, P)
        # Recovery noise (use same function with recovery window)
        noise_rec = extract_baseline_noise(sig, rec_window)  # (T_rec, P)

        P = noise_bl.shape[1]
        for p in range(P):
            std_bl = float(np.nanstd(noise_bl[:, p]))
            std_rec = float(np.nanstd(noise_rec[:, p]))
            ac_bl = _lag1_autocorrelation(noise_bl[:, p])
            ac_rec = _lag1_autocorrelation(noise_rec[:, p])

            rows.append({
                "segment_key": seg_key,
                "locus": p,
                "std_baseline": std_bl,
                "std_recovery": std_rec,
                "std_ratio": std_rec / std_bl if std_bl > 0 else float("nan"),
                "autocorr_baseline": ac_bl,
                "autocorr_recovery": ac_rec,
            })

    df = pd.DataFrame(rows)

    # Paired Wilcoxon signed-rank test on std ratio
    ratios = df["std_ratio"].dropna()
    if len(ratios) > 5:
        stat, pval = stats.wilcoxon(ratios - 1.0)
        print(f"  Wilcoxon signed-rank test on std_ratio - 1:")
        print(f"    statistic = {stat:.1f}, p = {pval:.4g}")
        print(f"    median ratio = {ratios.median():.3f}")
    else:
        stat, pval = float("nan"), float("nan")
        print("  Too few data points for Wilcoxon test")

    # Save CSV
    csv_path = out_dir / "noise_stationarity_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Scatter plot: baseline std vs recovery std
    fig, ax = plt.subplots(figsize=(5, 5))
    valid = df["std_baseline"].notna() & df["std_recovery"].notna()
    x_vals = df.loc[valid, "std_baseline"].values
    y_vals = df.loc[valid, "std_recovery"].values

    ax.scatter(x_vals, y_vals, s=12, alpha=0.5, edgecolors="none")
    lim = max(x_vals.max(), y_vals.max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, label="Identity")
    ax.set_xlabel("Baseline noise std (a.u.)")
    ax.set_ylabel("Recovery noise std (a.u.)")
    ax.set_title("Noise stationarity: baseline vs recovery")
    ax.legend(fontsize=8)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_dir / "fig_noise_stationarity.png")
    plt.close(fig)
    print(f"  Saved: {out_dir / 'fig_noise_stationarity.png'}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Vasomotion drift
# ═══════════════════════════════════════════════════════════════════════════

def _sms_trace(sig: SegmentSignal, bl_window: TimeWindow, window_sec: float = 4.0) -> np.ndarray:
    """SMS: median across loci -> sliding median filter -> percent change."""
    x = sig.x.copy()
    x[~sig.m] = np.nan
    y_abs = np.nanmedian(x, axis=1)

    fs = float(sig.protocol.fs)
    win = max(1, int(round(window_sec * fs)))
    T = y_abs.size
    y_smooth = np.full(T, np.nan, dtype=float)
    for i in range(T):
        lo = max(0, i - win // 2)
        hi = min(T, i + win // 2 + 1)
        y_smooth[i] = np.nanmedian(y_abs[lo:hi])

    bl_idx = np.where(bl_window.contains(sig.t))[0]
    b0 = float(np.nanmedian(y_smooth[bl_idx]))
    if not np.isfinite(b0) or b0 == 0:
        return y_smooth
    return 100.0 * (y_smooth - b0) / b0


def run_vasomotion_drift(
    segments: Dict[str, Segment],
    protocol: StimulusProtocol,
    out_dir: Path,
) -> None:
    """Test sensitivity of hybrid results to slow vasomotion drift."""
    print("\n=== 3. Vasomotion drift ===")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    rpca = MyMethodRPCA(config=RPCA_CFG)
    drift_amplitudes = [0.0, 0.5, 1.0, 2.0]
    n_samples = 100

    # Collect all segments into a list for round-robin sampling
    seg_list = list(segments.values())

    rows = []
    try:
        from tqdm import tqdm
        drift_iter = tqdm(drift_amplitudes, desc="Vasomotion drift", unit="level")
    except ImportError:
        drift_iter = drift_amplitudes
    for drift_a in drift_iter:
        sdr_rpca_list = []
        sdr_sms_list = []
        bias_rpca_list = []
        bias_sms_list = []

        for i in range(n_samples):
            seg = seg_list[i % len(seg_list)]
            vessel_type = seg.vessel_type
            is_artery = vessel_type.lower().startswith("a")

            if is_artery:
                params = sample_arterial_params(rng)
            else:
                params = sample_venous_params(rng)

            hs = build_hybrid(seg.signal, vessel_type, params, rng, cycle_jitter=True)

            # Add drift to the hybrid signal
            t = np.asarray(hs.signal.t, float)
            drift = drift_a * np.sin(2.0 * np.pi * 0.05 * t)  # 0.05 Hz
            x_drifted = hs.signal.x.copy()
            m = hs.signal.m.copy()
            # Drift is in percent, convert to same scale as x
            bl_window = protocol.global_baseline
            bl_idx = np.where(bl_window.contains(t))[0]
            x_masked = x_drifted.copy()
            x_masked[~m] = np.nan
            baselines = np.nanmedian(x_masked[bl_idx, :], axis=0)  # (P,)
            # Add drift as percent of baseline
            x_drifted[m] += (drift[:, np.newaxis] * baselines[np.newaxis, :] / 100.0)[m]

            drifted_sig = SegmentSignal(
                t=t, x=x_drifted, m=m, protocol=protocol,
                units=hs.signal.units, meta=hs.signal.meta,
            )

            gt = hs.ground_truth

            # RPCA
            try:
                rpca_trace = rpca.run(drifted_sig).s_hat
                sdr_rpca = compute_sdr(gt, rpca_trace)
            except Exception:
                sdr_rpca = float("nan")

            # SMS
            sms_trace = _sms_trace(drifted_sig, bl_window)
            sdr_sms = compute_sdr(gt, sms_trace)

            # Biomarker bias (MD only, average across cycles)
            try:
                biases_rpca = compute_biomarker_bias(gt, rpca_trace, t, protocol, vessel_type)
                md_bias_rpca = np.nanmean([c.get("MD", float("nan")) for c in biases_rpca])
            except Exception:
                md_bias_rpca = float("nan")

            biases_sms = compute_biomarker_bias(gt, sms_trace, t, protocol, vessel_type)
            md_bias_sms = np.nanmean([c.get("MD", float("nan")) for c in biases_sms])

            sdr_rpca_list.append(sdr_rpca)
            sdr_sms_list.append(sdr_sms)
            bias_rpca_list.append(md_bias_rpca)
            bias_sms_list.append(md_bias_sms)

            if (i + 1) % 25 == 0:
                print(f"    {i + 1}/{n_samples}")

        rows.append({
            "drift_amplitude": drift_a,
            "sdr_rpca_median": float(np.nanmedian(sdr_rpca_list)),
            "sdr_rpca_q25": float(np.nanpercentile(sdr_rpca_list, 25)),
            "sdr_rpca_q75": float(np.nanpercentile(sdr_rpca_list, 75)),
            "sdr_sms_median": float(np.nanmedian(sdr_sms_list)),
            "sdr_sms_q25": float(np.nanpercentile(sdr_sms_list, 25)),
            "sdr_sms_q75": float(np.nanpercentile(sdr_sms_list, 75)),
            "md_bias_rpca_median": float(np.nanmedian(bias_rpca_list)),
            "md_bias_sms_median": float(np.nanmedian(bias_sms_list)),
        })

    df = pd.DataFrame(rows)
    csv_path = out_dir / "vasomotion_drift_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Line plot: drift amplitude vs SDR
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(
        df["drift_amplitude"], df["sdr_rpca_median"],
        yerr=[
            df["sdr_rpca_median"] - df["sdr_rpca_q25"],
            df["sdr_rpca_q75"] - df["sdr_rpca_median"],
        ],
        marker="o", capsize=4, linewidth=1.5, label="RPCA",
    )
    ax.errorbar(
        df["drift_amplitude"], df["sdr_sms_median"],
        yerr=[
            df["sdr_sms_median"] - df["sdr_sms_q25"],
            df["sdr_sms_q75"] - df["sdr_sms_median"],
        ],
        marker="s", capsize=4, linewidth=1.5, label="SMS",
    )
    ax.set_xlabel("Drift amplitude (% of baseline)")
    ax.set_ylabel("SDR (dB)")
    ax.set_title("Vasomotion drift sensitivity")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_dir / "fig_vasomotion_drift.png")
    plt.close(fig)
    print(f"  Saved: {out_dir / 'fig_vasomotion_drift.png'}")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Rank-1 ablation
# ═══════════════════════════════════════════════════════════════════════════

def _load_retained_subjects(path: Optional[Path] = None) -> Optional[Dict[str, set]]:
    """Load the per-segment-label set of retained subject IDs.

    Mirrors `_load_subjects` in run_nonstationary_hybrid.py so that the
    rank-1 ablation runs on the same 91 retained segments as the main
    hybrid evaluation and the clinical analysis.
    """
    if path is None:
        path = Path(__file__).resolve().parent / "cycle_quality_subjects.json"
    if not path.exists():
        return None
    with open(path) as f:
        raw = json.load(f)
    return {
        seg: {str(s) for s in ids}
        for seg, ids in raw.items()
        if not seg.startswith("_")
    }


def _filter_retained(segments: Dict[str, Segment]) -> Dict[str, Segment]:
    """Filter segments dict to the retained subjects (cycle_quality_subjects.json).

    Returns the input unchanged if the JSON file is absent.
    """
    retained = _load_retained_subjects()
    if retained is None:
        print("  WARNING: cycle_quality_subjects.json not found; using all segments.")
        return segments
    out: Dict[str, Segment] = {}
    for key, seg in segments.items():
        # key is "<subject_id>_<visit_id>_<segment_label>"
        try:
            subject_id, _visit_id, seg_label = key.rsplit("_", 2)
        except ValueError:
            continue
        if subject_id in retained.get(seg_label, set()):
            out[key] = seg
    n_artery = sum(1 for s in out.values() if s.vessel_type == "artery")
    n_vein = sum(1 for s in out.values() if s.vessel_type == "vein")
    print(f"  Filtered to retained: {len(out)} segments "
          f"(arteries: {n_artery}, veins: {n_vein})")
    return out


def run_rank1_ablation(
    segments: Dict[str, Segment],
    protocol: StimulusProtocol,
    out_dir: Path,
    variability_levels: Optional[List[float]] = None,
    out_suffix: str = "",
) -> None:
    """Test RPCA sensitivity to rank-1 violation via per-locus amplitude variability.

    Uses the same RPCA config, support gate, sample size and retained-subject
    filtering as the main hybrid experiment for consistency.  SMS is included
    as a control because it is inherently rank-1 agnostic (locus median
    then smooth).

    Parameters
    ----------
    variability_levels : optional list of floats
        Per-locus amplitude scaling std-dev levels (CV). If None, uses the
        default {0, 0.05, 0.10, 0.15, 0.20}. Override with e.g. [0.25, 0.40]
        when extending the grid.
    out_suffix : str
        Optional suffix appended to output CSV/figure filenames (e.g.
        "_ext" so the extension run does not overwrite the primary CSV).
    """
    print("\n=== 4. Rank-1 ablation ===")
    out_dir.mkdir(parents=True, exist_ok=True)

    from dvanalysis.validation.runner import _sms_trace

    rpca = MyMethodRPCA(config=RPCA_CFG)
    if variability_levels is None:
        variability_levels = [0.0, 0.05, 0.10, 0.15, 0.20]
    print(f"  Variability levels: {variability_levels}")
    n_configs = 10  # per segment, matching main experiment
    bl_window = protocol.global_baseline

    # Filter to retained segments so the ablation runs on the same population
    # used for the main hybrid evaluation and the clinical analysis (91 segments).
    segments = _filter_retained(segments)
    seg_list = list(segments.values())

    rows = []
    try:
        from tqdm import tqdm
        lv_iter = tqdm(variability_levels, desc="Rank-1 ablation", unit="level")
    except ImportError:
        lv_iter = variability_levels

    for lv in lv_iter:
        # Build hybrid samples directly (not via HybridBuilder) so we can
        # turn off cycle_jitter and isolate spatial heterogeneity. With
        # cycle_jitter=False the GT and the data share the un-jittered
        # template, so the σ = 0 baseline is the maximally idealised case.
        rng = np.random.default_rng(SEED)

        sdr_rpca, sdr_sms = [], []
        md_bias_rpca, md_bias_sms = [], []

        for seg in seg_list:
            vessel_type = seg.vessel_type
            sampler = (sample_arterial_params if vessel_type == "artery"
                       else sample_venous_params)
            samples = []
            for _ in range(n_configs):
                params = sampler(rng)
                hs = build_hybrid(
                    seg.signal, vessel_type, params, rng=rng,
                    cycle_jitter=False, locus_variability=lv,
                    locus_jitter_samples=2,
                )
                samples.append(hs)

            for hs in samples:
                t = np.asarray(hs.signal.t, float)
                m = np.asarray(hs.signal.m, bool)

                # Ground truth: always the clean template (consistent across
                # all sigma levels, not mask-dependent)
                gt = hs.params["gt_pct_clean"].copy()
                no_obs = m.sum(axis=1) == 0
                gt[no_obs] = np.nan

                # Support gate (matching main experiment: 75% of loci valid)
                frac_valid = m.sum(axis=1) / m.shape[1]
                low_support = frac_valid < 0.75
                gt_gated = gt.copy()
                gt_gated[low_support] = np.nan

                # RPCA
                try:
                    rpca_trace = rpca.run(hs.signal).s_hat
                    rpca_gated = rpca_trace.copy()
                    rpca_gated[low_support] = np.nan
                    sdr_rpca.append(compute_sdr(gt_gated, rpca_gated))
                except Exception:
                    sdr_rpca.append(float("nan"))
                    rpca_trace = np.full_like(gt, np.nan)

                # SMS
                sms_trace = _sms_trace(hs.signal, bl_window)
                sms_gated = sms_trace.copy()
                sms_gated[low_support] = np.nan
                sdr_sms.append(compute_sdr(gt_gated, sms_gated))

                # Biomarker bias (MD, on ungated traces matching main experiment)
                try:
                    biases = compute_biomarker_bias(gt, rpca_trace, t, protocol, vessel_type)
                    md_bias_rpca.append(np.nanmean([c.get("MD", float("nan")) for c in biases]))
                except Exception:
                    md_bias_rpca.append(float("nan"))

                try:
                    biases_sms = compute_biomarker_bias(gt, sms_trace, t, protocol, vessel_type)
                    md_bias_sms.append(np.nanmean([c.get("MD", float("nan")) for c in biases_sms]))
                except Exception:
                    md_bias_sms.append(float("nan"))

        for method, sdr_list, md_list in [
            ("rpca", sdr_rpca, md_bias_rpca),
            ("sms", sdr_sms, md_bias_sms),
        ]:
            rows.append({
                "locus_variability": lv,
                "method": method,
                "n_samples": len(sdr_list),
                "sdr_median": float(np.nanmedian(sdr_list)),
                "sdr_q25": float(np.nanpercentile(sdr_list, 25)),
                "sdr_q75": float(np.nanpercentile(sdr_list, 75)),
                "md_bias_median": float(np.nanmedian(md_list)),
                "md_bias_q25": float(np.nanpercentile(md_list, 25)),
                "md_bias_q75": float(np.nanpercentile(md_list, 75)),
            })

    df = pd.DataFrame(rows)
    csv_path = out_dir / f"rank1_ablation_results{out_suffix}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    print(df.to_string(index=False))

    # Figure: locus variability vs SDR and MD bias (both methods)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for method, color, marker, label in [
        ("rpca", "#B2182B", "o", "RPCA"),
        ("sms", "#2166AC", "s", "SMS"),
    ]:
        sub = df[df["method"] == method]
        ax1.errorbar(
            sub["locus_variability"], sub["sdr_median"],
            yerr=[
                sub["sdr_median"] - sub["sdr_q25"],
                sub["sdr_q75"] - sub["sdr_median"],
            ],
            marker=marker, capsize=4, linewidth=1.5, color=color, label=label,
        )
        ax2.errorbar(
            sub["locus_variability"], sub["md_bias_median"],
            yerr=[
                sub["md_bias_median"] - sub["md_bias_q25"],
                sub["md_bias_q75"] - sub["md_bias_median"],
            ],
            marker=marker, capsize=4, linewidth=1.5, color=color, label=label,
        )

    ax1.set_xlabel("Locus amplitude variability ($\\sigma$)", fontsize=9)
    ax1.set_ylabel("SDR (dB)", fontsize=9)
    ax1.set_title("SDR vs locus variability", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.tick_params(labelsize=7)

    ax2.axhline(0, color="0.5", linewidth=0.5, linestyle="--")
    ax2.set_xlabel("Locus amplitude variability ($\\sigma$)", fontsize=9)
    ax2.set_ylabel("MD bias (% change)", fontsize=9)
    ax2.set_title("MD bias vs locus variability", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(labelsize=7)

    fig.tight_layout()
    fig_path = out_dir / f"fig_rank1_ablation{out_suffix}.png"
    fig.savefig(fig_path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Data loading (shared)
# ═══════════════════════════════════════════════════════════════════════════

def load_segments() -> Dict[str, Segment]:
    """Load DVA recordings and return retained segments (A1, V3)."""
    print("Loading data...")
    ds = ImedosReader(config=DataReaderConfig(protocol=PROTOCOL)).read(DATA_DIR)
    print(f"Loaded {len(ds.recordings)} recordings")

    segments: Dict[str, Segment] = {}
    for rec in ds.recordings:
        for seg_label, seg in rec.segments.items():
            if seg_label in SEGMENT_LABELS:
                key = f"{rec.subject_id}_{rec.visit_id}_{seg_label}"
                segments[key] = seg

    n_artery = sum(1 for s in segments.values() if s.vessel_type == "artery")
    n_vein = sum(1 for s in segments.values() if s.vessel_type == "vein")
    print(f"Total segments: {len(segments)} (arteries: {n_artery}, veins: {n_vein})")
    return segments


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Appendix analyses for DVA manuscript")
    parser.add_argument(
        "--analyses",
        nargs="*",
        default=["templates", "hybrid_example", "noise", "drift", "rank1"],
        choices=["templates", "hybrid_example", "noise", "drift", "rank1"],
        help="Which analyses to run (default: all five)",
    )
    parser.add_argument(
        "--levels",
        nargs="*",
        type=float,
        default=None,
        help="Per-locus amplitude variability levels (CV) for the rank1 analysis. "
             "Default: 0.0 0.05 0.10 0.15 0.20. Use e.g. --levels 0.25 0.40 to "
             "extend the grid in a follow-up run.",
    )
    parser.add_argument(
        "--out-suffix",
        default="",
        help="Suffix for rank1 output filenames (e.g. _ext) so a follow-up run "
             "does not overwrite the primary CSV.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    # Templates does not need data
    if "templates" in args.analyses:
        plot_templates(OUT_DIR)

    # The remaining analyses need loaded segments
    needs_data = {"hybrid_example", "noise", "drift", "rank1"}
    if needs_data & set(args.analyses):
        segments = load_segments()

        if "hybrid_example" in args.analyses:
            plot_hybrid_example(segments, OUT_DIR)

        if "noise" in args.analyses:
            run_noise_stationarity(segments, PROTOCOL, OUT_DIR)

        if "drift" in args.analyses:
            run_vasomotion_drift(segments, PROTOCOL, OUT_DIR)

        if "rank1" in args.analyses:
            run_rank1_ablation(
                segments, PROTOCOL, OUT_DIR,
                variability_levels=args.levels,
                out_suffix=args.out_suffix,
            )

    elapsed = time.perf_counter() - t0
    print(f"\nAll done in {elapsed:.0f}s ({elapsed / 60:.1f} min)")
