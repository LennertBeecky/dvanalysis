"""
Generate the hybrid ground-truth construction schematic for the manuscript.

Illustrates the additive observation model of Section 2.5.2,

    Y = M . (S + A),

where S is the synthetic PCHIP response template, A is the real tiled baseline
residual and M is the real-derived observation mask, and shows how the same
construction yields the two evaluation conditions (stationary vs non-stationary
cycles).

Everything is built with the *real* construction functions: the stationary
sample comes from ``dvanalysis.validation.hybrid.build_hybrid`` and the
non-stationary sample from ``run_nonstationary_hybrid._build_nonstationary``;
the synthetic template is ``arterial_template`` and the per-cycle multipliers
are drawn from the SMS-derived empirical pool written by
``run_nonstationary_hybrid``. This script only visualises their outputs.

Usage:
    conda run -n dvanalysis python dvanalysis/examples/generate_hybrid_construction_figure.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap

from dvanalysis.io import DataReaderConfig, ImedosReader
from dvanalysis.validation.hybrid import build_hybrid
from dvanalysis.validation.templates import arterial_template, sample_arterial_params

# Reuse the shared protocol and the real non-stationary builder from the
# experiment script (same directory) rather than reimplementing either.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_nonstationary_hybrid import PROTOCOL, _build_nonstationary  # noqa: E402


# ── Paths ─────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "Healthy_Volunteers"
OUT_DIR = Path(__file__).resolve().parent / "figures"
TRIPLETS_CSV = (
    Path(__file__).resolve().parent.parent.parent
    / "outputs" / "nonstationary" / "empirical_amp_triplets_sms.csv"
)

SEG_LABEL = "A1"          # arterial segment (matches generate_pipeline_figure.py)
VESSEL_TYPE = "artery"
SEED = 42                 # mirrors run_nonstationary_hybrid Stage 2 (first param draw)

# ── Style (matches fig_pipeline: arterial red + neutral greys, one warm accent) ──
# fig_pipeline uses only vessel red and a family of greys. The two real-recording
# ingredients (residual A, mask M) are rendered in those same neutral greys so the
# figure stays tonally uniform with it; the single non-red accent (gold, below) is
# reserved for the per-cycle amplitude coding, the one thing that needs to pop.
VESSEL_COLOUR = "#B2182B"      # arterial red, as in fig_pipeline
RESIDUAL_COLOUR = "#7A7A7A"    # neutral grey for the real residual A (cf. artefact grey)
MASK_COLOUR = "#4D4D4D"        # charcoal for the observation mask M
MASK_CMAP = LinearSegmentedColormap.from_list(  # white (missing) -> charcoal (observed)
    "mask_obs", ["#ffffff", MASK_COLOUR])
HYBRID_COLOUR = "#7f7f7f"      # grey noisy hybrid trace, as in the appendix example
FL_SHADE = "#cccccc"           # flicker shading
FL_ALPHA = 0.25
ONSET_LINE = "#aaaaaa"
ZERO_LINE = "#cccccc"

PRE_SEC = 10.0                 # seconds shown before flicker onset (single-cycle panels)
HYBRID_STRIDE = 4              # downsample the grey hybrid trace in D/E/F (schematic)


# ── Display helper (standard baseline-median % change, not a construction step) ──
def _locus_median_pct(sig, baseline_windows) -> np.ndarray:
    """Locus-median percent change from per-locus baseline median.

    This is the standard baseline normalisation every denoiser applies before
    aggregating across loci (cf. ``_sms_locus_median_pct`` and the raw panel of
    ``plot_pipeline_figure``). It is a display transform, not part of the hybrid
    construction.
    """
    t = np.asarray(sig.t, float)
    x = np.asarray(sig.x, float).copy()
    x[~np.asarray(sig.m, bool)] = np.nan
    idx = np.concatenate([np.where(w.contains(t))[0] for w in baseline_windows])
    bl = np.nanmedian(x[idx, :], axis=0)                       # (P,) resting diameter
    bl_safe = np.where(np.isfinite(bl) & (bl != 0), bl, np.nan)
    pct = 100.0 * (x - bl_safe[np.newaxis, :]) / bl_safe[np.newaxis, :]
    return np.nanmedian(pct, axis=1)                           # (T,)


def _baseline_windows(protocol):
    wins = []
    if protocol.global_baseline is not None:
        wins.append(protocol.global_baseline)
    for cyc in protocol.cycles:
        if cyc.baseline is not None:
            wins.append(cyc.baseline)
    return wins


def _bridge_short_gaps(y: np.ndarray, max_gap: int = 2) -> np.ndarray:
    """Linearly interpolate only NaN runs of length <= ``max_gap``.

    The alternating-frame flicker dropout leaves every other sample missing; a
    line plot renders those as invisible isolated points. Bridging just the short
    gaps makes the flicker cycles visible while leaving genuine longer gaps broken.
    """
    y = np.asarray(y, float).copy()
    n = y.size
    isn = ~np.isfinite(y)
    i = 0
    while i < n:
        if not isn[i]:
            i += 1
            continue
        j = i
        while j < n and isn[j]:
            j += 1
        if (j - i) <= max_gap and i > 0 and j < n:           # interior short gap
            y0, y1 = y[i - 1], y[j]
            for k in range(i, j):
                frac = (k - (i - 1)) / (j - (i - 1))
                y[k] = y0 + frac * (y1 - y0)
        i = j
    return y


# Per-cycle amplitude coding. A warm gold accent marks the cycles whose amplitude
# deviates from unity, i.e. the non-stationarity; near-unity cycles stay neutral
# grey. Gold is the only non-red hue in the figure and pairs with the arterial red
# as a single warm accent, keeping the palette tonally uniform with fig_pipeline.
DEV_COLOUR = "#E6AB02"         # gold: a cycle scaled away from unity
NEUTRAL_COLOUR = "#8A8A8A"     # grey: a near-unity (unscaled) cycle
DEV_TOL = 0.05                 # |m - 1| above this counts as a deviating cycle


def _is_dev(m: float) -> bool:
    return abs(m - 1.0) > DEV_TOL


def _mult_colour(m: float) -> str:
    """Colour for the per-cycle amplitude multiplier ``m`` (gold if it deviates
    from unity, grey otherwise)."""
    return DEV_COLOUR if _is_dev(m) else NEUTRAL_COLOUR


def _style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7)


def _draw_cycle_markers(ax, fl_start_rel, fl_end_rel):
    ax.axvspan(fl_start_rel, fl_end_rel, alpha=FL_ALPHA, color=FL_SHADE, zorder=0)
    ax.axhline(0, color=ZERO_LINE, linewidth=0.5, linestyle=":", zorder=1)
    ax.axvline(fl_start_rel, color=ONSET_LINE, linewidth=0.6, linestyle="--", zorder=1)
    ax.axvline(fl_end_rel, color=ONSET_LINE, linewidth=0.6, linestyle="--", zorder=1)


def _shade_flickers(ax, fl_windows):
    """Shade all three flicker blocks plus the zero line (full-recording panels)."""
    for fs, fe in fl_windows:
        ax.axvspan(fs, fe, alpha=FL_ALPHA, color=FL_SHADE, zorder=0)
    ax.axhline(0, color=ZERO_LINE, linewidth=0.5, linestyle=":", zorder=1)


def _panel_letter(ax, letter):
    """Large bold panel label at a uniform axis-fraction position (top-left)."""
    ax.text(0.022, 0.94, letter, transform=ax.transAxes, ha="left", va="top",
            fontsize=16, fontweight="bold", color="#111111", zorder=10)


# ── Build the data with the real construction functions ───────────────
def build_samples():
    print("Loading recording...")
    ds = ImedosReader(config=DataReaderConfig(protocol=PROTOCOL)).read(DATA_DIR)
    rec = next(r for r in ds.recordings if SEG_LABEL in r.segments)
    sig = rec.segments[SEG_LABEL].signal
    print(f"  {rec.subject_id}/{rec.visit_id} {SEG_LABEL}: T={sig.T}, P={sig.P}")

    # Base template parameters, drawn with the real sampler (matches Stage 2).
    rng = np.random.default_rng(SEED)
    base_params = sample_arterial_params(rng)
    print(f"  base_params: {base_params}")

    # Representative empirical multiplier triplet (closest to the artery median CV).
    if not TRIPLETS_CSV.exists():
        raise FileNotFoundError(
            f"Empirical triplet pool not found at {TRIPLETS_CSV}. "
            "Run run_nonstationary_hybrid.py (Stage 1) first to generate it."
        )
    pool = pd.read_csv(TRIPLETS_CSV)
    art = pool[pool.vessel_type == VESSEL_TYPE]
    rep = art.iloc[(art.cv - art.cv.median()).abs().argmin()]
    triplet = (float(rep.m0), float(rep.m1), float(rep.m2))
    print(f"  non-stationary triplet (CV={rep.cv:.2f}): {triplet}")

    # Stationary: identical cycles. Non-stationary: empirical per-cycle multipliers.
    hs_stat = build_hybrid(
        sig, VESSEL_TYPE, base_params, rng=rng,
        cycle_jitter=False, locus_variability=0.0, locus_jitter_samples=2,
    )
    hs_ns = _build_nonstationary(
        sig, VESSEL_TYPE, base_params, rng,
        multipliers=triplet, locus_variability=0.0, locus_jitter_samples=2,
    )

    # Clean single-cycle template S (cycle 0), the synthetic ground truth before
    # locus broadcasting -- exactly what build_hybrid passes to arterial_template.
    t = np.asarray(sig.t, float)
    s_clean = arterial_template(t, PROTOCOL, base_params)

    return dict(
        sig=sig, t=t, base_params=base_params, triplet=triplet,
        hs_stat=hs_stat, hs_ns=hs_ns, s_clean=s_clean,
    )


# ── Figure ────────────────────────────────────────────────────────────
def make_figure(data, save_path):
    t = data["t"]
    s_clean = data["s_clean"]
    hs_stat = data["hs_stat"]
    hs_ns = data["hs_ns"]
    triplet = data["triplet"]
    bwins = _baseline_windows(PROTOCOL)

    # Single-cycle window (cycle 0), flicker-onset aligned: panel A shows the
    # synthetic template as one "unit" cycle (the repeating building block), as
    # in fig_pipeline.
    cyc = PROTOCOL.cycles[0]
    fl_start = float(cyc.flicker.start_sec)
    fl_end = float(cyc.flicker.end_sec)
    fl_dur = fl_end - fl_start
    win_start = float(cyc.baseline.start_sec) - PRE_SEC
    win_end = float(cyc.recovery.end_sec)
    wmask = (t >= win_start) & (t < win_end)
    t_rel = t[wmask] - fl_start
    xlim_unit = (win_start - fl_start, win_end - fl_start)

    # Full-recording canvas: panels B-F share the three-cycle time axis, so the
    # hybrid sample (D) establishes the canvas that E/F then manipulate.
    fl_windows = [(float(c.flicker.start_sec), float(c.flicker.end_sec))
                  for c in PROTOCOL.cycles]
    # Per-cycle search windows (flicker onset -> recovery end) used to locate each
    # cycle's dilation peak, where the amplitude-coding marker is anchored.
    cyc_windows = [(float(c.flicker.start_sec), float(c.recovery.end_sec))
                   for c in PROTOCOL.cycles]
    t_max = float(PROTOCOL.cycles[-1].recovery.end_sec)
    full = (t >= 0) & (t <= t_max)
    tf = t[full]

    # Locus-median percent-change traces (full length). Bridge only the short
    # alternating-frame dropout gaps so the flicker cycles are visible.
    y_hybrid_raw = _locus_median_pct(hs_stat.signal, bwins)
    y_hyb_ns_raw = _locus_median_pct(hs_ns.signal, bwins)
    y_hybrid = _bridge_short_gaps(y_hybrid_raw, max_gap=2)
    y_hyb_ns = _bridge_short_gaps(y_hyb_ns_raw, max_gap=2)
    # Ground truth drawn as the clean continuous template (params["gt_pct_clean"]),
    # so the red response stays unbroken through the flicker dropout.
    gt_stat = np.asarray(hs_stat.params["gt_pct_clean"], float)
    gt_ns = np.asarray(hs_ns.params["gt_pct_clean"], float)
    # Tight shared y-range for D/E/F so the response fills the panel; derived from
    # the ground-truth range (incl. the constriction trough) plus a small margin.
    # Rare noise spikes clip at the edges -- the data is not modified, only the view.
    _gt_view = np.concatenate([gt_stat[full], gt_ns[full]])
    _g_lo, _g_hi = float(np.nanmin(_gt_view)), float(np.nanmax(_gt_view))
    _span = _g_hi - _g_lo
    branch_ylim = (_g_lo - 0.30 * _span, _g_hi + 0.20 * _span)
    # Panel B shows A, the real baseline residual *added* to S when the hybrid is
    # built (build_hybrid -> extract_baseline_residual + tile_residual). Because
    # Y = S + A on observed entries, it is read back here as A = Y - S (locus
    # medians); it is NOT computed by subtraction in the construction itself.
    a_resid = _bridge_short_gaps(y_hybrid_raw - gt_stat, max_gap=2)
    coverage = np.asarray(hs_stat.signal.m, bool).mean(axis=1)  # fraction observed

    # Schematic window for the ingredient panels B/C: one baseline + one flicker
    # segment, so they read as illustrations of the residual and dropout pattern
    # rather than full-recording data dumps.
    SCH_START, SCH_END = 40.0, 95.0
    smask = (t >= SCH_START) & (t < SCH_END)
    t_sch = t[smask]
    sch_fl = (float(PROTOCOL.cycles[0].flicker.start_sec),
              float(PROTOCOL.cycles[0].flicker.end_sec))

    # Layout mirrors the construction flow top-to-bottom:
    #   row 0  S  +  A  (.)  M     three equal-status ingredients
    #   row 1        Y           the combined hybrid, centred under row 0
    #   row 2     E      F        the two condition branches
    # Six equal columns let row 0 use 2-col cells, row 1 a centred 4-col cell,
    # and row 2 two 3-col cells; operators/arrows sit in the gaps between them.
    fig = plt.figure(figsize=(11.5, 9.4))
    outer = fig.add_gridspec(
        3, 12, height_ratios=[1.0, 1.05, 0.52],   # E/F ~half the height of D
        hspace=0.78, wspace=0.35, left=0.07, right=0.96, top=0.92, bottom=0.06,
    )
    # Row 0: ingredient group centred above D. S is a wide panel spanning both
    # sub-rows; A (residual) and M (mask) are stacked in a narrow right column,
    # grouping the two real-recording ingredients. Reads: [ S ] + [ A / M ].
    ing = outer[0, 2:10].subgridspec(
        2, 2, width_ratios=[2.1, 1.3], height_ratios=[1.0, 1.0],
        wspace=0.55, hspace=0.5,
    )
    ax_S = fig.add_subplot(ing[:, 0])
    ax_A = fig.add_subplot(ing[0, 1])
    ax_M = fig.add_subplot(ing[1, 1], sharex=ax_A)   # B and C share the time axis exactly
    # Row 1: hybrid D, centred and aligned with the ingredient group.
    ax_hy = fig.add_subplot(outer[1, 2:10])
    # Row 2: the two condition branches, full width.
    ax_stat = fig.add_subplot(outer[2, 2:6])     # compact boxes centred under D
    ax_ns = fig.add_subplot(outer[2, 6:10])

    # ── A: synthetic response template S ──────────────────
    ax = ax_S
    _draw_cycle_markers(ax, 0.0, fl_dur)
    ax.plot(t_rel, s_clean[wmask], color=VESSEL_COLOUR, linewidth=1.6, zorder=3)
    ax.fill_between(t_rel, 0, s_clean[wmask], color=VESSEL_COLOUR, alpha=0.10, zorder=2)
    ax.set_xlim(*xlim_unit)
    ax.set_ylabel("% change", fontsize=8)
    ax.set_xlabel("Time rel. flicker onset (s)", fontsize=8)
    ax.set_title(r"Synthetic template $\mathbf{S}$ (one cycle)", fontsize=9, loc="left")
    _panel_letter(ax, "A")
    _style_axis(ax)

    # ── B: residual A (schematic short window, top of the real-recording stack) ──
    ax = ax_A
    ax.axvspan(sch_fl[0], sch_fl[1], alpha=FL_ALPHA, color=FL_SHADE, zorder=0)
    ax.axhline(0, color=ZERO_LINE, linewidth=0.5, linestyle=":", zorder=1)
    ax.plot(t_sch, a_resid[smask], color=RESIDUAL_COLOUR, linewidth=0.6, zorder=3)
    ax.set_xlim(SCH_START, SCH_END)
    ax.set_xticks([50, 70])                 # identical ticks to C (shared x-axis)
    ax.set_yticks([-10, 0, 10])
    ax.set_ylabel("%", fontsize=7)
    ax.tick_params(labelbottom=False, labelsize=6)   # C below carries the tick labels
    ax.set_title(r"Residual $\mathbf{A}$", fontsize=9, loc="left")
    _panel_letter(ax, "B")
    _style_axis(ax)

    # ── C: mask M (raster strip; coverage = fraction of loci observed, 0..1) ──
    ax = ax_M
    ax.axvspan(sch_fl[0], sch_fl[1], alpha=FL_ALPHA, color=FL_SHADE, zorder=0)
    # Thin horizontal timeline: purple where observed, white where dropped, with
    # intermediate shade for partial loci coverage. interpolation="nearest" keeps
    # the alternating-frame dropout from being averaged into a flat band.
    strip = coverage[smask][np.newaxis, :]            # (1, T_window)
    ax.imshow(strip, aspect="auto", origin="lower", interpolation="nearest",
              extent=[SCH_START, SCH_END, 0.35, 0.65],
              cmap=MASK_CMAP, vmin=0.0, vmax=1.0, zorder=2)
    ax.set_xlim(SCH_START, SCH_END)
    ax.set_ylim(0, 1)
    ax.set_yticks([])                        # binary-style indicator: no quantitative y
    ax.set_xticks([50, 70])
    ax.set_ylabel("observed", fontsize=7)
    ax.set_xlabel("Time (s)", fontsize=7)
    ax.set_title(r"Mask $\mathbf{M}$", fontsize=9, loc="left")
    ax.text(float(np.mean(sch_fl)), 0.14, "alternating-frame dropout",
            ha="center", va="center", fontsize=5.5, color=MASK_COLOUR, style="italic")
    ax.tick_params(labelsize=6)
    _panel_letter(ax, "C")
    _style_axis(ax)

    # ── D: hybrid sample Y = M . (S + A) (full recording) ─
    ax = ax_hy
    _shade_flickers(ax, fl_windows)
    ax.plot(tf[::HYBRID_STRIDE], y_hybrid[full][::HYBRID_STRIDE], color=HYBRID_COLOUR,
            linewidth=0.4, alpha=0.4, zorder=2, label="Hybrid $\\mathbf{Y}$ (observed)")
    ax.plot(tf, gt_stat[full], color=VESSEL_COLOUR, linewidth=1.7, zorder=4,
            label="Ground truth $\\mathbf{S}$")
    ax.set_xlim(0, t_max)
    ax.set_ylim(*branch_ylim)
    ax.set_yticks([0])               # single baseline reference; no quantitative scale
    ax.set_ylabel("% change", fontsize=8)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_title(r"Hybrid sample  $\mathbf{Y}=\mathbf{M}\odot(\mathbf{S}+\mathbf{A})$",
                 fontsize=9, loc="left")
    ax.legend(fontsize=6.5, loc="upper right", framealpha=0.95,
              edgecolor="#dddddd", fancybox=False)
    _panel_letter(ax, "D")
    _style_axis(ax)

    # ── Bottom branches: stationary vs non-stationary ─────
    def _draw_branch(ax, gt_full, mults, title, letter, show_ylabel=True):
        # Compact outcome box: clean three-cycle ground-truth curve (the amplitude
        # pattern is the message), no dense noise -- D is where noise is shown. The
        # per-cycle amplitude state is triple-coded so it reads at a glance: each
        # flicker band is tinted gold (deviating) or grey (near-unity), a matching
        # dot sits on each cycle's dilation peak, and a bold chip gives the value.
        gt_f = gt_full[full]
        ax.axhline(0, color=ZERO_LINE, linewidth=0.5, linestyle=":", zorder=1)
        for (fs, fe), m in zip(fl_windows, mults):
            if _is_dev(m):
                ax.axvspan(fs, fe, alpha=0.24, color=DEV_COLOUR, zorder=0)
            else:
                ax.axvspan(fs, fe, alpha=FL_ALPHA, color=FL_SHADE, zorder=0)
        ax.plot(tf, gt_f, color=VESSEL_COLOUR, linewidth=1.6, zorder=4)
        ax.set_xlim(0, t_max)
        # Extra top headroom (vs the shared D range) so the value chips sit clear
        # above even the amplified non-stationary peak instead of colliding with it.
        lo, hi = branch_ylim
        top = hi + 0.30 * (hi - lo)
        ax.set_ylim(lo, top)
        ax.set_yticks([0])           # baseline reference only; no quantitative scale
        ax.set_xticks([])            # the three cycles are shown by the flicker shading
        ytxt = top - 0.075 * (top - lo)
        for (cs, ce), (fs, fe), m in zip(cyc_windows, fl_windows, mults):
            col = _mult_colour(m)
            dev = _is_dev(m)
            # Dot on this cycle's dilation peak (max of the ground-truth curve
            # within the flicker -> recovery window).
            seg = np.where((tf >= cs) & (tf <= ce))[0]
            if seg.size:
                k = seg[int(np.nanargmax(gt_f[seg]))]
                ax.plot([tf[k]], [gt_f[k]], marker="o",
                        markersize=9 if dev else 6,
                        markerfacecolor=col, markeredgecolor="white",
                        markeredgewidth=1.1, zorder=6, clip_on=False)
            # Bold value chip at the top of the band, outlined in the cycle colour.
            ax.annotate(rf"$\times{m:.2f}$", xy=((fs + fe) / 2, ytxt),
                        ha="center", va="center", fontsize=10.5,
                        color=col, fontweight="bold", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.22", fc="white",
                                  ec=col, lw=1.3 if dev else 0.9))
        if show_ylabel:
            ax.set_ylabel("% change", fontsize=7)
        else:
            ax.set_yticklabels([])   # shares E's scale; avoid crowding the E|F gap
        ax.set_xlabel("3 flicker cycles", fontsize=7)
        ax.set_title(title, fontsize=9, loc="left")
        _panel_letter(ax, letter)
        _style_axis(ax)

    _draw_branch(ax_stat, gt_stat, (1.0, 1.0, 1.0), "Stationary", "E")
    _draw_branch(ax_ns, gt_ns, triplet, "Non-stationary", "F", show_ylabel=False)

    # ── Operators and flow arrows (figure-fraction, tied to panel edges) ──
    fig.canvas.draw()
    bS = ax_S.get_position()
    bA = ax_A.get_position()
    bM = ax_M.get_position()
    bH = ax_hy.get_position()
    bSt = ax_stat.get_position()
    bNs = ax_ns.get_position()

    # "+" between S and the stacked A/M column, centred on the group height.
    grp_y = (bS.y0 + bS.y1) / 2
    fig.text((bS.x1 + bA.x0) / 2, grp_y, r"$\mathbf{+}$", ha="center", va="center",
             fontsize=24, color="#333333")
    # Bracket label tying A and M to the same real recording (rotated, to the
    # right of the column so it never collides with the panel titles).
    fig.text(bA.x1 + 0.006, grp_y, "from real recording", rotation=270,
             ha="left", va="center", fontsize=7, style="italic", color="#777777")

    def _arrow(p0, p1, arrowstyle="-|>", **kw):
        a = FancyArrowPatch(
            p0, p1, transform=fig.transFigure, arrowstyle=arrowstyle,
            mutation_scale=14, lw=1.3, color="#555555",
            shrinkA=2, shrinkB=2, **kw,
        )
        fig.add_artist(a)
        return a

    xc = (bH.x0 + bH.x1) / 2          # D centre = ingredient-group centre
    xS_c = (bS.x0 + bS.x1) / 2        # S panel centre
    xAM_c = (bA.x0 + bA.x1) / 2       # A/M column centre
    # Orthogonal merge, routed through the whitespace below the ingredient x-labels:
    # a vertical drop from S and from the A/M group, a horizontal join, then one
    # vertical arrow into D. The label states the operation.
    y_ing = bM.y0 - 0.050             # below the ingredient panels' x-axis labels
    y_join = bH.y1 + 0.45 * (y_ing - bH.y1)
    _arrow((xS_c, y_ing), (xS_c, y_join), arrowstyle="-")     # drop from S
    _arrow((xAM_c, y_ing), (xAM_c, y_join), arrowstyle="-")   # drop from A/M group
    _arrow((xS_c, y_join), (xAM_c, y_join), arrowstyle="-")   # horizontal join
    _arrow((xc, y_join), (xc, bH.y1))                          # single arrow into D
    fig.text(xc + 0.014, (y_join + bH.y1) / 2,
             r"$(\mathbf{S}+\mathbf{A})\,\odot\,\mathbf{M}$" + "\nelement-wise",
             ha="left", va="center", fontsize=8, style="italic", color="#555555")

    # Orthogonal split, routed below D's x-label into the two compact outcome boxes:
    # a vertical drop from D, a horizontal line, then two vertical arrows into E and F.
    xE = (bSt.x0 + bSt.x1) / 2        # E panel centre (figure coords)
    xF = (bNs.x0 + bNs.x1) / 2        # F panel centre (figure coords)
    x_mid = (xE + xF) / 2             # exact midpoint -> symmetric inverted-T
    y_dstart = bH.y0 - 0.050          # below D's x-axis label
    y_split = bSt.y1 + 0.45 * (y_dstart - bSt.y1)
    _arrow((x_mid, y_dstart), (x_mid, y_split), arrowstyle="-")  # drop from D
    _arrow((xE, y_split), (xF, y_split), arrowstyle="-")        # horizontal join
    _arrow((xE, y_split), (xE, bSt.y1))                          # straight down into E
    _arrow((xF, y_split), (xF, bNs.y1))                          # straight down into F
    fig.text(x_mid + 0.012, (y_dstart + y_split) / 2, "per-cycle scaling",
             ha="left", va="center", fontsize=8, style="italic", color="#555555")

    # ── Save ─────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{save_path}.png", dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(f"{save_path}.pdf", facecolor="white", bbox_inches="tight")
    print(f"Saved: {save_path}.png and .pdf")
    return fig


def main():
    data = build_samples()
    make_figure(data, str(OUT_DIR / "fig_hybrid_construction"))
    print("Done.")


if __name__ == "__main__":
    main()
