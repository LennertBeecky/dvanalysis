# visualisation/triage.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dvanalysis.domain import Segment, Recording, StimulusProtocol, TimeWindow


def _masked_median_over_loci(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    """
    x: (T, P), m: (T, P) bool
    returns (T,)
    """
    x_ma = np.where(m, x, np.nan)
    return np.nanmedian(x_ma, axis=1)


def _running_median_1d(y: np.ndarray, window: int) -> np.ndarray:
    """
    Simple causal running median with NaN handling.
    """
    if window <= 1:
        return y.copy()

    out = np.full_like(y, np.nan, dtype=float)
    for i in range(y.size):
        j0 = max(0, i - window + 1)
        out[i] = np.nanmedian(y[j0 : i + 1])
    return out


def _window_to_rects(protocol: StimulusProtocol) -> list[Tuple[float, float, str]]:
    """
    Returns list of (start_sec, end_sec, phase_name) for shading.
    Includes global_baseline if present, then each cycle baseline, flicker, recovery.
    """
    rects: list[Tuple[float, float, str]] = []
    if protocol.global_baseline is not None:
        rects.append((float(protocol.global_baseline.start_sec), float(protocol.global_baseline.end_sec), "global_baseline"))

    if protocol.cycles is None:
        return rects

    for cyc in protocol.cycles:
        if cyc.baseline is not None:
            rects.append((float(cyc.baseline.start_sec), float(cyc.baseline.end_sec), "baseline"))
        if cyc.flicker is not None:
            rects.append((float(cyc.flicker.start_sec), float(cyc.flicker.end_sec), "flicker"))
        if cyc.recovery is not None:
            rects.append((float(cyc.recovery.start_sec), float(cyc.recovery.end_sec), "recovery"))
    return rects


def _add_protocol_shading(fig: go.Figure, protocol: StimulusProtocol) -> None:
    """
    Adds vertical shaded regions for baseline, flicker, recovery.
    Uses plotly_dark friendly opacities.
    """
    rects = _window_to_rects(protocol)

    phase_style = {
        "global_baseline": dict(fillcolor="rgba(120,120,120,0.18)", line_width=0),
        "baseline": dict(fillcolor="rgba(120,120,120,0.14)", line_width=0),
        "flicker": dict(fillcolor="rgba(255,255,0,0.14)", line_width=0),
        "recovery": dict(fillcolor="rgba(0,200,255,0.10)", line_width=0),
    }

    for (a, b, name) in rects:
        style = phase_style.get(name, dict(fillcolor="rgba(160,160,160,0.10)", line_width=0))
        fig.add_vrect(x0=a, x1=b, **style)


def plot_triage_2d(
    segment: Segment,
    *,
    smooth_sec: float = 1.0,
    title_prefix: str = "",
    t_start_sec: Optional[float] = None,
    t_end_sec: Optional[float] = None,
    show: bool = True,
    save_html: Optional[Path] = None,
) -> go.Figure:
    """
    Triage plot for one vessel segment.
    Shows:
      - raw median over loci (thin)
      - running median of that trace (thick)
      - protocol windows shaded (baseline, flicker, recovery)

    smooth_sec is converted to a window length using protocol.fs.
    """
    sig = segment.signal
    t = sig.t
    y_raw = _masked_median_over_loci(sig.x, sig.m)

    fs = float(sig.protocol.fs)
    win = max(1, int(round(float(smooth_sec) * fs)))
    y_smooth = _running_median_1d(y_raw, win)

    if t_start_sec is None:
        t_start_sec = float(t[0])
    if t_end_sec is None:
        t_end_sec = float(t[-1]) + 1e-12

    mask = (t >= float(t_start_sec)) & (t <= float(t_end_sec))
    t_s = t[mask]
    y_raw_s = y_raw[mask]
    y_smooth_s = y_smooth[mask]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=t_s,
            y=y_raw_s,
            mode="lines",
            name="raw median over loci",
            line=dict(width=1),
            opacity=0.55,
            connectgaps=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t_s,
            y=y_smooth_s,
            mode="lines",
            name=f"running median ({smooth_sec:.2f}s)",
            line=dict(width=3),
            connectgaps=False,
        )
    )

    _add_protocol_shading(fig, sig.protocol)

    title = f"{title_prefix}{segment.segment_label} ({segment.vessel_type})"
    fig.update_layout(
        title=title,
        template="plotly_dark",
        xaxis=dict(title="Time (s)", showgrid=True),
        yaxis=dict(title=f"Amplitude ({sig.units})", showgrid=True),
        legend=dict(borderwidth=1),
        margin=dict(l=40, r=20, b=40, t=60),
        width=1100,
        height=450,
    )

    if save_html is not None:
        save_html = Path(save_html)
        save_html.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_html))

    if show:
        fig.show()

    return fig


def plot_triage_2d_multi(
    recording: Recording,
    *,
    segment_labels: Sequence[str] = ("A1", "A2", "V3", "V4"),
    smooth_sec: float = 1.0,
    title_prefix: str = "",
    t_start_sec: Optional[float] = None,
    t_end_sec: Optional[float] = None,
    show: bool = True,
    save_html: Optional[Path] = None,
) -> go.Figure:
    """
    One figure with stacked panels for multiple segments in a recording.
    Each panel shows raw median over loci + running median.
    Protocol windows shaded in every panel.
    """
    existing = [s for s in segment_labels if s in recording.segments]
    if len(existing) == 0:
        raise ValueError(f"No segment labels found in recording. Requested: {list(segment_labels)}")

    first_seg = recording.segments[existing[0]]
    sig0 = first_seg.signal
    fs = float(sig0.protocol.fs)
    win = max(1, int(round(float(smooth_sec) * fs)))

    t0 = sig0.t
    if t_start_sec is None:
        t_start_sec = float(t0[0])
    if t_end_sec is None:
        t_end_sec = float(t0[-1]) + 1e-12

    fig = make_subplots(
        rows=len(existing),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[f"{lab} ({recording.segments[lab].vessel_type})" for lab in existing],
    )

    for r, lab in enumerate(existing, start=1):
        seg = recording.segments[lab]
        sig = seg.signal

        t = sig.t
        y_raw = _masked_median_over_loci(sig.x, sig.m)
        y_smooth = _running_median_1d(y_raw, win)

        mask = (t >= float(t_start_sec)) & (t <= float(t_end_sec))
        t_s = t[mask]
        y_raw_s = y_raw[mask]
        y_smooth_s = y_smooth[mask]

        fig.add_trace(
            go.Scatter(
                x=t_s,
                y=y_raw_s,
                mode="lines",
                name="raw median over loci" if r == 1 else None,
                showlegend=(r == 1),
                line=dict(width=1),
                opacity=0.55,
                connectgaps=False,
            ),
            row=r,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=t_s,
                y=y_smooth_s,
                mode="lines",
                name=f"running median ({smooth_sec:.2f}s)" if r == 1 else None,
                showlegend=(r == 1),
                line=dict(width=3),
                connectgaps=False,
            ),
            row=r,
            col=1,
        )

        _add_protocol_shading(fig, sig.protocol)

        fig.update_yaxes(title_text=f"{sig.units}", row=r, col=1)

    fig.update_xaxes(title_text="Time (s)", row=len(existing), col=1)

    fig.update_layout(
        title=f"{title_prefix}{recording.subject_id}_{recording.visit_id} | triage",
        template="plotly_dark",
        legend=dict(borderwidth=1),
        margin=dict(l=50, r=20, b=50, t=80),
        width=1200,
        height=320 * len(existing),
    )

    if save_html is not None:
        save_html = Path(save_html)
        save_html.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_html))

    if show:
        fig.show()

    return fig
