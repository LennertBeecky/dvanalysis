from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Tuple, Union

import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from dvanalysis.domain import Segment, SegmentSignal, StimulusProtocol, TimeWindow
from dvanalysis.preprocessing.result import PreprocessResult


def _slice_time(signal: SegmentSignal, t_start_sec: float, t_end_sec: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if t_end_sec <= t_start_sec:
        raise ValueError("t_end_sec must be > t_start_sec.")
    idx = (signal.t >= t_start_sec) & (signal.t < t_end_sec)
    return signal.t[idx], signal.x[idx, :], signal.m[idx, :]


def _slice_by_time(t: np.ndarray, y: np.ndarray, t_start: float, t_end: float):
    if t.ndim != 1:
        raise ValueError("t must be 1D.")
    mask = (t >= t_start) & (t <= t_end)
    if y.ndim == 1:
        return t[mask], y[mask]
    if y.ndim == 2:
        return t[mask], y[mask, :]
    raise ValueError("y must be 1D or 2D.")


def _masked_mean_over_loci(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    x_ma = np.where(m, x, np.nan)
    return np.nanmean(x_ma, axis=1)


def _masked_median_over_loci(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    x_ma = np.where(m, x, np.nan)
    return np.nanmedian(x_ma, axis=1)


def _extract_series(obj: Union[SegmentSignal, PreprocessResult], agg: str = "mean") -> Tuple[np.ndarray, np.ndarray, str, StimulusProtocol]:
    """
    Returns (t, y, units, protocol) where y is 1D.
    SegmentSignal -> masked mean/median over loci
    PreprocessResult -> s_hat (1D)
    """
    if isinstance(obj, SegmentSignal):
        t = obj.t
        y = _masked_median_over_loci(obj.x, obj.m) if agg == "median" else _masked_mean_over_loci(obj.x, obj.m)
        return t, y, obj.units, obj.protocol

    if isinstance(obj, PreprocessResult):
        if obj.s_hat is None:
            raise ValueError("PreprocessResult.s_hat is None.")
        t = obj.input_signal.t
        y = obj.s_hat
        return t, y, obj.input_signal.units, obj.input_signal.protocol

    raise TypeError(f"Unsupported type in methods: {type(obj)}")


def _default_baseline_window(protocol: StimulusProtocol) -> TimeWindow:
    if protocol.global_baseline is not None:
        return protocol.global_baseline
    if protocol.cycles is None or len(protocol.cycles) == 0 or protocol.cycles[0].baseline is None:
        raise ValueError("No baseline window available in protocol for percent normalisation.")
    return protocol.cycles[0].baseline


def _to_percent_baseline(y: np.ndarray, b0: float) -> np.ndarray:
    return (y - b0) / b0 * 100.0


def _interp_onto(t_src: np.ndarray, y_src: np.ndarray, t_ref: np.ndarray) -> np.ndarray:
    if y_src.ndim != 1:
        raise ValueError("Interpolation expects 1D y.")
    ok = np.isfinite(t_src) & np.isfinite(y_src)
    if ok.sum() < 2:
        return np.full_like(t_ref, np.nan, dtype=float)

    ts = t_src[ok]
    ys = y_src[ok]

    y_ref = np.interp(t_ref, ts, ys)
    y_ref[(t_ref < ts.min()) | (t_ref > ts.max())] = np.nan
    return y_ref


def _baseline_scalar(t: np.ndarray, y: np.ndarray, baseline_window: TimeWindow, stat: str = "median") -> float:
    mask = baseline_window.contains(t)
    if not np.any(mask):
        raise ValueError("Baseline window does not overlap time axis.")
    yb = y[mask]
    b0 = float(np.nanmean(yb)) if stat == "mean" else float(np.nanmedian(yb))
    if not np.isfinite(b0) or abs(b0) < 1e-12:
        raise ValueError(f"Invalid baseline scalar computed: {b0}")
    return b0


@dataclass
class PlotterConfig:
    template: str = "plotly_dark"
    width: int = 1100
    height: int = 700

    colorscale_red: Tuple[Tuple[float, str], ...] = (
        (0.0, "rgb(255, 245, 240)"),
        (0.2, "rgb(254, 224, 210)"),
        (0.4, "rgb(252, 187, 161)"),
        (0.6, "rgb(252, 146, 114)"),
        (0.8, "rgb(251, 106, 74)"),
        (1.0, "rgb(222, 45, 38)"),
    )

    colorscale_blue: Tuple[Tuple[float, str], ...] = (
        (0.0, "rgb(255, 255, 255)"),
        (0.2, "rgb(220, 240, 255)"),
        (0.4, "rgb(150, 210, 252)"),
        (0.6, "rgb(80, 180, 250)"),
        (0.8, "rgb(30, 120, 240)"),
        (1.0, "rgb(10, 50, 180)"),
    )

    show_first_locus_highlight: bool = True


@dataclass
class Plotter:
    config: PlotterConfig = field(default_factory=PlotterConfig)

    def plot_locus_surface_3d(self, segment: Segment, t_start_sec: Optional[float] = None, t_end_sec: Optional[float] = None,
                             surface_range: Optional[Tuple[float, float]] = None,
                             colorscale: Optional[Tuple[Tuple[float, str], ...]] = None,
                             title_prefix: str = "", save_html: Optional[Path] = None, show: bool = True) -> go.Figure:
        sig = segment.signal
        if t_start_sec is None:
            t_start_sec = float(sig.t[0])
        if t_end_sec is None:
            t_end_sec = float(sig.t[-1]) + 1e-12

        t, x, m = _slice_time(sig, t_start_sec, t_end_sec)
        x_vis = np.where(m, x, np.nan)
        z = x_vis.T
        loci = np.arange(1, z.shape[0] + 1, dtype=float)

        if colorscale is None:
            colorscale = self.config.colorscale_red if segment.vessel_type == "artery" else self.config.colorscale_blue

        if surface_range is None:
            x_valid = x[m]
            if x_valid.size == 0:
                raise ValueError("No valid data points available for scaling.")
            lo, hi = np.percentile(x_valid, [1, 99])
            pad = 0.05 * (hi - lo)
            surface_range = (lo - pad, hi + pad)

        fig = go.Figure(
            data=[
                go.Surface(
                    z=z, x=t, y=loci,
                    colorscale=colorscale,
                    cmin=surface_range[0], cmax=surface_range[1],
                    showscale=True,
                    colorbar=dict(
                        title=dict(text=f"Amplitude ({sig.units})", side="right", font=dict(size=12)),
                        tickfont=dict(size=10), len=0.4, thickness=10, x=1.15,
                    ),
                )
            ]
        )

        if self.config.show_first_locus_highlight and z.shape[0] >= 1:
            fig.add_trace(
                go.Scatter3d(
                    x=t, y=[1] * len(t), z=z[0, :],
                    mode="lines", line=dict(color="white", width=4), name="Locus 1",
                )
            )

        title = f"{title_prefix}{segment.segment_label} ({segment.vessel_type})"
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis=dict(title="Time (s)", showgrid=True, color="white"),
                yaxis=dict(title="Locus index", showgrid=True, color="white"),
                zaxis=dict(title=f"Amplitude ({sig.units})", range=list(surface_range), showgrid=True, color="white"),
                aspectratio=dict(x=1.8, y=1, z=0.7),
            ),
            template=self.config.template,
            margin=dict(l=20, r=20, b=20, t=50),
            paper_bgcolor="black" if self.config.template == "plotly_dark" else None,
            plot_bgcolor="black" if self.config.template == "plotly_dark" else None,
            width=self.config.width, height=self.config.height,
        )

        if save_html is not None:
            save_html = Path(save_html)
            save_html.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(save_html))

        if show:
            fig.show()

        return fig

    def plot_method_comparison_2d(
        self,
        segment_label: str,
        methods: Mapping[str, Union[SegmentSignal, PreprocessResult]],
        t_start_sec: Optional[float] = None,
        t_end_sec: Optional[float] = None,
        title_prefix: str = "",
        save_html: Optional[Path] = None,
        show: bool = True,
        reference: str = "first",
        normalize_to_percent_baseline: bool = False,
        baseline_window: Optional[TimeWindow] = None,
        baseline_stat: str = "median",
        agg: str = "mean",
        series_units: Optional[Mapping[str, str]] = None,
    ) -> go.Figure:
        if len(methods) == 0:
            raise ValueError("methods is empty.")

        series_units = dict(series_units or {})

        ref_obj = next(iter(methods.values())) if reference == "first" else methods[reference]
        t_ref, y_ref_native, ref_units, ref_protocol = _extract_series(ref_obj, agg=agg)

        if t_start_sec is None:
            t_start_sec = float(t_ref[0])
        if t_end_sec is None:
            t_end_sec = float(t_ref[-1]) + 1e-12

        t_ref_s, _ = _slice_by_time(t_ref, t_ref, float(t_start_sec), float(t_end_sec))

        if normalize_to_percent_baseline:
            bw = baseline_window or _default_baseline_window(ref_protocol)
            t_tmp, y_tmp = _slice_by_time(t_ref, y_ref_native, float(t_start_sec), float(t_end_sec))
            b0 = _baseline_scalar(t_tmp, y_tmp, bw, stat=baseline_stat)

        fig = go.Figure()

        for name, obj in methods.items():
            t, y, units, _ = _extract_series(obj, agg=agg)
            t_s, y_s = _slice_by_time(t, y, float(t_start_sec), float(t_end_sec))

            if normalize_to_percent_baseline:
                declared = series_units.get(name, "native")
                y_plot = y_s if declared == "percent" else _to_percent_baseline(y_s, b0)
                y_label_units = "% baseline"
            else:
                y_plot = y_s
                y_label_units = ref_units

            y_aligned = _interp_onto(t_s, y_plot, t_ref_s)

            fig.add_trace(go.Scatter(x=t_ref_s, y=y_aligned, mode="lines", name=name, connectgaps=False))

        fig.update_layout(
            title=f"{title_prefix}{segment_label} method comparison",
            xaxis=dict(title="Time (s)", showgrid=True),
            yaxis=dict(title=f"Amplitude ({y_label_units})", showgrid=True),
            template=self.config.template,
        )

        if save_html is not None:
            save_html = Path(save_html)
            save_html.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(save_html))

        if show:
            fig.show()

        return fig

    def plot_one_trace_2d(
        self,
        sig: Union[SegmentSignal, PreprocessResult],
        *,
        title: str = "",
        agg: str = "median",
        smooth_sec: float | None = None,
        protocol: Optional[StimulusProtocol] = None,
        ylim=None,
    ) -> None:
        if isinstance(sig, SegmentSignal):
            t = np.asarray(sig.t, float)
            if agg == "median":
                y = _masked_median_over_loci(sig.x, sig.m)
            elif agg == "mean":
                y = _masked_mean_over_loci(sig.x, sig.m)
            else:
                raise ValueError("agg must be 'median' or 'mean'")
            units = sig.units
            protocol = protocol or sig.protocol

        elif isinstance(sig, PreprocessResult):
            t = np.asarray(sig.input_signal.t, float)
            y = np.asarray(sig.s_hat, float)
            units = sig.input_signal.units
            protocol = protocol or sig.input_signal.protocol

        else:
            raise TypeError("sig must be SegmentSignal or PreprocessResult")

        if smooth_sec is not None and smooth_sec > 0:
            fs = 1.0 / np.median(np.diff(t))
            w = max(1, int(round(smooth_sec * fs)))
            if w % 2 == 0:
                w += 1
            pad = w // 2
            ypad = np.pad(y, (pad, pad), mode="edge")
            y = np.array([np.nanmedian(ypad[i:i + w]) for i in range(len(y))], float)

        plt.figure(figsize=(12, 4))
        plt.plot(t, y)
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel(f"Signal ({units})" if units else "Signal")
        if ylim is not None:
            plt.ylim(*ylim)

        if protocol is not None:
            if protocol.global_baseline is not None:
                gb = protocol.global_baseline
                plt.axvspan(gb.start_sec, gb.end_sec, alpha=0.08)

            if protocol.cycles is not None:
                for cyc in protocol.cycles:
                    plt.axvspan(cyc.baseline.start_sec, cyc.baseline.end_sec, alpha=0.05)
                    plt.axvspan(cyc.flicker.start_sec, cyc.flicker.end_sec, alpha=0.10)
                    plt.axvspan(cyc.recovery.start_sec, cyc.recovery.end_sec, alpha=0.05)

        plt.tight_layout()
        plt.show()
