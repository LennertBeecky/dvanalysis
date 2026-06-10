# preprocessing/kotliar.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from dvanalysis.domain import SegmentSignal, StimulusProtocol
from .base import BasePreprocessor
from .configs import PreprocessorConfig
from .result import PreprocessResult


@dataclass(frozen=True)
class KotliarConfig(PreprocessorConfig):
    """
    Kotliar style preprocessing for the 350 s DVA protocol used here.

    References:
        - Kotliar, K. E., Lanzl, I. M., Schmidt-Trucksäss, A., Sitnikova, D., Ali, M.,
        Blume, K., Halle, M., & Hanssen, H. (2011). Dynamic retinal vessel response
        to flicker in obesity: A methodological approach. Microvascular Research,
        81(1), 123–128. https://doi.org/10.1016/j.mvr.2010.11.007
        - Kotliar, K., Hauser, C., Ortner, M., Muggenthaler, C., Diehl-Schmid, J.,
        Angermann, S., Hapfelmeier, A., Schmaderer, C., & Grimmer, T. (2017).
        Altered neurovascular coupling as measured by optical imaging: a biomarker
        for Alzheimer's disease. Scientific Reports, 7(1), Article 12906.
        https://doi.org/10.1038/s41598-017-13349-5

    Expected protocol structure
      - global initial baseline buffer: 20 s
      - cycles: 3 repetitions of
          baseline: 30 s
          flicker: 20 s
          recovery: 50 s (constriction and relaxation)
      - global final buffer: 30 s

    Baseline normalization ambiguity
      The papers state "recalculated in % to baseline" but do not consistently specify
      whether this is percent change or percent ratio.
      We support both.

        delta_over_baseline:
          100 * (D(t) - D0) / D0

        ratio:
          100 * D(t) / D0

    Output
      - Default (mode="cycle_template"): s_hat is a full length (T,) trace, filled in
        the cycle windows with the averaged smoothed cycle curve, and NaN outside.
      - Option (mode="smooth_only"): s_hat is the smoothed percent trace of the full
        recording (optionally NaN outside cycle windows).
    """
    cycle_baseline_sec: float = 30.0
    expected_flicker_sec: float = 20.0
    expected_recovery_sec: float = 50.0

    expected_global_pre_sec: float = 20.0
    expected_global_post_sec: float = 30.0

    expected_n_cycles: int = 3
    enforce_total_duration: bool = True

    running_median_sec: float = 4.0
    backshift: bool = True

    locus_agg: str = "median"  # "median" or "mean"
    percent_mode: str = "delta_over_baseline"  # "delta_over_baseline" or "ratio"

    validate_protocol: bool = True
    strict_cycle_bounds: bool = True

    # New: switch between original Kotliar behaviour and "median smooth only"
    mode: str = "cycle_template"  # "cycle_template" or "smooth_only"
    smooth_only_baseline_source: str = "protocol_global"  # "protocol_global" or "cycle0"
    smooth_only_keep_outside_cycles: bool = True  # if False, NaN outside cycle windows


class KotliarPreprocessor(BasePreprocessor[KotliarConfig]):
    name = "Kotliar"
    version = "0.3"

    def run(self, signal: SegmentSignal) -> PreprocessResult:
        fs = float(signal.protocol.fs)

        if self.config.validate_protocol:
            self._validate_protocol(signal.protocol)

        # Aggregate loci to a single absolute diameter trace D(t)
        d_abs = self._aggregate_loci(signal.x, signal.m, mode=self.config.locus_agg)

        if self.config.mode == "smooth_only":
            return self._run_smooth_only(signal, d_abs, fs)

        # Default: Kotliar cycle template
        cycle_curves: List[np.ndarray] = []
        cycle_indices: List[np.ndarray] = []

        for cyc in signal.protocol.cycles:
            if cyc.baseline is None:
                raise ValueError(
                    f"Cycle {cyc.index} has no baseline TimeWindow. "
                    "KotliarPreprocessor expects explicit per-cycle baseline windows."
                )

            start_sec = float(cyc.baseline.start_sec)
            end_sec = float(cyc.recovery.end_sec)

            idx = np.where((signal.t >= start_sec) & (signal.t < end_sec))[0]
            if idx.size == 0:
                if self.config.strict_cycle_bounds:
                    raise ValueError(f"Empty cycle window for cycle {cyc.index}.")
                continue

            # Per-cycle baseline: baseline window itself
            b_idx = np.where(
                (signal.t >= float(cyc.baseline.start_sec)) & (signal.t < float(cyc.baseline.end_sec))
            )[0]
            if b_idx.size == 0:
                if self.config.strict_cycle_bounds:
                    raise ValueError(f"Empty baseline window for cycle {cyc.index}.")
                continue

            b0 = float(np.nanmedian(d_abs[b_idx]))
            if not np.isfinite(b0) or b0 == 0.0:
                raise ValueError(f"Invalid baseline in cycle {cyc.index}: median baseline={b0}.")

            d_cycle = d_abs[idx]
            y_cycle = self._to_percent(d_cycle, b0, mode=self.config.percent_mode)

            cycle_curves.append(y_cycle)
            cycle_indices.append(idx)

        if len(cycle_curves) == 0:
            raise ValueError("No cycles could be extracted for Kotliar preprocessing.")

        y_avg = self._median_across_cycles(cycle_curves)

        win = max(1, int(round(self.config.running_median_sec * fs)))
        y_smooth = self._running_median(y_avg, win, backshift=self.config.backshift)

        # Fill a full-length trace with the smoothed average cycle curve, NaN elsewhere
        s_hat_full = np.full(signal.T, np.nan, dtype=float)
        for idx in cycle_indices:
            L = min(idx.size, y_smooth.size)
            s_hat_full[idx[:L]] = y_smooth[:L]

        return PreprocessResult(
            preprocessor_name=self.name,
            preprocessor_version=self.version,
            config=self.config.to_dict(),
            input_signal=signal,
            s_hat=s_hat_full,
            m_used=signal.m.copy(),
            diagnostics={
                "mode": "cycle_template",
                "n_cycles_used": int(len(cycle_curves)),
                "cycle_curve_len_samples": int(y_avg.size),
                "running_median_window_samples": int(win),
                "percent_mode": self.config.percent_mode,
                "locus_agg": self.config.locus_agg,
                "backshift": bool(self.config.backshift),
            },
        )

    # -------------------------
    # New: smooth-only mode
    # -------------------------

    def _run_smooth_only(self, signal: SegmentSignal, d_abs: np.ndarray, fs: float) -> PreprocessResult:
        prot = signal.protocol

        if self.config.smooth_only_baseline_source == "protocol_global" and getattr(prot, "global_baseline", None) is not None:
            b = prot.global_baseline
        else:
            # fallback to cycle0 baseline
            if prot.cycles is None or len(prot.cycles) == 0 or prot.cycles[0].baseline is None:
                raise ValueError("smooth_only baseline_source='cycle0' requires a baseline window in cycle 0.")
            b = prot.cycles[0].baseline

        b_idx = np.where((signal.t >= float(b.start_sec)) & (signal.t < float(b.end_sec)))[0]
        if b_idx.size == 0:
            raise ValueError("Empty baseline window for smooth_only Kotliar.")

        b0 = float(np.nanmedian(d_abs[b_idx]))
        if not np.isfinite(b0) or b0 == 0.0:
            raise ValueError(f"Invalid baseline for smooth_only Kotliar: median baseline={b0}.")

        y = self._to_percent(d_abs, b0, mode=self.config.percent_mode)

        win = max(1, int(round(self.config.running_median_sec * fs)))
        y_smooth = self._running_median(y, win, backshift=self.config.backshift)

        if self.config.smooth_only_keep_outside_cycles:
            s_hat_full = y_smooth
        else:
            s_hat_full = np.full(signal.T, np.nan, dtype=float)
            for cyc in prot.cycles:
                idx = np.where(
                    (signal.t >= float(cyc.baseline.start_sec)) & (signal.t < float(cyc.recovery.end_sec))
                )[0]
                s_hat_full[idx] = y_smooth[idx]

        return PreprocessResult(
            preprocessor_name=self.name,
            preprocessor_version=self.version,
            config=self.config.to_dict(),
            input_signal=signal,
            s_hat=s_hat_full,
            m_used=signal.m.copy(),
            diagnostics={
                "mode": "smooth_only",
                "running_median_window_samples": int(win),
                "percent_mode": self.config.percent_mode,
                "locus_agg": self.config.locus_agg,
                "backshift": bool(self.config.backshift),
                "baseline_source": self.config.smooth_only_baseline_source,
                "keep_outside_cycles": bool(self.config.smooth_only_keep_outside_cycles),
            },
        )

    # -------------------------
    # Protocol validation
    # -------------------------

    def _validate_protocol(self, protocol: StimulusProtocol) -> None:
        if protocol.cycles is None or len(protocol.cycles) == 0:
            raise ValueError("Kotliar requires at least one cycle.")

        for c in protocol.cycles:
            if c.baseline is None:
                raise ValueError("Kotliar requires an explicit baseline window per cycle.")
            b = c.baseline
            f = c.flicker
            r = c.recovery

            if abs((b.end_sec - b.start_sec) - 30.0) > 1e-6:
                raise ValueError(
                    f"Kotliar expects 30s baseline per cycle, got {b.end_sec - b.start_sec:.3f}s in cycle {c.index}."
                )
            if abs((f.end_sec - f.start_sec) - 20.0) > 1e-6:
                raise ValueError(
                    f"Kotliar expects 20s flicker, got {f.end_sec - f.start_sec:.3f}s in cycle {c.index}."
                )
            if abs((r.end_sec - r.start_sec) - 50.0) > 1e-6:
                raise ValueError(
                    f"Kotliar expects 50s recovery, got {r.end_sec - r.start_sec:.3f}s in cycle {c.index}."
                )

            if not (b.start_sec <= b.end_sec <= f.start_sec <= f.end_sec <= r.start_sec <= r.end_sec):
                raise ValueError(f"Kotliar windows not ordered in cycle {c.index}.")

    # -------------------------
    # Helpers
    # -------------------------

    def _aggregate_loci(self, x: np.ndarray, m: np.ndarray, mode: str) -> np.ndarray:
        x_masked = np.where(m, x, np.nan)
        if mode == "mean":
            return np.nanmean(x_masked, axis=1)
        if mode == "median":
            return np.nanmedian(x_masked, axis=1)
        raise ValueError(f"Unknown locus_agg='{mode}'. Use 'median' or 'mean'.")

    def _to_percent(self, d: np.ndarray, baseline: float, mode: str) -> np.ndarray:
        if mode == "ratio":
            return 100.0 * (d / baseline)
        if mode == "delta_over_baseline":
            return 100.0 * ((d - baseline) / baseline)
        raise ValueError(f"Unknown percent_mode='{mode}'. Use 'delta_over_baseline' or 'ratio'.")

    def _median_across_cycles(self, cycles: List[np.ndarray]) -> np.ndarray:
        L = min(int(c.size) for c in cycles)
        stack = np.stack([c[:L] for c in cycles], axis=0)  # (C,L)
        return np.nanmedian(stack, axis=0)

    def _running_median(self, y: np.ndarray, window: int, backshift: bool) -> np.ndarray:
        if window <= 1:
            return y.copy()

        out = np.full_like(y, np.nan, dtype=float)

        for i in range(0, y.size):
            j = min(y.size, i + window)
            out[i] = np.nanmedian(y[i:j])

        if backshift:
            shift = window // 2
            if shift > 0:
                out = np.roll(out, -shift)
                out[-shift:] = np.nan

        return out
