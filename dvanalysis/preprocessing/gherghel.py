# preprocessing/gherghel.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from dvanalysis.domain import SegmentSignal
from .base import BasePreprocessor
from .configs import PreprocessorConfig
from .result import PreprocessResult


@dataclass(frozen=True)
class GherghelConfig(PreprocessorConfig):
    """
    Gherghel style preprocessing.

    References:
    - Mroczkowska, S., Ekart, A., Sung, V., Negi, A., Qin, L., Patel, S. R., Jacob, S.,
      Atkins, C., Benavente-Perez, A., & Gherghel, D. (2012). Coexistence of macro- and
      micro-vascular abnormalities in newly diagnosed normal tension glaucoma patients.
      Acta Ophthalmologica (Oxford, England), 90(7), e553–e559.

    What the paper describes, operationally
      1. Export raw response data
      2. Fit a polynomial regression curve using MATLAB polyfit and polyval
      3. Use the fitted curve to study slopes and dynamics

    Baseline window rule (Nagel et al. 2004)
      For each flicker cycle, baseline is the interval:
        from 30 s to 5 s prior to flicker start
      This yields a 25 s baseline window that avoids the last 5 s before flicker.

    Fit scope ambiguity
      The paper text implies a polynomial regression on the overall vascular response
      profile (global curve). It does not explicitly say cycle-wise fits.

      Therefore:
        fit_mode = "global" is the default and paper-faithful
        fit_mode = "per_cycle" is an extension for experimentation

    Normalisation ambiguity
      The papers say "recalculated in % to baseline" but do not consistently specify
      ratio versus percent change. We support both.

        delta_over_baseline:
          100 * (D(t) - D0) / D0

        ratio:
          100 * D(t) / D0

    Output
      - If fit_mode == "global":
          s_hat is a full length (T,) trace with the global fitted polynomial curve.
      - If fit_mode == "per_cycle":
          s_hat is a full length (T,) trace filled with fitted polynomial per-cycle
          in the cycle windows, NaN outside those windows.
    """

    baseline_start_before_flicker_sec: float = 30.0
    baseline_end_before_flicker_sec: float = 5.0

    polynomial_degree: int = 4

    locus_agg: str = "median"  # "median" or "mean"
    percent_mode: str = "delta_over_baseline"  # "delta_over_baseline" or "ratio"

    fit_mode: str = "global"  # "global" or "per_cycle"

    validate_protocol: bool = True
    strict_cycle_bounds: bool = True


class GherghelPreprocessor(BasePreprocessor[GherghelConfig]):
    name = "GherghelPolynomial"
    version = "0.3"

    def run(self, signal: SegmentSignal) -> PreprocessResult:
        if self.config.validate_protocol:
            self._validate_protocol(signal)

        d_abs = self._aggregate_loci(signal.x, signal.m, mode=self.config.locus_agg)

        fit_mode = str(self.config.fit_mode).lower().strip()
        if fit_mode not in {"global", "per_cycle"}:
            raise ValueError("GherghelConfig.fit_mode must be 'global' or 'per_cycle'.")

        if fit_mode == "global":
            s_hat, diagnostics = self._fit_global(signal=signal, d_abs=d_abs)
        else:
            s_hat, diagnostics = self._fit_per_cycle(signal=signal, d_abs=d_abs)

        return PreprocessResult(
            preprocessor_name=self.name,
            preprocessor_version=self.version,
            config=self.config.to_dict(),
            input_signal=signal,
            s_hat=s_hat,
            m_used=signal.m.copy(),
            diagnostics=diagnostics,
        )

    # ---------------------------------------------------------------------
    # Global fit (paper-faithful default)
    # ---------------------------------------------------------------------

    def _fit_global(self, signal: SegmentSignal, d_abs: np.ndarray) -> tuple[np.ndarray, dict]:
        prot = signal.protocol

        # Use a single baseline for the full curve.
        # We choose the first available baseline window:
        # 1) protocol.global_baseline if present
        # 2) baseline window defined by Nagel rule for the first cycle
        if prot.global_baseline is not None:
            b0_start = float(prot.global_baseline.start_sec)
            b0_end = float(prot.global_baseline.end_sec)
        else:
            c0 = prot.cycles[0]
            flicker_start = float(c0.flicker.start_sec)
            b0_start = flicker_start - float(self.config.baseline_start_before_flicker_sec)
            b0_end = flicker_start - float(self.config.baseline_end_before_flicker_sec)

        idx_base = np.where((signal.t >= b0_start) & (signal.t < b0_end))[0]
        if idx_base.size == 0:
            raise ValueError(
                "Missing baseline samples for global normalization. "
                f"Expected baseline window [{b0_start:.3f}, {b0_end:.3f}) s."
            )

        baseline_level = float(np.nanmedian(d_abs[idx_base]))
        if not np.isfinite(baseline_level) or baseline_level == 0.0:
            raise ValueError(f"Invalid global baseline: median baseline={baseline_level}.")

        y = self._to_percent(d_abs, baseline_level, mode=self.config.percent_mode)

        ok = np.isfinite(signal.t) & np.isfinite(y)
        if int(ok.sum()) < (self.config.polynomial_degree + 1):
            raise ValueError(
                f"Not enough valid samples to fit degree {self.config.polynomial_degree} polynomial globally. "
                f"Need at least {self.config.polynomial_degree + 1}, got {int(ok.sum())}."
            )

        coeffs = np.polyfit(signal.t[ok], y[ok], deg=int(self.config.polynomial_degree))
        y_fit = np.polyval(coeffs, signal.t)

        diagnostics = {
            "fit_mode": "global",
            "locus_agg": self.config.locus_agg,
            "percent_mode": self.config.percent_mode,
            "polynomial_degree": int(self.config.polynomial_degree),
            "baseline_start_sec": float(b0_start),
            "baseline_end_sec": float(b0_end),
            "global_coeffs": [float(c) for c in coeffs.tolist()],
        }
        return y_fit.astype(float), diagnostics

    # ---------------------------------------------------------------------
    # Per-cycle fit (extension)
    # ---------------------------------------------------------------------

    def _fit_per_cycle(self, signal: SegmentSignal, d_abs: np.ndarray) -> tuple[np.ndarray, dict]:
        s_hat_full = np.full(signal.T, np.nan, dtype=float)

        cycle_coeffs: List[List[float]] = []
        cycle_windows: List[dict] = []

        for cyc in signal.protocol.cycles:
            flicker_start = float(cyc.flicker.start_sec)

            # Fit window: the cycle window as defined by the protocol
            cycle_start = (
                float(cyc.baseline.start_sec)
                if cyc.baseline is not None
                else flicker_start - float(self.config.baseline_start_before_flicker_sec)
            )
            cycle_end = float(cyc.recovery.end_sec)

            idx_cycle = np.where((signal.t >= cycle_start) & (signal.t < cycle_end))[0]
            if idx_cycle.size == 0:
                if self.config.strict_cycle_bounds:
                    raise ValueError(f"Empty cycle window for cycle {cyc.index}.")
                continue

            # Baseline window is explicitly [flicker_start-30, flicker_start-5]
            b0_start = flicker_start - float(self.config.baseline_start_before_flicker_sec)
            b0_end = flicker_start - float(self.config.baseline_end_before_flicker_sec)

            idx_base = np.where((signal.t >= b0_start) & (signal.t < b0_end))[0]
            if idx_base.size == 0:
                raise ValueError(
                    f"Missing baseline samples for cycle {cyc.index}. "
                    f"Expected baseline window [{b0_start:.3f}, {b0_end:.3f}) s."
                )

            baseline_level = float(np.nanmedian(d_abs[idx_base]))
            if not np.isfinite(baseline_level) or baseline_level == 0.0:
                raise ValueError(
                    f"Invalid baseline for cycle {cyc.index}: median baseline={baseline_level}."
                )

            t_cycle = signal.t[idx_cycle]
            d_cycle = d_abs[idx_cycle]

            y_cycle = self._to_percent(d_cycle, baseline_level, mode=self.config.percent_mode)

            ok = np.isfinite(t_cycle) & np.isfinite(y_cycle)
            if int(ok.sum()) < (self.config.polynomial_degree + 1):
                raise ValueError(
                    f"Not enough valid samples to fit degree {self.config.polynomial_degree} polynomial "
                    f"in cycle {cyc.index}. Need at least {self.config.polynomial_degree + 1}, got {int(ok.sum())}."
                )

            coeffs = np.polyfit(t_cycle[ok], y_cycle[ok], deg=int(self.config.polynomial_degree))
            y_fit = np.polyval(coeffs, t_cycle)

            s_hat_full[idx_cycle] = y_fit

            cycle_coeffs.append([float(c) for c in coeffs.tolist()])
            cycle_windows.append(
                {
                    "cycle_index": int(cyc.index),
                    "cycle_start_sec": float(cycle_start),
                    "cycle_end_sec": float(cycle_end),
                    "baseline_start_sec": float(b0_start),
                    "baseline_end_sec": float(b0_end),
                }
            )

        if len(cycle_coeffs) == 0:
            raise ValueError("No cycles could be processed for Gherghel polynomial preprocessing.")

        diagnostics = {
            "fit_mode": "per_cycle",
            "locus_agg": self.config.locus_agg,
            "percent_mode": self.config.percent_mode,
            "polynomial_degree": int(self.config.polynomial_degree),
            "cycle_coeffs": cycle_coeffs,
            "cycle_windows": cycle_windows,
        }
        return s_hat_full, diagnostics

    # ---------------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------------

    def _validate_protocol(self, signal: SegmentSignal) -> None:
        prot = signal.protocol
        if prot.cycles is None or len(prot.cycles) == 0:
            raise ValueError("GherghelPreprocessor requires cycles in StimulusProtocol.")

        for cyc in prot.cycles:
            flicker_start = float(cyc.flicker.start_sec)
            b0_start = flicker_start - float(self.config.baseline_start_before_flicker_sec)
            b0_end = flicker_start - float(self.config.baseline_end_before_flicker_sec)

            if b0_end <= b0_start:
                raise ValueError(
                    "Invalid baseline window definition. "
                    f"Computed [{b0_start:.3f}, {b0_end:.3f})."
                )

            if b0_start < float(signal.t[0]) - 1e-9:
                raise ValueError(
                    f"Baseline window starts before recording for cycle {cyc.index}. "
                    f"baseline_start={b0_start:.3f}, recording_start={float(signal.t[0]):.3f}."
                )
            if b0_end > float(signal.t[-1]) + 1e-9:
                raise ValueError(
                    f"Baseline window ends after recording for cycle {cyc.index}. "
                    f"baseline_end={b0_end:.3f}, recording_end={float(signal.t[-1]):.3f}."
                )

    # ---------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------

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
