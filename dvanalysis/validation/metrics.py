"""Evaluation metrics for DVA denoising: SDR, NMSE, biomarker bias, plateau stability."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from dvanalysis.domain import StimulusProtocol, TimeWindow
from dvanalysis.biomarkers.library import (
    calc_dilation_max_bc,
    calc_constr_max_bc,
    calc_da_bc,
    calc_dilation_max_t_bc,
    calc_dilation_max_30_t,
    calc_baseline_diameter,
)


# ---------------------------------------------------------------------------
# Signal-level metrics
# ---------------------------------------------------------------------------

def compute_sdr(gt: np.ndarray, recovered: np.ndarray) -> float:
    """Signal-to-distortion ratio in dB.

    Computed over finite entries only.
    """
    finite = np.isfinite(gt) & np.isfinite(recovered)
    if finite.sum() < 2:
        return float("nan")
    g = gt[finite]
    r = recovered[finite]
    sig_power = float(np.sum(g ** 2))
    err_power = float(np.sum((g - r) ** 2))
    if err_power == 0:
        return 60.0
    if sig_power == 0:
        return float("nan")
    return float(10.0 * np.log10(sig_power / err_power))


def compute_nmse(gt: np.ndarray, recovered: np.ndarray) -> float:
    """Normalised mean squared error.

    Computed over finite entries only.
    """
    finite = np.isfinite(gt) & np.isfinite(recovered)
    if finite.sum() < 2:
        return float("nan")
    g = gt[finite]
    r = recovered[finite]
    sig_power = float(np.sum(g ** 2))
    if sig_power == 0:
        return float("nan")
    return float(np.sum((g - r) ** 2) / sig_power)


def compute_delta_sdr(
    gt: np.ndarray,
    corrupted: np.ndarray,
    recovered: np.ndarray,
) -> float:
    """Improvement in SDR: SDR(output) - SDR(input)."""
    sdr_out = compute_sdr(gt, recovered)
    sdr_in = compute_sdr(gt, corrupted)
    if not np.isfinite(sdr_out) or not np.isfinite(sdr_in):
        return float("nan")
    return sdr_out - sdr_in


# ---------------------------------------------------------------------------
# Biomarker bias
# ---------------------------------------------------------------------------

def compute_biomarker_bias(
    gt: np.ndarray,
    recovered: np.ndarray,
    t: np.ndarray,
    protocol: StimulusProtocol,
    vessel_type: str = "artery",
) -> List[Dict[str, float]]:
    """Compute per-cycle biomarker bias (recovered - ground truth).

    Parameters
    ----------
    gt, recovered : (T,) traces in percent change from baseline.
    t : (T,) time vector.
    protocol : stimulus protocol.
    vessel_type : "artery" or "vein".

    Returns
    -------
    List of dicts, one per cycle, each mapping biomarker name to bias value.
    """
    is_artery = vessel_type.lower().startswith("a")
    biases = []

    for cyc in protocol.cycles:
        cycle_bias = {}

        # Search windows
        w_flicker_ext = TimeWindow("search", cyc.flicker.start_sec, cyc.flicker.end_sec + 10.0)
        w_baseline = cyc.baseline if cyc.baseline is not None else protocol.global_baseline
        w_recovery = cyc.recovery

        # MD: max dilation (baseline-corrected)
        try:
            md_gt = calc_dilation_max_bc(gt, t, w_flicker_ext, w_baseline, strict=False)
            md_rec = calc_dilation_max_bc(recovered, t, w_flicker_ext, w_baseline, strict=False)
            cycle_bias["MD"] = float(md_rec - md_gt)
        except Exception:
            cycle_bias["MD"] = float("nan")

        # MC: max constriction (arteries only)
        if is_artery:
            try:
                mc_gt = calc_constr_max_bc(gt, t, w_recovery, w_baseline, strict=False)
                mc_rec = calc_constr_max_bc(recovered, t, w_recovery, w_baseline, strict=False)
                cycle_bias["MC"] = float(mc_rec - mc_gt)
            except Exception:
                cycle_bias["MC"] = float("nan")

        # DA: dilation amplitude (arteries only)
        if is_artery:
            try:
                da_gt = calc_da_bc(gt, t, w_flicker_ext, w_recovery, w_baseline, strict=False)
                da_rec = calc_da_bc(recovered, t, w_flicker_ext, w_recovery, w_baseline, strict=False)
                cycle_bias["DA"] = float(da_rec - da_gt)
            except Exception:
                cycle_bias["DA"] = float("nan")

        # tMAD30: time to 30% of max dilation
        try:
            bd_gt = calc_baseline_diameter(gt, t, w_baseline, strict=False)
            t30_gt = calc_dilation_max_30_t(gt, t, w_flicker_ext, w_baseline, strict=False)
            bd_rec = calc_baseline_diameter(recovered, t, w_baseline, strict=False)
            t30_rec = calc_dilation_max_30_t(recovered, t, w_flicker_ext, w_baseline, strict=False)
            cycle_bias["tMAD30"] = float(t30_rec - t30_gt)
        except Exception:
            cycle_bias["tMAD30"] = float("nan")

        # tMAD: time to max dilation (argmax)
        try:
            tmad_gt = calc_dilation_max_t_bc(gt, t, w_flicker_ext, w_baseline,
                                              strict=False, relative_to_start=True)
            tmad_rec = calc_dilation_max_t_bc(recovered, t, w_flicker_ext, w_baseline,
                                               strict=False, relative_to_start=True)
            cycle_bias["tMAD"] = float(tmad_rec - tmad_gt)
        except Exception:
            cycle_bias["tMAD"] = float("nan")

        biases.append(cycle_bias)

    return biases


# ---------------------------------------------------------------------------
# Plateau stability
# ---------------------------------------------------------------------------

def compute_plateau_stability(
    gt: np.ndarray,
    recovered: np.ndarray,
    t: np.ndarray,
    protocol: StimulusProtocol,
    tolerance_sec: float = 0.5,
) -> float:
    """Fraction of cycles where |tMAD_recovered - tMAD_gt| > tolerance.

    Quantifies the instability of argmax-based timing on plateau-shaped responses.

    Parameters
    ----------
    gt, recovered : (T,) traces.
    t : (T,) time vector.
    protocol : stimulus protocol.
    tolerance_sec : threshold in seconds (default 0.5s = ~12 samples at 25 Hz).

    Returns
    -------
    Fraction of cycles exceeding the tolerance (0 to 1).
    """
    n_cycles = 0
    n_unstable = 0

    for cyc in protocol.cycles:
        w = TimeWindow("search", cyc.flicker.start_sec, cyc.flicker.end_sec + 10.0)
        w_bl = cyc.baseline if cyc.baseline is not None else protocol.global_baseline

        try:
            tmad_gt = calc_dilation_max_t_bc(gt, t, w, w_bl, strict=False, relative_to_start=True)
            tmad_rec = calc_dilation_max_t_bc(recovered, t, w, w_bl, strict=False, relative_to_start=True)

            if np.isfinite(tmad_gt) and np.isfinite(tmad_rec):
                n_cycles += 1
                if abs(tmad_rec - tmad_gt) > tolerance_sec:
                    n_unstable += 1
        except Exception:
            pass

    if n_cycles == 0:
        return float("nan")
    return float(n_unstable / n_cycles)
