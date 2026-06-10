"""Hybrid evaluation runner: generate samples, run methods, collect metrics."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from dvanalysis.domain import Segment, TimeWindow
from dvanalysis.preprocessing.rpca_denoise import MyMethodRPCA, MyMethodConfig
from dvanalysis.validation.hybrid import HybridBuilder, HybridSample
from dvanalysis.validation.metrics import (
    compute_sdr,
    compute_nmse,
    compute_delta_sdr,
    compute_biomarker_bias,
    compute_plateau_stability,
)


def _raw_median_trace(sig, baseline_window) -> np.ndarray:
    """Raw median across loci, expressed as percent change from baseline."""
    x = sig.x.copy()
    x[~sig.m] = np.nan
    y_abs = np.nanmedian(x, axis=1)
    bl_idx = np.where(baseline_window.contains(sig.t))[0]
    b0 = float(np.nanmedian(y_abs[bl_idx]))
    if not np.isfinite(b0) or b0 == 0:
        return y_abs
    return 100.0 * (y_abs - b0) / b0


def _sms_trace(sig, baseline_window, window_sec: float = 4.0) -> np.ndarray:
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

    bl_idx = np.where(baseline_window.contains(sig.t))[0]
    b0 = float(np.nanmedian(y_smooth[bl_idx]))
    if not np.isfinite(b0) or b0 == 0:
        return y_smooth
    return 100.0 * (y_smooth - b0) / b0


def _sms_avg_trace(sig, baseline_window, window_sec: float = 4.0) -> np.ndarray:
    """SMS-avg: SMS trace -> align cycles to flicker onset -> pointwise average -> tile back.

    Returns a full-length trace where the averaged cycle is tiled to all three cycle positions.
    """
    sms = _sms_trace(sig, baseline_window, window_sec)
    t = np.asarray(sig.t, float)
    protocol = sig.protocol
    cycles = protocol.cycles

    if not cycles:
        return sms

    # Extract per-cycle traces aligned to flicker onset
    cycle_traces = []
    cycle_lengths = []
    for cyc in cycles:
        bl_start = cyc.baseline.start_sec if cyc.baseline is not None else cyc.flicker.start_sec - 30.0
        cyc_end = cyc.recovery.end_sec
        idx = np.where((t >= bl_start) & (t < cyc_end))[0]
        if idx.size > 0:
            cycle_traces.append(sms[idx])
            cycle_lengths.append(idx.size)

    if not cycle_traces:
        return sms

    # Trim to shortest cycle length and average
    L = min(len(c) for c in cycle_traces)
    stack = np.stack([c[:L] for c in cycle_traces], axis=0)
    avg_cycle = np.nanmedian(stack, axis=0)

    # Tile the averaged cycle back to all three cycle positions
    y_tiled = np.full_like(sms, np.nan, dtype=float)
    for cyc in cycles:
        bl_start = cyc.baseline.start_sec if cyc.baseline is not None else cyc.flicker.start_sec - 30.0
        cyc_end = cyc.recovery.end_sec
        idx = np.where((t >= bl_start) & (t < cyc_end))[0]
        n = min(idx.size, L)
        y_tiled[idx[:n]] = avg_cycle[:n]

    return y_tiled


def _average_ground_truth_across_cycles(gt, t, protocol) -> np.ndarray:
    """Average the ground truth across cycles (for fair SMS-avg biomarker comparison).

    Returns a full-length trace with the cycle-averaged ground truth tiled back.
    """
    cycles = protocol.cycles
    if not cycles:
        return gt

    cycle_traces = []
    for cyc in cycles:
        bl_start = cyc.baseline.start_sec if cyc.baseline is not None else cyc.flicker.start_sec - 30.0
        cyc_end = cyc.recovery.end_sec
        idx = np.where((t >= bl_start) & (t < cyc_end))[0]
        if idx.size > 0:
            cycle_traces.append(gt[idx])

    if not cycle_traces:
        return gt

    L = min(len(c) for c in cycle_traces)
    stack = np.stack([c[:L] for c in cycle_traces], axis=0)
    avg_cycle = np.nanmedian(stack, axis=0)

    gt_avg = np.full_like(gt, np.nan, dtype=float)
    for cyc in cycles:
        bl_start = cyc.baseline.start_sec if cyc.baseline is not None else cyc.flicker.start_sec - 30.0
        cyc_end = cyc.recovery.end_sec
        idx = np.where((t >= bl_start) & (t < cyc_end))[0]
        n = min(idx.size, L)
        gt_avg[idx[:n]] = avg_cycle[:n]

    return gt_avg


def run_hybrid_evaluation(
    segments: Dict[str, Segment],
    n_configs: int = 10,
    seed: int = 42,
    rpca_max_iter: int = 70,
    rpca_lmb: float = 0.55,
    rpca_gamma: float = 1000.0,
) -> pd.DataFrame:
    """Run the full hybrid evaluation on a set of segments.

    Methods evaluated:
    - rpca: full RPCA with temporal smoothness
    - rpca_gamma0: RPCA ablation without smoothness (gamma=0)
    - sms: sliding median smoothing (per-cycle)
    - sms_avg: cycle-averaged SMS (tiled back to full length, compared against per-cycle GT)
    - sms_avg_fair: cycle-averaged SMS compared against cycle-averaged GT (best-case for SMS-avg)

    Parameters
    ----------
    segments : dict mapping segment_key to Segment.
    n_configs : number of response configurations per segment.
    seed : random seed.
    rpca_max_iter : RPCA iteration budget.
    rpca_lmb, rpca_gamma : RPCA hyperparameters.

    Returns
    -------
    DataFrame with one row per (segment, config, method).
    """
    builder = HybridBuilder(n_configs=n_configs, seed=seed)

    rpca_cfg = MyMethodConfig(
        rpca_lmb=rpca_lmb, rpca_gamma=rpca_gamma, rpca_max_iter=rpca_max_iter,
        rpca_tol_rel=1e-3, standardize=True, harmonize_output=True,
        harmonize_percent_mode="delta_over_baseline",
        harmonize_baseline_source="protocol_global",
        harmonize_aggregation_order="percent_then_aggregate",
        harmonize_baseline_per_locus=True,
        support_min_valid_frac=0.75,
        support_min_valid_abs=12, hampel_enable=True,
    )
    rpca = MyMethodRPCA(config=rpca_cfg)

    rpca_ns_cfg = MyMethodConfig(
        rpca_lmb=rpca_lmb, rpca_gamma=0.0, rpca_max_iter=rpca_max_iter,
        rpca_tol_rel=1e-3, standardize=True, harmonize_output=True,
        harmonize_percent_mode="delta_over_baseline",
        harmonize_baseline_source="protocol_global",
        harmonize_aggregation_order="percent_then_aggregate",
        harmonize_baseline_per_locus=True,
        support_min_valid_frac=0.75,
        support_min_valid_abs=12, hampel_enable=True,
    )
    rpca_no_smooth = MyMethodRPCA(config=rpca_ns_cfg)

    rows: List[Dict[str, Any]] = []

    try:
        from tqdm import tqdm
        seg_iter = tqdm(segments.items(), desc="Hybrid evaluation", unit="seg")
    except ImportError:
        seg_iter = segments.items()

    for seg_key, seg in seg_iter:
        vessel_type = seg.vessel_type
        samples = builder.build_for_segment(seg.signal, vessel_type)

        for hs in samples:
            gt = hs.ground_truth
            t = np.asarray(hs.signal.t, float)
            protocol = hs.signal.protocol
            bl_window = protocol.global_baseline or protocol.cycles[0].baseline

            corrupted = _raw_median_trace(hs.signal, bl_window)

            # Per-cycle methods: compare against per-cycle ground truth
            per_cycle_methods = {}
            per_cycle_methods["sms"] = _sms_trace(hs.signal, bl_window)

            try:
                per_cycle_methods["rpca"] = rpca.run(hs.signal).s_hat
            except Exception:
                pass

            try:
                per_cycle_methods["rpca_gamma0"] = rpca_no_smooth.run(hs.signal).s_hat
            except Exception:
                pass

            for method_name, recovered in per_cycle_methods.items():
                row = _build_row(seg_key, vessel_type, hs, method_name,
                                 gt, corrupted, recovered, t, protocol)
                rows.append(row)

            # SMS-avg: tiled back, compared against per-cycle GT (Option C)
            sms_avg = _sms_avg_trace(hs.signal, bl_window)
            row_c = _build_row(seg_key, vessel_type, hs, "sms_avg",
                               gt, corrupted, sms_avg, t, protocol)
            rows.append(row_c)

            # SMS-avg fair: compared against cycle-averaged GT (Option D)
            gt_avg = _average_ground_truth_across_cycles(gt, t, protocol)
            row_d = _build_row(seg_key, vessel_type, hs, "sms_avg_fair",
                               gt_avg, corrupted, sms_avg, t, protocol)
            rows.append(row_d)

    return pd.DataFrame(rows)


def _build_row(
    seg_key: str,
    vessel_type: str,
    hs: HybridSample,
    method_name: str,
    gt: np.ndarray,
    corrupted: np.ndarray,
    recovered: np.ndarray,
    t: np.ndarray,
    protocol,
    support_frac: float = 0.75,
) -> Dict[str, Any]:
    """Build one result row for a (segment, config, method) combination."""
    # Apply common support gate: restrict all metrics to timepoints where
    # at least support_frac of loci were observed, so that all methods are
    # evaluated on the same set of timepoints.
    m = np.asarray(hs.signal.m, bool)
    frac_valid = m.sum(axis=1) / m.shape[1]
    low_support = frac_valid < support_frac
    gt_masked = gt.copy()
    rec_masked = recovered.copy()
    cor_masked = corrupted.copy()
    gt_masked[low_support] = np.nan
    rec_masked[low_support] = np.nan
    cor_masked[low_support] = np.nan

    row = {
        "segment_key": seg_key,
        "vessel_type": vessel_type,
        "config_index": hs.params.get("config_index", -1),
        "method": method_name,
        "SDR": compute_sdr(gt_masked, rec_masked),
        "NMSE": compute_nmse(gt_masked, rec_masked),
        "delta_SDR": compute_delta_sdr(gt_masked, cor_masked, rec_masked),
        "plateau_stability": compute_plateau_stability(gt, recovered, t, protocol),
    }

    biases = compute_biomarker_bias(gt, recovered, t, protocol, vessel_type)
    for c_idx, cyc_bias in enumerate(biases):
        for bio_name, bias_val in cyc_bias.items():
            row[f"bias_{bio_name}_c{c_idx}"] = bias_val

    return row
