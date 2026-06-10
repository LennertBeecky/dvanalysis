"""Low-level operations on masked signals for biomarker extraction."""
from __future__ import annotations

import numpy as np
from dvanalysis.domain import StimulusCycle


def masked_trace_mean(sig) -> np.ndarray:
    """Mean across loci with mask, returns (T,) in original units."""
    x = np.where(sig.m, sig.x, np.nan)
    return np.nanmean(x, axis=1)


def masked_trace_median(sig) -> np.ndarray:
    """Median across loci with mask, returns (T,) in original units."""
    x = np.where(sig.m, sig.x, np.nan)
    return np.nanmedian(x, axis=1)


def cycle_slice_indices(t: np.ndarray, start_sec: float, end_sec: float) -> np.ndarray:
    return np.where((t >= float(start_sec)) & (t < float(end_sec)))[0]


def max_dilation_and_amplitude_in_cycle(
    y: np.ndarray,
    t: np.ndarray,
    cycle: StimulusCycle,
) -> tuple[float, float]:
    """Compute max dilation and dilation amplitude for one cycle."""
    idx_f = cycle_slice_indices(t, cycle.flicker.start_sec, cycle.flicker.end_sec)
    idx_r = cycle_slice_indices(t, cycle.recovery.start_sec, cycle.recovery.end_sec)

    if idx_f.size == 0:
        raise ValueError(f"Empty flicker window for cycle {cycle.index}.")
    if idx_r.size == 0:
        raise ValueError(f"Empty recovery window for cycle {cycle.index}.")

    max_dil = float(np.nanmax(y[idx_f]))
    min_rec = float(np.nanmin(y[idx_r]))
    amp = float(max_dil - min_rec)
    return max_dil, amp
