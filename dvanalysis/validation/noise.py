"""Noise and residual extraction for DVA hybrid ground-truth construction.

Extracts the baseline residual (cardiac pulsatility + measurement noise +
vasomotion + micro-saccade artefacts) from non-flicker baseline windows.
No attempt is made to separate cardiac from noise; the residual preserves
the full temporal structure of real DVA measurements.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from dvanalysis.domain import SegmentSignal, TimeWindow


def extract_baseline_residual(
    sig: SegmentSignal,
    baseline_windows: List[TimeWindow],
) -> np.ndarray:
    """Extract per-locus baseline residual from multiple baseline windows.

    For each locus, the residual is the raw signal minus the per-locus
    median (resting diameter). NaN is preserved where the locus was not
    observed. No cardiac separation is performed.

    Parameters
    ----------
    sig : SegmentSignal
        The raw DVA signal.
    baseline_windows : list of TimeWindow
        All non-flicker baseline epochs (e.g. global baseline + 3 pre-flicker baselines).

    Returns
    -------
    (T_total, P) residual array. NaN where mask is False.
    """
    t = np.asarray(sig.t, float)
    x = np.asarray(sig.x, float).copy()
    m = np.asarray(sig.m, bool)
    x[~m] = np.nan

    # Each baseline block is independently centered on its own per-locus
    # median. This removes inter-baseline drift (vasomotion, fatigue) that
    # tiling cannot represent and would otherwise create DC jumps.
    blocks = []
    for w in baseline_windows:
        idx = np.where(w.contains(t))[0]
        if len(idx) == 0:
            continue
        block = x[idx, :]
        block_median = np.nanmedian(block, axis=0)  # (P,)
        blocks.append(block - block_median[np.newaxis, :])

    return np.concatenate(blocks, axis=0)


def tile_residual(
    residual_block: np.ndarray,
    target_length: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Tile a residual block to the desired length.

    When *rng* is provided each repetition is circularly shifted by a random
    offset applied uniformly across all loci, breaking the strict periodicity
    that deterministic tiling would introduce while preserving inter-locus
    coherence (e.g. cardiac phase). NaN entries are preserved and tile along
    with the real values.

    Parameters
    ----------
    residual_block : (T_block, P) residual array (may contain NaN).
    target_length : desired number of rows.
    rng : optional random generator for randomised circular shifts.

    Returns
    -------
    (target_length, P) tiled residual array (NaN preserved).
    """
    T_block, P = residual_block.shape

    if target_length <= T_block:
        return residual_block[:target_length, :].copy()

    n_reps = int(np.ceil(target_length / T_block))

    if rng is None:
        tiled = np.tile(residual_block, (n_reps, 1))
        return tiled[:target_length, :].copy()

    pieces = []
    for _ in range(n_reps):
        block = residual_block.copy()
        # Same shift for all loci to preserve inter-locus coherence (cardiac phase)
        shift = int(rng.integers(0, T_block))
        block = np.roll(block, shift, axis=0)
        pieces.append(block)
    tiled = np.concatenate(pieces, axis=0)
    return tiled[:target_length, :].copy()


def build_flicker_mask(
    t: np.ndarray,
    protocol,
    fs: float = 25.0,
    flicker_freq: float = 12.5,
) -> np.ndarray:
    """Build a deterministic flicker dropout mask.

    During flicker periods, every other frame is a dark frame (no measurement).
    Outside flicker, all frames are observable.

    Parameters
    ----------
    t : (T,) time vector in seconds.
    protocol : StimulusProtocol with cycles.
    fs : sampling rate (Hz).
    flicker_freq : flicker frequency (Hz). At fs=25 and flicker_freq=12.5,
        every other frame during flicker is masked.

    Returns
    -------
    (T,) boolean array. True = observable, False = dark frame.
    """
    mask = np.ones(len(t), dtype=bool)
    samples_per_flicker = int(round(fs / flicker_freq))  # 2 at 25/12.5

    for cyc in protocol.cycles:
        fl_idx = np.where(
            (t >= cyc.flicker.start_sec) & (t < cyc.flicker.end_sec)
        )[0]
        if len(fl_idx) == 0:
            continue
        # Mask every other frame within this flicker period
        for i, idx in enumerate(fl_idx):
            if i % samples_per_flicker != 0:
                mask[idx] = False

    return mask


# ---------------------------------------------------------------------------
# Legacy API (kept for backward compatibility with existing tests)
# ---------------------------------------------------------------------------

def extract_cardiac(
    sig: SegmentSignal,
    baseline_window: TimeWindow,
    bandwidth_hz: float = 1.0,
) -> np.ndarray:
    """Extract the cardiac pulsatile component from a DVA signal.

    DEPRECATED: Use extract_baseline_residual instead, which preserves
    cardiac pulsatility without separating it from noise.
    """
    from dvanalysis.preprocessing.filtering import (
        extract_heartbeat_psd,
        lowpass_filter_masked,
    )

    t = np.asarray(sig.t, float)
    x = np.asarray(sig.x, float).copy()
    m = np.asarray(sig.m, bool)
    T, P = x.shape
    fs = float(sig.protocol.fs)
    timestep_ms = 1000.0 / fs

    bl_idx = np.where(baseline_window.contains(t))[0]
    x_masked = x.copy()
    x_masked[~m] = np.nan
    y_avg = np.nanmean(x_masked[bl_idx, :], axis=1)
    m_avg = np.any(m[bl_idx, :], axis=1)

    hb_bpm, _ = extract_heartbeat_psd(y_avg, timestep_ms=timestep_ms, mask=m_avg)

    cardiac = np.full_like(x, np.nan, dtype=float)

    if not np.isfinite(hb_bpm):
        cardiac[m] = 0.0
        return cardiac

    hb_hz = hb_bpm / 60.0
    cutoff_low = max(0.3, hb_hz - bandwidth_hz / 2.0)
    cutoff_high = hb_hz + bandwidth_hz / 2.0

    for p in range(P):
        y_p = x[:, p]
        m_p = m[:, p]

        lp_high, _ = lowpass_filter_masked(y_p, m_p, cutoff=cutoff_high,
                                            sampling_rate=fs, order=4)
        lp_low, _ = lowpass_filter_masked(y_p, m_p, cutoff=cutoff_low,
                                           sampling_rate=fs, order=4)

        valid = np.isfinite(lp_high) & np.isfinite(lp_low)
        cardiac[valid, p] = lp_high[valid] - lp_low[valid]

    return cardiac


def extract_baseline_noise(
    sig: SegmentSignal,
    baseline_window: TimeWindow,
    bandwidth_hz: float = 1.0,
) -> np.ndarray:
    """Extract baseline noise residual after removing offset and cardiac.

    DEPRECATED: Use extract_baseline_residual instead.
    """
    t = np.asarray(sig.t, float)
    x = np.asarray(sig.x, float).copy()
    m = np.asarray(sig.m, bool)

    bl_idx = np.where(baseline_window.contains(t))[0]

    x_masked = x.copy()
    x_masked[~m] = np.nan
    offsets = np.nanmedian(x_masked[bl_idx, :], axis=0)
    residual = x_masked[bl_idx, :] - offsets[np.newaxis, :]

    cardiac_full = extract_cardiac(sig, baseline_window, bandwidth_hz=bandwidth_hz)
    cardiac_bl = cardiac_full[bl_idx, :]

    valid = np.isfinite(residual) & np.isfinite(cardiac_bl)
    noise = np.full_like(residual, np.nan, dtype=float)
    noise[valid] = residual[valid] - cardiac_bl[valid]

    return noise


def tile_noise(
    noise_block: np.ndarray,
    target_length: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Tile a noise block to the desired length.

    DEPRECATED: Use tile_residual instead.
    """
    return tile_residual(noise_block, target_length, rng=rng)
