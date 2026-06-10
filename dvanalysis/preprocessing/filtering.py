"""Signal filtering utilities — heartbeat PSD, lowpass, bandstop, mask-aware filtering."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import welch, butter, filtfilt

from dvanalysis.utils.masks import contiguous_true_runs


def extract_heartbeat_psd(
    signal: np.ndarray,
    timestep_ms: float,
    mask: Optional[np.ndarray] = None,
    min_freq_hz: float = 0.5,
    max_freq_hz: float = 2.0,
    nperseg: int = 256,
    min_block_len: int = 64,
) -> Tuple[float, float]:
    """Extract heartbeat rate and PSD amplitude using Welch's method.

    PSD is computed on the longest contiguous valid block (no interpolation).

    Returns
    -------
    bpm : float
        Heartbeat rate in beats per minute, or NaN if not estimable.
    peak_amplitude : float
        Peak PSD amplitude in the heartbeat band, or NaN.
    """
    y = np.asarray(signal, dtype=float).ravel()
    T = int(y.size)

    if T == 0:
        return float("nan"), float("nan")

    if mask is None:
        m = np.isfinite(y)
    else:
        m = np.asarray(mask, dtype=bool).ravel()
        if m.size != T:
            raise ValueError("mask must have same length as signal.")
        m = m & np.isfinite(y)

    runs = contiguous_true_runs(m)
    if not runs:
        return float("nan"), float("nan")

    run = max(runs, key=lambda r: r.size)
    if run.size < int(min_block_len):
        return float("nan"), float("nan")

    seg = y[run]
    if not np.all(np.isfinite(seg)):
        return float("nan"), float("nan")

    sampling_rate = 1000.0 / float(timestep_ms)
    nper = int(min(nperseg, seg.size))
    if nper < 8:
        return float("nan"), float("nan")

    freqs, psd = welch(seg, fs=sampling_rate, nperseg=nper)

    valid = np.where((freqs >= min_freq_hz) & (freqs <= max_freq_hz))[0]
    if valid.size == 0:
        return float("nan"), float("nan")

    peak_index = valid[int(np.argmax(psd[valid]))]
    peak_frequency = float(freqs[peak_index])
    peak_amplitude = float(psd[peak_index])

    return float(peak_frequency * 60.0), float(peak_amplitude)


def bandstop_filter(
    signal: np.ndarray,
    lowcut: float,
    highcut: float,
    sampling_rate: float,
    order: int = 4,
) -> np.ndarray:
    """Butterworth bandstop filter. Requires fully observed input (no NaNs)."""
    y = np.asarray(signal, dtype=float)
    if not np.all(np.isfinite(y)):
        raise ValueError("bandstop_filter requires finite input without NaNs.")
    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="bandstop")
    return filtfilt(b, a, y)


def lowpass_filter(
    signal: np.ndarray,
    cutoff: float,
    sampling_rate: float,
    order: int = 4,
) -> np.ndarray:
    """Butterworth lowpass filter. Requires fully observed input (no NaNs)."""
    y = np.asarray(signal, dtype=float)
    if not np.all(np.isfinite(y)):
        raise ValueError("lowpass_filter requires finite input without NaNs.")
    nyquist = 0.5 * sampling_rate
    normalized_cutoff = cutoff / nyquist
    b, a = butter(order, normalized_cutoff, btype="low")
    return filtfilt(b, a, y)


def _apply_edge_taper(seg: np.ndarray, taper_len: int) -> np.ndarray:
    """Apply a short Hann taper to reduce edge ringing on short blocks."""
    n = int(seg.size)
    tl = int(max(0, taper_len))
    if tl <= 0 or n < 2 * tl + 1:
        return seg

    w = np.hanning(2 * tl)
    w_in = w[:tl]
    w_out = w[tl:]

    out = seg.copy()
    out[:tl] = out[:tl] * w_in
    out[-tl:] = out[-tl:] * w_out
    return out


def lowpass_filter_masked(
    signal: np.ndarray,
    mask: np.ndarray,
    cutoff: float,
    sampling_rate: float,
    order: int = 4,
    min_block_len: Optional[int] = None,
    padlen: Optional[int] = None,
    taper_len: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mask-aware lowpass Butterworth — filters only contiguous valid blocks.

    Returns
    -------
    lp : lowpass output (NaN where mask is False)
    hi : residual (signal - lp on valid entries, NaN elsewhere)
    """
    y = np.asarray(signal, dtype=float).ravel()
    m = np.asarray(mask, dtype=bool).ravel()

    if y.ndim != 1:
        raise ValueError("signal must be 1D.")
    if m.ndim != 1 or m.size != y.size:
        raise ValueError("mask must be 1D and same length as signal.")

    T = int(y.size)
    lp = np.full(T, np.nan, dtype=float)

    nyq = 0.5 * sampling_rate
    norm_cutoff = cutoff / nyq

    valid = m & np.isfinite(y)

    if not np.isfinite(norm_cutoff) or norm_cutoff <= 0.0 or norm_cutoff >= 1.0:
        lp[valid] = y[valid]
        hi = np.full(T, np.nan, dtype=float)
        hi[valid] = y[valid] - lp[valid]
        return lp, hi

    b, a = butter(order, norm_cutoff, btype="low")

    default_min_len = 3 * (max(len(a), len(b)) - 1) + 1
    min_len = int(default_min_len if min_block_len is None else min_block_len)

    runs = contiguous_true_runs(valid)
    for run in runs:
        if run.size < min_len:
            lp[run] = y[run]
            continue

        seg = y[run]
        seg2 = _apply_edge_taper(seg, taper_len=int(taper_len)) if taper_len else seg

        local_padlen = padlen
        if local_padlen is not None:
            local_padlen = int(local_padlen)
            if local_padlen >= seg2.size:
                local_padlen = max(0, int(seg2.size) - 1)

        try:
            lp_seg = filtfilt(b, a, seg2, padlen=local_padlen) if local_padlen is not None else filtfilt(b, a, seg2)
        except ValueError:
            lp_seg = seg
        lp[run] = lp_seg

    hi = np.full(T, np.nan, dtype=float)
    ok = valid & np.isfinite(lp)
    hi[ok] = y[ok] - lp[ok]
    return lp, hi


def process_average_signal(
    average_signal: np.ndarray,
    weighted_signal: np.ndarray,
    timestep_ms: float,
    average_mask: Optional[np.ndarray] = None,
    weighted_mask: Optional[np.ndarray] = None,
    bandwidth_hz: float = 1.1,
    order: int = 4,
) -> np.ndarray:
    """Convenience: estimate heartbeat from weighted signal, bandstop-filter the average.

    Returns average_signal unchanged if heartbeat is not estimable or if gaps exist.
    """
    avg = np.asarray(average_signal, dtype=float).ravel()
    w = np.asarray(weighted_signal, dtype=float).ravel()

    if avg.size != w.size:
        raise ValueError("average_signal and weighted_signal must have same length.")

    if average_mask is None:
        m_avg = np.isfinite(avg)
    else:
        m_avg = np.asarray(average_mask, dtype=bool).ravel()
        if m_avg.size != avg.size:
            raise ValueError("average_mask must match average_signal length.")
        m_avg = m_avg & np.isfinite(avg)

    if weighted_mask is None:
        m_w = np.isfinite(w)
    else:
        m_w = np.asarray(weighted_mask, dtype=bool).ravel()
        if m_w.size != w.size:
            raise ValueError("weighted_mask must match weighted_signal length.")
        m_w = m_w & np.isfinite(w)

    hb_bpm, hb_amp = extract_heartbeat_psd(w, timestep_ms=timestep_ms, mask=m_w)
    if not np.isfinite(hb_bpm):
        return avg

    sampling_rate = 1000.0 / float(timestep_ms)
    heartbeat_freq = hb_bpm / 60.0

    lowcut = heartbeat_freq - bandwidth_hz / 2.0
    highcut = heartbeat_freq + bandwidth_hz / 2.0

    if not m_avg.all():
        return avg

    filtered_avg = bandstop_filter(avg, lowcut, highcut, sampling_rate, order=order)
    return filtered_avg + float(hb_amp)
