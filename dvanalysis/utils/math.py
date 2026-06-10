"""Shared mathematical utilities for dvanalysis."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# MAD and robust scale
# ---------------------------------------------------------------------------

def mad_scale(x: np.ndarray) -> float:
    """Median absolute deviation scaled to be consistent with std for normal data.

    Returns 1.4826 * median(|x - median(x)|). Returns 0.0 if all values are identical.
    NaN values are ignored.
    """
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return 1.4826 * mad


# ---------------------------------------------------------------------------
# Nanstat helpers
# ---------------------------------------------------------------------------

def nanstat(x: np.ndarray, stat: str) -> float:
    """Apply a named statistic ignoring NaNs.

    Parameters
    ----------
    x : array
    stat : one of "mean", "median", "max", "min"

    Returns
    -------
    float — NaN if *x* is empty.
    """
    if x.size == 0:
        return float("nan")
    if stat == "mean":
        return float(np.nanmean(x))
    if stat == "median":
        return float(np.nanmedian(x))
    if stat == "max":
        return float(np.nanmax(x))
    if stat == "min":
        return float(np.nanmin(x))
    raise ValueError(f"Unknown stat '{stat}'")


def safe_nanargmax(x: np.ndarray) -> Optional[int]:
    """Argmax ignoring NaNs. Returns None if all values are NaN."""
    if not np.any(np.isfinite(x)):
        return None
    return int(np.nanargmax(x))


def safe_nanargmin(x: np.ndarray) -> Optional[int]:
    """Argmin ignoring NaNs. Returns None if all values are NaN."""
    if not np.any(np.isfinite(x)):
        return None
    return int(np.nanargmin(x))


# ---------------------------------------------------------------------------
# Missing value interpolation
# ---------------------------------------------------------------------------

def fill_missing_interp(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Linear interpolation per column (locus) for missing entries.

    Requires at least 2 valid points per column; otherwise leaves as-is.

    Parameters
    ----------
    X : (T, P) data matrix
    M : (T, P) boolean mask (True = valid)

    Returns
    -------
    (T, P) filled matrix (copy).
    """
    T, P = X.shape
    out = X.copy().astype(float)
    idx = np.arange(T)

    for p in range(P):
        valid = M[:, p].astype(bool)
        if valid.all() or int(valid.sum()) < 2:
            continue
        y = out[:, p]
        miss = ~valid
        y[miss] = np.interp(idx[miss], idx[valid], y[valid])
        out[:, p] = y

    return out


# ---------------------------------------------------------------------------
# Percent conversion
# ---------------------------------------------------------------------------

def to_percent(
    y: np.ndarray,
    baseline: float,
    mode: str = "delta_over_baseline",
) -> np.ndarray:
    """Convert *y* to percent relative to *baseline*.

    Parameters
    ----------
    y : array
    baseline : scalar reference value
    mode : "delta_over_baseline" → 100*(y-b)/b, or "ratio" → 100*y/b

    Returns
    -------
    array of same shape, all NaN if baseline is zero or non-finite.
    """
    if (not np.isfinite(baseline)) or baseline == 0.0:
        return np.full_like(y, np.nan, dtype=float)
    if mode == "delta_over_baseline":
        return 100.0 * ((y - baseline) / baseline)
    if mode == "ratio":
        return 100.0 * (y / baseline)
    raise ValueError(f"Unknown percent mode '{mode}'. Use 'delta_over_baseline' or 'ratio'.")


# ---------------------------------------------------------------------------
# Hampel filter
# ---------------------------------------------------------------------------

def hampel_filter_1d(
    y: np.ndarray,
    *,
    k: int = 9,
    nsigmas: float = 3.0,
    replace: str = "median",
) -> Tuple[np.ndarray, np.ndarray]:
    """Hampel filter (NaN-aware) for 1-D arrays.

    Parameters
    ----------
    y : (T,) array
    k : half-window size (total window = 2k+1)
    nsigmas : threshold in MAD units
    replace : "median" to replace outliers with window median, or "nan"

    Returns
    -------
    y_out : filtered signal
    is_outlier : boolean mask of detected outliers
    """
    y = np.asarray(y, float)
    T = y.size
    y_out = y.copy()
    is_out = np.zeros(T, dtype=bool)

    if T == 0 or k <= 0:
        return y_out, is_out

    c = 1.4826  # MAD-to-std consistency factor for normal distribution

    for i in range(T):
        if not np.isfinite(y[i]):
            continue

        lo = max(0, i - k)
        hi = min(T, i + k + 1)
        w = y[lo:hi]
        w = w[np.isfinite(w)]
        if w.size < 3:
            continue

        med = float(np.median(w))
        mad = float(np.median(np.abs(w - med)))
        sigma = c * mad

        if not np.isfinite(sigma) or sigma <= 0.0:
            continue

        if np.abs(y[i] - med) > nsigmas * sigma:
            is_out[i] = True
            if replace == "median":
                y_out[i] = med
            elif replace == "nan":
                y_out[i] = np.nan
            else:
                raise ValueError("replace must be 'median' or 'nan'.")

    return y_out, is_out
