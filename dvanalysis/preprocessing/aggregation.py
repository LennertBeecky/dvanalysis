"""Locus aggregation, support gate, and post-processing utilities."""
from __future__ import annotations

import numpy as np


def aggregate_loci(x: np.ndarray, m: np.ndarray, mode: str = "median") -> np.ndarray:
    """Aggregate (T, P) matrix across loci to (T,) trace, respecting mask.

    Parameters
    ----------
    x : (T, P) data
    m : (T, P) boolean mask
    mode : "median" or "mean"
    """
    x_masked = x.copy()
    x_masked[~m] = np.nan
    if mode == "median":
        return np.nanmedian(x_masked, axis=1)
    if mode == "mean":
        return np.nanmean(x_masked, axis=1)
    raise ValueError(f"Unknown locus_agg '{mode}', use 'mean' or 'median'.")


def moving_average_cols(X: np.ndarray, window: int) -> np.ndarray:
    """Apply a centred moving average to each column of (T, P) matrix.

    Parameters
    ----------
    X : (T, P) matrix
    window : window size (must be >= 1)
    """
    if window <= 1:
        return X
    half = window // 2
    out = np.empty_like(X, dtype=float)
    T, P = X.shape
    for p in range(P):
        y = X[:, p]
        yy = np.empty(T, dtype=float)
        for i in range(T):
            a = max(0, i - half)
            b = min(T, i + half + 1)
            yy[i] = float(np.mean(y[a:b]))
        out[:, p] = yy
    return out


def segment_trace_from_loci(L: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Produce a (T,) vessel trace from (T, P) locus matrix, masking invalid entries."""
    tmp = L.astype(float).copy()
    tmp[~M.astype(bool)] = np.nan
    return np.nanmean(tmp, axis=1)
