"""Baseline-aware centering and robust MAD scaling."""
from __future__ import annotations

from typing import Tuple

import numpy as np


def baseline_center_and_scale(
    X: np.ndarray,
    M: np.ndarray,
    baseline_idx: np.ndarray,
    scale_mode: str = "global",
    scale_floor: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre by per-locus baseline median and scale by MAD.

    Parameters
    ----------
    X : (T, P) data matrix
    M : (T, P) boolean mask
    baseline_idx : 1-D int array of baseline sample indices
    scale_mode : "global" (pooled MAD across all loci) or "per_locus"
    scale_floor : minimum allowed scale value

    Returns
    -------
    Z : (T, P) standardised matrix
    mu : (1, P) per-locus median
    sd : (1, P) scale factors
    """
    X = X.astype(float)
    M = M.astype(bool)
    M_eff = M & np.isfinite(X)

    bidx = np.asarray(baseline_idx, dtype=int).ravel()
    if bidx.size == 0:
        raise ValueError("baseline_idx is empty.")
    if np.any((bidx < 0) | (bidx >= X.shape[0])):
        raise ValueError("baseline_idx contains indices outside [0, T).")

    Xm = X.copy()
    Xm[~M_eff] = np.nan
    Xb = Xm[bidx, :]
    mu = np.nanmedian(Xb, axis=0, keepdims=True)

    rb = Xb - mu

    if scale_mode == "global":
        pool = rb[np.isfinite(rb)]
        if pool.size == 0:
            sd_global = 1.0
        else:
            med = float(np.median(pool))
            mad = float(np.median(np.abs(pool - med)))
            sd_global = 1.4826 * mad
            if (not np.isfinite(sd_global)) or sd_global <= 0.0:
                sd_global = float(np.std(pool)) if pool.size > 1 else 1.0
                if (not np.isfinite(sd_global)) or sd_global <= 0.0:
                    sd_global = 1.0
        sd = np.full((1, X.shape[1]), max(float(sd_global), float(scale_floor)), dtype=float)

    elif scale_mode == "per_locus":
        sd = np.empty((1, X.shape[1]), dtype=float)
        for p in range(X.shape[1]):
            r = rb[:, p]
            r = r[np.isfinite(r)]
            if r.size == 0:
                s = 1.0
            else:
                med = float(np.median(r))
                mad = float(np.median(np.abs(r - med)))
                s = 1.4826 * mad
                if (not np.isfinite(s)) or s <= 0.0:
                    s = float(np.std(r)) if r.size > 1 else 1.0
                    if (not np.isfinite(s)) or s <= 0.0:
                        s = 1.0
            sd[0, p] = max(float(s), float(scale_floor))

    else:
        raise ValueError("scale_mode must be 'global' or 'per_locus'.")

    Z = (X - mu) / sd
    return Z, mu, sd
