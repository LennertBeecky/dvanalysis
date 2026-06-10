"""Mask manipulation utilities for dvanalysis."""
from __future__ import annotations

from typing import List, Optional

import numpy as np


def contiguous_true_runs(mask: np.ndarray) -> List[np.ndarray]:
    """Return a list of index arrays, each a contiguous run of True values in *mask*.

    Parameters
    ----------
    mask : 1-D boolean array

    Returns
    -------
    List of 1-D int arrays, each containing the indices of one contiguous True block.
    """
    m = np.asarray(mask, dtype=bool)
    idx = np.flatnonzero(m)
    if idx.size == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0] + 1
    return list(np.split(idx, splits))


def support_good_timepoints(M: np.ndarray, min_valid_abs: int) -> np.ndarray:
    """Return (T,) boolean mask where at least *min_valid_abs* loci are valid.

    Parameters
    ----------
    M : (T, P) boolean mask
    min_valid_abs : minimum number of valid loci required
    """
    nvalid = np.sum(M, axis=1)
    return nvalid >= int(min_valid_abs)


def resolve_min_valid_abs(
    M: np.ndarray,
    min_abs: int,
    min_frac: Optional[float],
) -> int:
    """Choose an absolute min-valid threshold.

    If *min_frac* is provided, compute ``ceil(min_frac * P)``; otherwise use *min_abs*.

    Parameters
    ----------
    M : (T, P) mask (used to read P)
    min_abs : absolute minimum count (used when min_frac is None)
    min_frac : fraction of total loci (e.g. 0.75 means 75%)
    """
    P = int(M.shape[1])
    if min_frac is not None:
        f = float(min_frac)
        if not (0.0 < f <= 1.0):
            raise ValueError("min_frac must be in (0, 1].")
        return int(np.ceil(f * P))
    return int(min_abs)
