"""Search window helpers for biomarker extraction."""
from __future__ import annotations

from typing import Tuple, Union

import numpy as np

from dvanalysis.domain import TimeWindow

WindowLike = Union[TimeWindow, Tuple[float, float]]


def as_window(w: WindowLike, phase: str = "custom") -> TimeWindow:
    """Coerce a (start, end) tuple to a TimeWindow."""
    if isinstance(w, TimeWindow):
        return w
    start_sec, end_sec = float(w[0]), float(w[1])
    return TimeWindow(phase=phase, start_sec=start_sec, end_sec=end_sec)


def idx_for_window(t: np.ndarray, w: WindowLike, *, strict: bool = True, label: str = "") -> np.ndarray:
    """Return indices of *t* that fall inside window *w*.

    Raises ValueError if strict=True and the selection is empty.
    """
    ww = as_window(w)
    idx = np.where(ww.contains(t))[0]
    if strict and idx.size == 0:
        msg = f"Empty window selection for {label or ww.phase}: [{ww.start_sec:.3f}, {ww.end_sec:.3f})"
        raise ValueError(msg)
    return idx
