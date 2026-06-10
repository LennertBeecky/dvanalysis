from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np

from .stimulus_protocol import StimulusProtocol
from .time_window import TimeWindow


@dataclass(frozen=True)
class SegmentSignal:
    """Multichannel DVA signal for one vessel segment.

    Parameters
    ----------
    t : ndarray, shape (T,)
        Time vector in seconds.
    x : ndarray, shape (T, P)
        Diameter measurements. NaN where invalid.
    m : ndarray, shape (T, P)
        Boolean observation mask (True = observed).
    protocol : StimulusProtocol
        Stimulus timing information.
    units : str
        Measurement units (default "a.u.").
    meta : dict
        Arbitrary metadata (source path, subject/visit, etc.).
    """

    t: np.ndarray
    x: np.ndarray
    m: np.ndarray
    protocol: StimulusProtocol
    units: str = "a.u."
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        t = np.asarray(self.t, dtype=float).copy()
        x = np.asarray(self.x, dtype=float).copy()
        m = np.asarray(self.m).copy()

        if t.ndim != 1:
            raise ValueError("t must be 1D with shape (T,).")
        if x.ndim != 2:
            raise ValueError("x must be 2D with shape (T,P).")
        if m.shape != x.shape:
            raise ValueError("m must have the same shape as x.")
        if x.shape[0] != t.shape[0]:
            raise ValueError("x and t must agree on T.")

        if m.dtype != bool:
            m = m.astype(bool)

        # Enforce NaNs where invalid — single source of truth
        x = x.copy()
        x[~m] = np.nan

        # Make arrays read-only
        t.setflags(write=False)
        x.setflags(write=False)
        m.setflags(write=False)

        object.__setattr__(self, "t", t)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "m", m)
        object.__setattr__(self, "meta", dict(self.meta))

    @property
    def T(self) -> int:
        """Number of timepoints."""
        return int(self.x.shape[0])

    @property
    def P(self) -> int:
        """Number of measurement loci."""
        return int(self.x.shape[1])

    def valid_fraction(self) -> float:
        """Fraction of observed entries in the mask."""
        return float(self.m.mean())

    def __str__(self) -> str:
        nan_frac = float(np.isnan(self.x).mean())
        return (
            f"SegmentSignal(T={self.T}, P={self.P}, units={self.units}, "
            f"valid_fraction={self.valid_fraction():.3f}, nan_fraction={nan_frac:.3f})"
        )

    def slice_window(self, w: TimeWindow) -> "SegmentSignal":
        """Return a new SegmentSignal restricted to the given time window."""
        idx = w.contains(self.t)
        return SegmentSignal(
            t=self.t[idx],
            x=self.x[idx, :],
            m=self.m[idx, :],
            protocol=self.protocol,
            units=self.units,
            meta=dict(self.meta),
        )

    def baseline_per_locus(self, w: TimeWindow, *, stat: str = "median") -> np.ndarray:
        """Compute per-locus baseline statistic over the given window."""
        xs = self.slice_window(w).x
        if stat == "mean":
            return np.nanmean(xs, axis=0)
        return np.nanmedian(xs, axis=0)

    def aggregate_over_loci(self, *, agg: str = "median") -> np.ndarray:
        """Aggregate across loci to a single (T,) trace, respecting the mask."""
        if agg == "mean":
            return np.nanmean(self.x, axis=1)
        return np.nanmedian(self.x, axis=1)
