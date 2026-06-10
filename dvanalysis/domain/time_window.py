from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TimeWindow:
    """A named time interval within a DVA recording.

    Parameters
    ----------
    phase : str
        Label for this window (e.g. "baseline", "flicker", "recovery").
    start_sec : float
        Start time in seconds (inclusive).
    end_sec : float
        End time in seconds (exclusive).
    """

    phase: str
    start_sec: float
    end_sec: float

    def contains(self, t: np.ndarray) -> np.ndarray:
        """Return boolean mask where ``start_sec <= t < end_sec``."""
        return (t >= self.start_sec) & (t < self.end_sec)

    def to_indices(self, t: np.ndarray) -> np.ndarray:
        """Return integer indices where *t* falls inside this window."""
        return np.where((t >= self.start_sec) & (t < self.end_sec))[0]

    def to_indices_from_fs(self, fs: float) -> np.ndarray:
        """Return integer sample indices computed from sampling rate *fs*."""
        start = int(round(self.start_sec * fs))
        end = int(round(self.end_sec * fs))
        return np.arange(start, end, dtype=int)
