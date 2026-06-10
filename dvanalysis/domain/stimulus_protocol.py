from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .time_window import TimeWindow
from .stimulus_cycle import StimulusCycle


@dataclass(frozen=True)
class StimulusProtocol:
    """DVA stimulus protocol describing timing of baseline, flicker, and recovery phases.

    Parameters
    ----------
    name : str
        Protocol identifier.
    fs : float
        Sampling rate in Hz.
    description : str
        Free-text description.
    global_baseline : TimeWindow or None
        Initial baseline window before the first cycle.
    cycles : list of StimulusCycle
        Ordered list of stimulus cycles.
    end_baseline : TimeWindow or None
        Optional trailing baseline after the last cycle.
    """

    name: str
    fs: float
    description: str = ""
    global_baseline: Optional[TimeWindow] = None
    cycles: List[StimulusCycle] = None
    end_baseline: Optional[TimeWindow] = None

    def all_windows(self) -> List[TimeWindow]:
        """Return every window in chronological order."""
        windows: List[TimeWindow] = []
        if self.global_baseline is not None:
            windows.append(self.global_baseline)
        if self.cycles is not None:
            for c in self.cycles:
                windows.extend(c.all_windows())
        if self.end_baseline is not None:
            windows.append(self.end_baseline)
        return windows

    def mask_for_phase(self, t: np.ndarray, phase: str) -> np.ndarray:
        """Return boolean mask selecting all timepoints that fall in windows with the given *phase*."""
        m = np.zeros_like(t, dtype=bool)

        if self.global_baseline is not None and phase == self.global_baseline.phase:
            m |= self.global_baseline.contains(t)

        if self.cycles is not None:
            for c in self.cycles:
                for w in c.all_windows():
                    if w.phase == phase:
                        m |= w.contains(t)

        if self.end_baseline is not None and phase == self.end_baseline.phase:
            m |= self.end_baseline.contains(t)

        return m

    def mask_for_cycle(self, t: np.ndarray, cycle_index: int) -> np.ndarray:
        """Return boolean mask for all windows in the cycle at *cycle_index*."""
        if self.cycles is None:
            raise ValueError("Protocol has no cycles.")
        c = self.cycles[cycle_index]
        m = np.zeros_like(t, dtype=bool)
        for w in c.all_windows():
            m |= w.contains(t)
        return m

    def flicker_windows(self) -> List[TimeWindow]:
        """Return flicker windows from all cycles."""
        if self.cycles is None:
            return []
        return [c.flicker for c in self.cycles]

    def recovery_windows(self) -> List[TimeWindow]:
        """Return recovery windows from all cycles."""
        if self.cycles is None:
            return []
        return [c.recovery for c in self.cycles]

    def baseline_windows(self) -> List[TimeWindow]:
        """Return all baseline windows (global + per-cycle + end)."""
        out: List[TimeWindow] = []
        if self.global_baseline is not None:
            out.append(self.global_baseline)
        if self.cycles is not None:
            for c in self.cycles:
                if c.baseline is not None:
                    out.append(c.baseline)
        if self.end_baseline is not None:
            out.append(self.end_baseline)
        return out

    def end_time_sec(self) -> float:
        """Return the latest end_sec across all windows."""
        ends = [w.end_sec for w in self.all_windows()]
        return float(max(ends)) if ends else 0.0
