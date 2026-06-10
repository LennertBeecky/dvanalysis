from dataclasses import dataclass
from typing import List, Optional

from .time_window import TimeWindow


@dataclass(frozen=True)
class StimulusCycle:
    """One cycle of the DVA flicker stimulus protocol.

    Parameters
    ----------
    index : int
        Zero-based cycle index.
    baseline : TimeWindow or None
        Per-cycle baseline window (may be absent for some protocols).
    flicker : TimeWindow
        Flicker stimulation window (always present).
    recovery : TimeWindow
        Post-flicker recovery window (always present).
    """

    index: int
    baseline: Optional[TimeWindow]
    flicker: TimeWindow
    recovery: TimeWindow

    def all_windows(self) -> List[TimeWindow]:
        """Return all non-None windows in chronological order."""
        return [w for w in [self.baseline, self.flicker, self.recovery] if w is not None]
