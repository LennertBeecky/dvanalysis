from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from .segment_signal import SegmentSignal


@dataclass
class Segment:
    """One vessel segment within a DVA recording.

    Parameters
    ----------
    segment_label : str
        Segment identifier (e.g. "A1", "V3", "Art2").
    vessel_type : str
        "artery", "vein", or "unknown".
    signal : SegmentSignal
        The multichannel measurement data.
    meta : dict
        Arbitrary metadata.
    """

    segment_label: str
    vessel_type: str
    signal: SegmentSignal
    meta: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        s = self.signal
        return (
            f"Segment {self.segment_label} ({self.vessel_type})\n"
            f"  T = {s.T}, P = {s.P}\n"
            f"  Valid fraction = {s.valid_fraction():.3f}\n"
            f"  Units = {s.units}"
        )

    def __str__(self) -> str:
        return self.summary()
