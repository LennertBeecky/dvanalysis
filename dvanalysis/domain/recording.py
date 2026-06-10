from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable

from .segment import Segment


@dataclass
class Recording:
    """All vessel segments from one DVA recording session.

    Parameters
    ----------
    subject_id : str
        Subject identifier.
    visit_id : str
        Visit identifier.
    recording_datetime : str or None
        ISO datetime string if available.
    segments : dict
        Mapping from segment_label to Segment.
    meta : dict
        Arbitrary metadata.
    """

    subject_id: str
    visit_id: str
    recording_datetime: str | None = None
    segments: Dict[str, Segment] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def add_segment(self, seg: Segment) -> None:
        """Add a segment, keyed by its label."""
        self.segments[seg.segment_label] = seg

    def iter_segments(self) -> Iterable[Segment]:
        """Iterate over all segments."""
        return self.segments.values()

    def summary(self) -> str:
        lines = [
            "Recording summary",
            f"  Subject ID: {self.subject_id}",
            f"  Visit ID:   {self.visit_id}",
        ]
        if self.recording_datetime is not None:
            lines.append(f"  Datetime:   {self.recording_datetime}")
        lines.append(f"  #Segments:  {len(self.segments)}")

        for seg in self.iter_segments():
            seg_str = str(seg)
            for line in seg_str.splitlines():
                lines.append("    " + line)

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()
