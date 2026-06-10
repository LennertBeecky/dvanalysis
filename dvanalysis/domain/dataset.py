from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .recording import Recording
from .segment import Segment


@dataclass
class Dataset:
    """Collection of DVA recordings.

    Parameters
    ----------
    name : str
        Dataset identifier.
    description : str
        Free-text description.
    root_path : Path or None
        Root directory of the source data.
    cache_path : Path or None
        Optional cache directory.
    config : dict
        Reader/processing config used to build this dataset.
    recordings : list of Recording
        The recordings in this dataset.
    """

    name: str
    description: str = ""
    root_path: Optional[Path] = None
    cache_path: Optional[Path] = None
    config: Dict[str, Any] = field(default_factory=dict)
    recordings: List[Recording] = field(default_factory=list)

    def add_recording(self, rec: Recording) -> None:
        """Append a recording."""
        self.recordings.append(rec)

    def iter_segments(self) -> Iterable[Tuple[Recording, Segment]]:
        """Iterate over (recording, segment) pairs across all recordings."""
        for rec in self.recordings:
            for seg in rec.iter_segments():
                yield rec, seg

    def get_recording(self, subject_id: str, visit_id: str) -> Recording:
        """Look up a recording by subject and visit ID.

        Raises
        ------
        KeyError
            If no matching recording is found.
        """
        for rec in self.recordings:
            if str(rec.subject_id) == str(subject_id) and str(rec.visit_id) == str(visit_id):
                return rec
        raise KeyError(f"Recording not found for subject_id={subject_id}, visit_id={visit_id}")
