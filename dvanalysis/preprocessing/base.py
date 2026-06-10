"""Abstract base classes for preprocessors."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Generic, TypeVar

from dvanalysis.domain import SegmentSignal, Recording
from .configs import PreprocessorConfig
from .result import PreprocessResult

C = TypeVar("C", bound=PreprocessorConfig)


class BasePreprocessor(ABC, Generic[C]):
    """Interface for segment-level preprocessors.

    Operates on a single SegmentSignal and returns one PreprocessResult.
    """

    name: str = "BasePreprocessor"
    version: str = "0.0"

    def __init__(self, config: C) -> None:
        self.config = config

    @abstractmethod
    def run(self, signal: SegmentSignal) -> PreprocessResult:
        raise NotImplementedError


class BaseRecordingPreprocessor(ABC, Generic[C]):
    """Interface for recording-level preprocessors.

    Needed for methods that jointly process multiple segments within a recording
    (e.g. group RPCA across all artery loci).

    Returns a dict mapping segment_label -> PreprocessResult.
    """

    name: str = "BaseRecordingPreprocessor"
    version: str = "0.0"

    def __init__(self, config: C) -> None:
        self.config = config

    @abstractmethod
    def run(self, recording: Recording) -> Dict[str, PreprocessResult]:
        raise NotImplementedError
