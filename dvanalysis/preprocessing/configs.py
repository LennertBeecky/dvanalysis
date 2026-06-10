"""Base configuration for all preprocessors."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict


@dataclass(frozen=True)
class PreprocessorConfig:
    """Base config. Concrete preprocessors should subclass this."""

    debug: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
