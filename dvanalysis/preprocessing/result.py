"""PreprocessResult — output container for all preprocessing methods."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from dvanalysis.domain import SegmentSignal


@dataclass
class PreprocessResult:
    """Output of a preprocessing step.

    Always present:
        input_signal : the SegmentSignal that went in
        s_hat : (T,) vessel-level response trace
        m_used : (T, P) mask used inside the method

    Optional:
        S_hat : (T, P) locus-level physiology estimate
        A_hat : (T, P) artefact estimate
        H_hat : (T, P) heartbeat / high-frequency component
    """

    preprocessor_name: str
    preprocessor_version: str
    config: Dict[str, Any]

    input_signal: SegmentSignal

    s_hat: np.ndarray
    m_used: np.ndarray

    S_hat: Optional[np.ndarray] = None
    A_hat: Optional[np.ndarray] = None
    H_hat: Optional[np.ndarray] = None

    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        T, P = self.input_signal.T, self.input_signal.P

        self.s_hat = np.asarray(self.s_hat, dtype=float)
        if self.s_hat.ndim != 1 or self.s_hat.shape[0] != T:
            raise ValueError(f"s_hat must be (T,), got {self.s_hat.shape}, expected T={T}")

        if self.m_used.shape != (T, P):
            raise ValueError(f"m_used must be (T,P)={(T, P)}, got {self.m_used.shape}")

        if self.m_used.dtype != bool:
            self.m_used = self.m_used.astype(bool)

        for name, arr in [("S_hat", self.S_hat), ("A_hat", self.A_hat), ("H_hat", self.H_hat)]:
            if arr is None:
                continue
            if arr.shape != (T, P):
                raise ValueError(f"{name} must be (T,P)={(T, P)}, got {arr.shape}")

    @property
    def T(self) -> int:
        return self.input_signal.T

    @property
    def P(self) -> int:
        return self.input_signal.P

    def as_segment_signal(self) -> SegmentSignal:
        """Wrap the estimated physiology as a SegmentSignal for plotting."""
        if self.S_hat is not None:
            x = self.S_hat
        else:
            x = np.repeat(self.s_hat[:, None], self.P, axis=1)

        return SegmentSignal(
            t=self.input_signal.t.copy(),
            x=x,
            m=self.m_used.copy(),
            protocol=self.input_signal.protocol,
            units=getattr(self.input_signal, "units", "a.u."),
            meta={
                "source": "PreprocessResult.as_segment_signal",
                "preprocessor": self.preprocessor_name,
                "version": self.preprocessor_version,
                **getattr(self.input_signal, "meta", {}),
            },
        )
