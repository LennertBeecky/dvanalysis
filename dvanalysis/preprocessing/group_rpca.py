# preprocessing/group_rpca.py
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from dvanalysis.domain import Recording, Segment, SegmentSignal
from .base import BaseRecordingPreprocessor
from .configs import PreprocessorConfig
from .result import PreprocessResult

from .rpca import RobustPCA
from .filtering import extract_heartbeat_psd, lowpass_filter
from .aggregation import segment_trace_from_loci, moving_average_cols


@dataclass(frozen=True)
class RPCARecordingConfig(PreprocessorConfig):
    """
    Recording level RPCA preprocessing that mirrors your legacy pipeline:
    fill missing -> heartbeat estimate -> lowpass -> RPCA (artery group and vein group) -> optional smoothing.

    Important
    This method jointly factorizes all artery loci in the recording together,
    and all vein loci together, so it must run at Recording level.
    """
    # General
    strict: bool = False
    debug: bool = False

    # Timing
    timestep_ms: float = 40.0

    # Missing value handling
    fill_missing: bool = True

    # Standardization
    standardize: bool = True

    # Lowpass
    extra_cutoff_hz: float = 1.1 / 2.0
    lp_order: int = 4

    # RPCA hyperparameters (forwarded to RobustPCA)
    rpca_lmb: float = 1e-2
    rpca_mu: float = 1e-5
    rpca_rho: float = 2.0
    rpca_gamma: float = 10.0
    rpca_max_iter: int = 70
    rpca_tol_rel: float = 1e-3
    rpca_path: str = ""

    # Optional smoothing (moving average on the recovered low rank component)
    smooth_enabled: bool = True
    smooth_window: int = 25


class RPCARecordingPreprocessor(BaseRecordingPreprocessor[RPCARecordingConfig]):
    name: str = "RPCARecordingPreprocessor"
    version: str = "1.0"

    def run(self, recording: Recording) -> Dict[str, PreprocessResult]:
        segments = list(recording.segments.values())
        arteries = [s for s in segments if self._is_artery_segment(s)]
        veins = [s for s in segments if self._is_vein_segment(s)]

        results: Dict[str, PreprocessResult] = {}

        if len(arteries) > 0:
            results.update(self._run_group(recording=recording, group_segments=arteries, artery=True))

        if len(veins) > 0:
            results.update(self._run_group(recording=recording, group_segments=veins, artery=False))

        return results

    def _run_group(self, recording: Recording, group_segments: List[Segment], artery: bool) -> Dict[str, PreprocessResult]:
        cfg = self.config

        signals = [seg.signal for seg in group_segments]
        X, M, sizes = self._concat_segments(signals)  # (T, sumP), (T, sumP)

        # Fill missing (interpolation per locus)
        X_filled = self._fill_missing(X, M) if cfg.fill_missing else X.astype(float).copy()

        # Heartbeat estimate on filled, segment averaged traces
        hb = self._estimate_heartbeat(signals)

        # Lowpass cutoff derived from heartbeat, fallback if unavailable
        if not np.isfinite(hb["heartbeat_bpm"]):
            msg = "Heartbeat bpm not finite. Using no lowpass (lp = filled, high = zeros)."
            if cfg.strict:
                raise ValueError(msg)
            if cfg.debug:
                print(f"[{self.name}] {msg}")
            X_lp = X_filled
            X_hi = np.zeros_like(X_filled)
            hb["cutoff_hz"] = float("nan")
        else:
            X_lp, X_hi, cutoff_hz = self._apply_lowpass(X_filled, hb["heartbeat_bpm"])
            hb["cutoff_hz"] = float(cutoff_hz)

        # RPCA on lowpassed data
        L_lp, S_lp = self._run_rpca(X_lp, artery=artery)

        # Optional moving average smoothing on recovered low rank component
        if cfg.smooth_enabled and cfg.smooth_window > 1:
            L_lp = self._moving_average_cols(L_lp, cfg.smooth_window)

        # Split group matrices back per segment
        L_parts = self._split(L_lp, sizes)
        S_parts = self._split(S_lp, sizes)
        HI_parts = self._split(X_hi, sizes)
        M_parts = self._split(M.astype(bool), sizes)

        out: Dict[str, PreprocessResult] = {}

        for seg, L_i, S_i, HI_i, M_i in zip(group_segments, L_parts, S_parts, HI_parts, M_parts):
            # Segment-level trace expected by downstream ParameterExtractor when trace_source is s_hat
            s_hat = self._segment_trace_from_loci(L_i, M_i)

            out[seg.segment_label] = PreprocessResult(
                preprocessor_name=self.name,
                preprocessor_version=self.version,
                config=asdict(cfg),
                input_signal=seg.signal,
                s_hat=s_hat,
                m_used=M_i.astype(bool),
                S_hat=L_i,
                A_hat=S_i,
                H_hat=HI_i,
                diagnostics={
                    "group": "artery" if artery else "vein",
                    "n_segments_in_group": len(group_segments),
                    "segment_label": seg.segment_label,
                    **hb,
                },
            )

        return out

    # -------------------------
    # Segment selection helpers
    # -------------------------
    @staticmethod
    def _is_artery_segment(seg: Segment) -> bool:
        vt = (seg.vessel_type or "").lower()
        if vt.startswith("a"):
            return True
        return (seg.segment_label or "").upper().startswith("A")

    @staticmethod
    def _is_vein_segment(seg: Segment) -> bool:
        vt = (seg.vessel_type or "").lower()
        if vt.startswith("v"):
            return True
        return (seg.segment_label or "").upper().startswith("V")

    # -------------------------
    # Matrix plumbing helpers
    # -------------------------
    @staticmethod
    def _concat_segments(signals: List[SegmentSignal]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        if len(signals) == 0:
            raise ValueError("No signals provided")

        T = signals[0].T
        t0 = signals[0].t

        sizes: List[int] = []
        xs: List[np.ndarray] = []
        ms: List[np.ndarray] = []

        for sig in signals:
            if sig.T != T:
                raise ValueError("All segments must have the same T for group RPCA")
            if not np.allclose(sig.t, t0):
                raise ValueError("All segments must share the same time grid for group RPCA")

            sizes.append(sig.P)
            xs.append(sig.x.astype(float))
            ms.append(sig.m.astype(bool))

        X = np.concatenate(xs, axis=1)  # (T, sumP)
        M = np.concatenate(ms, axis=1)  # (T, sumP)
        return X, M, sizes

    @staticmethod
    def _split(X: np.ndarray, sizes: List[int]) -> List[np.ndarray]:
        parts: List[np.ndarray] = []
        c = 0
        for p in sizes:
            parts.append(X[:, c : c + p])
            c += p
        return parts

    # -------------------------
    # Missing value fill
    # -------------------------
    @staticmethod
    def _fill_missing(X: np.ndarray, M: np.ndarray) -> np.ndarray:
        """
        Linear interpolation per locus.
        Requires at least 2 valid points per locus, otherwise leaves as is.
        """
        T, P = X.shape
        out = X.copy().astype(float)
        idx = np.arange(T)

        for p in range(P):
            valid = M[:, p].astype(bool)
            if valid.all():
                continue
            if valid.sum() < 2:
                continue

            y = out[:, p]
            miss = ~valid
            y[miss] = np.interp(idx[miss], idx[valid], y[valid])
            out[:, p] = y

        return out

    # -------------------------
    # Heartbeat estimate and lowpass
    # -------------------------
    def _estimate_heartbeat(self, signals: List[SegmentSignal]) -> Dict[str, float]:
        cfg = self.config
        rates: List[float] = []
        amps: List[float] = []

        for sig in signals:
            x = sig.x.astype(float).copy()
            m = sig.m.astype(bool)
            x[~m] = np.nan
            y = np.nanmean(x, axis=1)

            hb_bpm, hb_amp = extract_heartbeat_psd(y, timestep_ms=cfg.timestep_ms)
            if np.isfinite(hb_bpm):
                rates.append(float(hb_bpm))
            if np.isfinite(hb_amp):
                amps.append(float(hb_amp))

        return {
            "heartbeat_bpm": float(np.mean(rates)) if len(rates) else float("nan"),
            "heartbeat_amp": float(np.mean(amps)) if len(amps) else float("nan"),
        }

    def _apply_lowpass(self, X: np.ndarray, heartbeat_bpm: float) -> Tuple[np.ndarray, np.ndarray, float]:
        cfg = self.config
        fs = 1000.0 / cfg.timestep_ms
        cutoff_hz = heartbeat_bpm / 60.0 + cfg.extra_cutoff_hz

        X_lp = np.empty_like(X, dtype=float)
        X_hi = np.empty_like(X, dtype=float)

        for p in range(X.shape[1]):
            y = X[:, p]
            y_lp = lowpass_filter(
                y,
                cutoff=cutoff_hz,
                sampling_rate=fs,
                order=cfg.lp_order,
            )
            X_lp[:, p] = y_lp
            X_hi[:, p] = y - y_lp

        return X_lp, X_hi, float(cutoff_hz)

    # -------------------------
    # RPCA
    # -------------------------
    def _run_rpca(self, X_lp: np.ndarray, artery: bool) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        X = X_lp.astype(float)

        if cfg.standardize:
            mu = np.nanmean(X, axis=0, keepdims=True)
            sd = np.nanstd(X, axis=0, keepdims=True)
            sd[sd == 0] = 1.0
            Z = (X - mu) / sd
        else:
            mu = np.zeros((1, X.shape[1]), dtype=float)
            sd = np.ones((1, X.shape[1]), dtype=float)
            Z = X

        # Keep orientation consistent with your legacy code: feed (P, T) into RobustPCA.fit
        rpca = RobustPCA(
            lmb=cfg.rpca_lmb,
            mu=cfg.rpca_mu,
            rho=cfg.rpca_rho,
            gamma=cfg.rpca_gamma,
            max_iter=cfg.rpca_max_iter,
            tol_rel=cfg.rpca_tol_rel,
            path=cfg.rpca_path,
            artery=artery,
        )

        L_pt, S_pt, _diag = rpca.fit(Z.T)  # (P, T)
        L = L_pt.T  # (T, P)
        S = S_pt.T  # (T, P)

        if cfg.standardize:
            L = L * sd + mu
            S = S * sd + mu

        return L, S

    # -------------------------
    # Post processing
    # -------------------------
    @staticmethod
    def _segment_trace_from_loci(L: np.ndarray, M: np.ndarray) -> np.ndarray:
        tmp = L.astype(float).copy()
        tmp[~M.astype(bool)] = np.nan
        return np.nanmean(tmp, axis=1)

    @staticmethod
    def _moving_average_cols(X: np.ndarray, window: int) -> np.ndarray:
        if window <= 1:
            return X
        half = window // 2
        out = np.empty_like(X, dtype=float)
        T, P = X.shape
        for p in range(P):
            y = X[:, p]
            yy = np.empty(T, dtype=float)
            for i in range(T):
                a = max(0, i - half)
                b = min(T, i + half + 1)
                yy[i] = float(np.mean(y[a:b]))
            out[:, p] = yy
        return out
