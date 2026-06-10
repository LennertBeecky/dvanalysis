"""Quality assessment: missingness, gaps, and baseline stability."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from dvanalysis.domain import SegmentSignal, StimulusProtocol
from .reporting import QualityReport, QualityRun, QualityItem


@dataclass(frozen=True)
class QualityThresholds:
    max_missing_total: float = 0.50
    max_missing_flicker: float = 0.67
    small_gap: int = 5
    medium_gap: int = 10
    large_gap: int = 25
    min_locus_points: int = 10


@dataclass(frozen=True)
class QualityAnalyzerConfig:
    name: str = "QualityAnalyzer"
    version: str = "0.1"
    thresholds: QualityThresholds = QualityThresholds()
    debug: bool = False
    compute_aggregated: bool = True
    aggregated_mode: str = "mean"


class QualityAnalyzer:
    def __init__(self, config: QualityAnalyzerConfig) -> None:
        self.config = config

    def analyze_segment_signal(
        self,
        signal: SegmentSignal,
        target_id: Optional[str] = None,
        stage: str = "raw",
    ) -> QualityReport:
        if signal.P < self.config.thresholds.min_locus_points:
            return QualityReport(
                analyzer_name=self.config.name,
                analyzer_version=self.config.version,
                target_type=stage,
                target_id=target_id,
                metrics={"P": signal.P, "T": signal.T},
                flags={"too_few_locus_points": True},
                notes=f"P={signal.P} < min_locus_points={self.config.thresholds.min_locus_points}",
            )

        protocol = signal.protocol
        fs = float(protocol.fs)

        metrics: Dict[str, Any] = {}
        flags: Dict[str, bool] = {}

        missing_total = 1.0 - float(signal.m.mean())
        metrics["missing_fraction_total"] = missing_total

        window_metrics = self._missingness_by_protocol(signal, protocol)
        metrics.update(window_metrics)

        flicker_missing = self._missingness_over_windows(signal, protocol.flicker_windows())
        metrics["missing_fraction_flicker"] = flicker_missing

        gap_counts, gap_info = self._gap_stats_from_mask(signal.m)
        metrics["gap_counts_total"] = gap_counts
        metrics["gap_info_total"] = gap_info

        if self.config.compute_aggregated:
            agg = self._aggregate_trace(signal.x, signal.m, mode=self.config.aggregated_mode)
            metrics.update(self._baseline_snr_metrics(agg, protocol, fs))

        thr = self.config.thresholds
        flags["missing_total_too_high"] = missing_total > thr.max_missing_total
        flags["missing_flicker_too_high"] = flicker_missing > thr.max_missing_flicker
        flags["large_gaps_present"] = gap_counts["large"] > 0

        return QualityReport(
            analyzer_name=self.config.name,
            analyzer_version=self.config.version,
            target_type=stage,
            target_id=target_id,
            metrics=metrics,
            flags=flags,
        )

    def _missingness_by_protocol(self, signal: SegmentSignal, protocol: StimulusProtocol) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if protocol.global_baseline is not None:
            out["missing_global_baseline"] = self._missingness_over_window(signal, protocol.global_baseline)
        for cyc in protocol.cycles:
            if cyc.baseline is not None:
                out[f"missing_cycle{cyc.index}_baseline"] = self._missingness_over_window(signal, cyc.baseline)
            out[f"missing_cycle{cyc.index}_flicker"] = self._missingness_over_window(signal, cyc.flicker)
            out[f"missing_cycle{cyc.index}_recovery"] = self._missingness_over_window(signal, cyc.recovery)
        return out

    def _missingness_over_windows(self, signal: SegmentSignal, windows: List[Any]) -> float:
        if not windows:
            return float("nan")
        vals = [self._missingness_over_window(signal, w) for w in windows]
        return float(np.nanmean(vals))

    def _missingness_over_window(self, signal: SegmentSignal, window: Any) -> float:
        idx = window.to_indices(signal.t)
        if idx.size == 0:
            return float("nan")
        m_win = signal.m[idx, :]
        return 1.0 - float(m_win.mean())

    def _gap_stats_from_mask(self, m: np.ndarray) -> Tuple[Dict[str, int], Dict[str, List[Tuple[int, int]]]]:
        thr = self.config.thresholds
        valid_t = np.any(m, axis=1)
        invalid = ~valid_t

        gap_info: Dict[str, List[Tuple[int, int]]] = {"small": [], "medium": [], "large": []}
        gap_start: Optional[int] = None

        for i in range(invalid.shape[0]):
            if invalid[i] and gap_start is None:
                gap_start = i
            elif (not invalid[i]) and gap_start is not None:
                gap_end = i - 1
                gap_len = gap_end - gap_start + 1
                self._assign_gap(gap_info, gap_start, gap_end, gap_len, thr)
                gap_start = None

        if gap_start is not None:
            gap_end = invalid.shape[0] - 1
            gap_len = gap_end - gap_start + 1
            self._assign_gap(gap_info, gap_start, gap_end, gap_len, thr)

        gap_counts = {k: len(v) for k, v in gap_info.items()}
        return gap_counts, gap_info

    def _assign_gap(self, gap_info, start, end, length, thr):
        if thr.small_gap <= length < thr.medium_gap:
            gap_info["small"].append((start, end))
        elif thr.medium_gap <= length < thr.large_gap:
            gap_info["medium"].append((start, end))
        elif length >= thr.large_gap:
            gap_info["large"].append((start, end))

    def _aggregate_trace(self, x: np.ndarray, m: np.ndarray, mode: str = "mean") -> np.ndarray:
        x_masked = np.where(m, x, np.nan)
        if mode == "median":
            return np.nanmedian(x_masked, axis=1)
        return np.nanmean(x_masked, axis=1)

    def _baseline_snr_metrics(self, agg: np.ndarray, protocol: StimulusProtocol, fs: float) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if protocol.global_baseline is not None:
            idx = protocol.global_baseline.to_indices_from_fs(fs)
        else:
            idx = np.arange(int(20 * fs), dtype=int)
        idx = idx[idx < agg.shape[0]]
        if idx.size < 10:
            return out
        baseline = agg[idx]
        mu = float(np.nanmean(baseline))
        sd = float(np.nanstd(baseline))
        out["baseline_mean"] = mu
        out["baseline_std"] = sd
        if sd > 0 and np.isfinite(mu) and np.isfinite(sd):
            out["baseline_snr_db"] = float(20.0 * np.log10(abs(mu) / sd))
        else:
            out["baseline_snr_db"] = float("nan")
        return out

    def run_on_recording(self, recording, stage: str = "raw", target_id_field: str = "source_path") -> QualityRun:
        items: List[QualityItem] = []
        for seg_label, seg in recording.segments.items():
            target_id = None
            if isinstance(seg.signal.meta, dict):
                target_id = seg.signal.meta.get(target_id_field)
            report = self.analyze_segment_signal(seg.signal, target_id=target_id, stage=stage)
            items.append(QualityItem(
                subject_id=recording.subject_id, visit_id=recording.visit_id,
                segment_label=seg.segment_label, vessel_type=seg.vessel_type,
                stage=stage, target_id=target_id, report=report,
            ))
        return QualityRun(items=items, meta={
            "scope": "recording", "subject_id": recording.subject_id,
            "visit_id": recording.visit_id, "stage": stage,
            "analyzer": self.config.name, "version": self.config.version,
        })

    def run_on_dataset(self, dataset, stage: str = "raw", target_id_field: str = "source_path") -> QualityRun:
        items: List[QualityItem] = []
        for rec in dataset.recordings:
            rec_run = self.run_on_recording(recording=rec, stage=stage, target_id_field=target_id_field)
            items.extend(rec_run.items)
        return QualityRun(items=items, meta={
            "scope": "dataset", "dataset_name": getattr(dataset, "name", None),
            "root_path": str(getattr(dataset, "root_path", "")),
            "stage": stage, "analyzer": self.config.name, "version": self.config.version,
        })
