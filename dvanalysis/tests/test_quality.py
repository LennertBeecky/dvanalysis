"""Tests for dvanalysis.quality."""
import numpy as np
import pytest

from dvanalysis.domain import TimeWindow, StimulusCycle, StimulusProtocol, SegmentSignal, Segment, Recording
from dvanalysis.quality import (
    QualityReport, QualityAnalyzer, QualityAnalyzerConfig, QualityThresholds,
    QualityRun, QualityItem, quality_run_to_dataframe,
)


def _make_protocol(fs=25.0):
    gb = TimeWindow("baseline", 0.0, 20.0)
    cycles = [StimulusCycle(index=0,
        baseline=TimeWindow("baseline", 20.0, 50.0),
        flicker=TimeWindow("flicker", 50.0, 70.0),
        recovery=TimeWindow("recovery", 70.0, 120.0))]
    return StimulusProtocol(name="test", fs=fs, global_baseline=gb, cycles=cycles)


def _make_signal(T=3000, P=15, fs=25.0, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(T, dtype=float) / fs
    x = rng.normal(100.0, 1.0, (T, P))
    m = np.ones((T, P), dtype=bool)
    return SegmentSignal(t=t, x=x, m=m, protocol=_make_protocol(fs))


class TestQualityReport:
    def test_passed_no_flags(self):
        r = QualityReport(analyzer_name="test", analyzer_version="0", target_type="raw")
        assert r.passed()

    def test_passed_all_false_flags(self):
        r = QualityReport(analyzer_name="test", analyzer_version="0", target_type="raw",
                          flags={"a": False, "b": False})
        assert r.passed()

    def test_failed_when_flag_true(self):
        r = QualityReport(analyzer_name="test", analyzer_version="0", target_type="raw",
                          flags={"bad": True})
        assert not r.passed()


class TestQualityAnalyzer:
    def test_analyze_clean_signal(self):
        sig = _make_signal()
        qa = QualityAnalyzer(QualityAnalyzerConfig())
        report = qa.analyze_segment_signal(sig)
        assert isinstance(report, QualityReport)
        assert report.passed()
        assert report.metrics["missing_fraction_total"] == 0.0

    def test_detect_large_gap(self):
        sig = _make_signal(T=3000, P=15)
        # Create a signal with a large gap (>= 25 samples where ALL loci are missing)
        x = sig.x.copy()
        m = sig.m.copy()
        m[200:230, :] = False  # 30-sample gap across all loci
        x[200:230, :] = np.nan
        sig2 = SegmentSignal(t=sig.t, x=x, m=m, protocol=sig.protocol)

        qa = QualityAnalyzer(QualityAnalyzerConfig())
        report = qa.analyze_segment_signal(sig2)
        assert report.flags["large_gaps_present"]

    def test_too_few_loci(self):
        rng = np.random.default_rng(0)
        t = np.arange(100, dtype=float) / 25.0
        x = rng.normal(100, 1, (100, 3))  # only 3 loci
        m = np.ones((100, 3), dtype=bool)
        sig = SegmentSignal(t=t, x=x, m=m, protocol=_make_protocol())
        qa = QualityAnalyzer(QualityAnalyzerConfig(thresholds=QualityThresholds(min_locus_points=10)))
        report = qa.analyze_segment_signal(sig)
        assert report.flags.get("too_few_locus_points")


class TestQualityRunExport:
    def test_to_dataframe(self):
        report = QualityReport(analyzer_name="test", analyzer_version="0", target_type="raw",
                               metrics={"missing_fraction_total": 0.1}, flags={"bad": False})
        item = QualityItem(subject_id="001", visit_id="0", segment_label="A1",
                           vessel_type="artery", stage="raw", target_id=None, report=report)
        run = QualityRun(items=[item], meta={"scope": "test"})
        df = quality_run_to_dataframe(run)
        assert len(df) == 1
        assert "missing_fraction_total" in df.columns
