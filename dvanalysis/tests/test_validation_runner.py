"""Tests for dvanalysis.validation.runner — full hybrid evaluation pipeline."""
import numpy as np
import pandas as pd
import pytest

from dvanalysis.domain import TimeWindow, StimulusCycle, StimulusProtocol, SegmentSignal, Segment, Recording
from dvanalysis.validation.runner import run_hybrid_evaluation


def _make_protocol(fs=25.0):
    gb = TimeWindow("baseline", 0.0, 20.0)
    cycles = [StimulusCycle(index=i,
        baseline=TimeWindow("baseline", float(20+i*100), float(50+i*100)),
        flicker=TimeWindow("flicker", float(50+i*100), float(70+i*100)),
        recovery=TimeWindow("recovery", float(70+i*100), float(120+i*100)))
        for i in range(3)]
    return StimulusProtocol(name="test", fs=fs, global_baseline=gb, cycles=cycles)


def _make_signal(T=8000, P=20, fs=25.0, seed=42):
    rng = np.random.default_rng(seed)
    protocol = _make_protocol(fs)
    t = np.arange(T, dtype=float) / fs
    baseline = 130.0 + rng.normal(0, 0.5, (T, P))
    cardiac = 2.0 * np.sin(2 * np.pi * 1.2 * t)[:, None] * np.ones((1, P))
    x = baseline + cardiac
    m = np.ones((T, P), dtype=bool)
    m[100:105, 3] = False
    return SegmentSignal(t=t, x=x, m=m, protocol=protocol)


def _make_segment(label="A1", vessel_type="artery", seed=42):
    sig = _make_signal(seed=seed)
    return Segment(segment_label=label, vessel_type=vessel_type, signal=sig)


class TestRunHybridEvaluation:
    def test_returns_dataframe(self):
        seg = _make_segment()
        df = run_hybrid_evaluation(
            segments={"001_0_A1": seg},
            n_configs=2,
            seed=42,
            rpca_max_iter=10,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_contains_expected_columns(self):
        seg = _make_segment()
        df = run_hybrid_evaluation(
            segments={"001_0_A1": seg},
            n_configs=2,
            seed=42,
            rpca_max_iter=10,
        )
        expected_cols = ["segment_key", "config_index", "method", "SDR", "NMSE"]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_multiple_methods(self):
        seg = _make_segment()
        df = run_hybrid_evaluation(
            segments={"001_0_A1": seg},
            n_configs=2,
            seed=42,
            rpca_max_iter=10,
        )
        methods = df["method"].unique()
        assert "rpca" in methods
        assert "sms" in methods

    def test_all_methods_produce_finite_sdr(self):
        """All methods should produce finite SDR values."""
        seg = _make_segment()
        df = run_hybrid_evaluation(
            segments={"001_0_A1": seg},
            n_configs=3,
            seed=42,
            rpca_max_iter=20,
        )
        for method in df["method"].unique():
            sdr = df[df["method"] == method]["SDR"].median()
            assert np.isfinite(sdr), f"Non-finite SDR for {method}: {sdr}"

    def test_vein_segment(self):
        seg = _make_segment(label="V3", vessel_type="vein", seed=99)
        df = run_hybrid_evaluation(
            segments={"001_0_V3": seg},
            n_configs=2,
            seed=42,
            rpca_max_iter=10,
        )
        assert len(df) > 0
        assert all(df["vessel_type"] == "vein")
