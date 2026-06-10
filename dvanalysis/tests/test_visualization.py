"""Tests for dvanalysis.visualization."""
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI
import matplotlib.pyplot as plt

from dvanalysis.visualization.cycles import extract_cycle_trace, plot_cycle_quality_overlay
from dvanalysis.visualization.bland_altman import compute_bland_altman_data, plot_bland_altman, plot_bland_altman_grid
from dvanalysis.visualization.comparison import plot_method_scatter, plot_method_comparison_grid


# ---------------------------------------------------------------------------
# cycles.py
# ---------------------------------------------------------------------------

class TestExtractCycleTrace:
    def test_returns_resampled_trace(self):
        fs = 25.0
        T = 5000
        t = np.arange(T, dtype=float) / fs
        s_hat = np.sin(2 * np.pi * 0.05 * t)  # slow oscillation

        trace = extract_cycle_trace(s_hat, t, flicker_start_sec=50.0, pre_sec=10, post_sec=30, fs=fs)
        assert trace is not None
        assert trace.ndim == 1
        expected_len = int((10 + 30) * fs)
        assert abs(len(trace) - expected_len) <= 1

    def test_returns_none_when_no_baseline(self):
        # Signal starts after the requested pre-window
        t = np.arange(100, dtype=float) / 25.0 + 100.0  # starts at 100s
        s_hat = np.ones(100)
        trace = extract_cycle_trace(s_hat, t, flicker_start_sec=50.0, pre_sec=15, post_sec=50)
        assert trace is None

    def test_handles_nan_gaps(self):
        fs = 25.0
        T = 3000
        t = np.arange(T, dtype=float) / fs
        s_hat = np.ones(T) * 100.0
        s_hat[1250:1260] = np.nan  # small gap near flicker=50s

        trace = extract_cycle_trace(s_hat, t, flicker_start_sec=50.0, pre_sec=10, post_sec=30, fs=fs)
        assert trace is not None
        # Gap should be interpolated
        assert np.sum(np.isfinite(trace)) > 0.9 * len(trace)


class TestPlotCycleQualityOverlay:
    def test_creates_figure(self):
        t_common = np.arange(-10, 40, 1.0 / 25.0)
        rng = np.random.default_rng(42)
        traces = {
            "best": [rng.normal(2, 0.5, len(t_common)) for _ in range(10)],
            "middle": [rng.normal(1, 0.5, len(t_common)) for _ in range(10)],
            "worst": [rng.normal(0, 0.5, len(t_common)) for _ in range(10)],
        }
        fig = plot_cycle_quality_overlay(traces, t_common, vessel_type="artery")
        assert fig is not None
        plt.close(fig)

    def test_empty_rank(self):
        t_common = np.arange(-5, 20, 0.04)
        traces = {"best": [np.ones(len(t_common))]}
        fig = plot_cycle_quality_overlay(traces, t_common)
        assert fig is not None
        plt.close(fig)


# ---------------------------------------------------------------------------
# bland_altman.py
# ---------------------------------------------------------------------------

class TestComputeBlandAltmanData:
    def test_basic(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "subject_id": np.repeat(np.arange(20), 3),
            "biomarker": rng.normal(5, 1, 60),
        })
        data = compute_bland_altman_data(df, "biomarker")
        assert "means" in data
        assert "deviations" in data
        assert np.isfinite(data["loa_lower"])
        assert np.isfinite(data["loa_upper"])
        assert data["loa_lower"] < data["loa_upper"]


class TestPlotBlandAltman:
    def test_single_panel(self):
        rng = np.random.default_rng(1)
        means = rng.normal(5, 1, 50)
        devs = rng.normal(0, 0.5, 50)
        ax = plot_bland_altman(means, devs, -1.0, 1.0)
        assert ax is not None
        plt.close(ax.get_figure())

    def test_grid(self):
        rng = np.random.default_rng(2)
        ba_data = {}
        for vessel in ("artery", "vein"):
            for bio in ("md", "da"):
                ba_data[(vessel, bio)] = {
                    "means": rng.normal(5, 1, 30),
                    "deviations": rng.normal(0, 0.3, 30),
                    "loa_lower": -0.6, "loa_upper": 0.6,
                    "mean_dev": 0.0, "sd_dev": 0.3,
                }
        fig = plot_bland_altman_grid(ba_data, ["md", "da"], ["Max dil.", "Dil. amp."])
        assert fig is not None
        plt.close(fig)


# ---------------------------------------------------------------------------
# comparison.py
# ---------------------------------------------------------------------------

class TestPlotMethodScatter:
    def test_single_scatter(self):
        rng = np.random.default_rng(3)
        x = rng.normal(5, 1, 30)
        y = x + rng.normal(0, 0.3, 30)
        ax = plot_method_scatter(x, y, xlabel="RPCA", ylabel="SMS")
        assert ax is not None
        plt.close(ax.get_figure())

    def test_grid(self):
        rng = np.random.default_rng(4)
        pairs = {}
        for panel in ("best", "cycle0", "cycle1", "cycle2"):
            pairs[panel] = {}
            for vessel in ("artery", "vein"):
                x = rng.normal(3, 1, 20)
                y = x + rng.normal(0, 0.5, 20)
                pairs[panel][vessel] = (x, y)
        fig = plot_method_comparison_grid(pairs, biomarker_label="max dilation (%)")
        assert fig is not None
        plt.close(fig)
