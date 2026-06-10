"""Tests for dvanalysis.validation.metrics — SDR, NMSE, biomarker bias."""
import numpy as np
import pytest

from dvanalysis.domain import TimeWindow, StimulusCycle, StimulusProtocol
from dvanalysis.validation.metrics import (
    compute_sdr,
    compute_nmse,
    compute_delta_sdr,
    compute_biomarker_bias,
    compute_plateau_stability,
)


def _make_protocol(fs=25.0):
    gb = TimeWindow("baseline", 0.0, 20.0)
    cycles = [StimulusCycle(index=i,
        baseline=TimeWindow("baseline", float(20+i*100), float(50+i*100)),
        flicker=TimeWindow("flicker", float(50+i*100), float(70+i*100)),
        recovery=TimeWindow("recovery", float(70+i*100), float(120+i*100)))
        for i in range(3)]
    return StimulusProtocol(name="test", fs=fs, global_baseline=gb, cycles=cycles)


# ---------------------------------------------------------------------------
# Signal-level metrics
# ---------------------------------------------------------------------------

class TestSDR:
    def test_perfect_recovery(self):
        gt = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert compute_sdr(gt, gt) > 50.0  # should be very high

    def test_noisy_recovery(self):
        gt = np.ones(100)
        recovered = gt + np.random.default_rng(0).normal(0, 0.1, 100)
        sdr = compute_sdr(gt, recovered)
        assert 10.0 < sdr < 30.0

    def test_terrible_recovery(self):
        gt = np.ones(100)
        recovered = np.random.default_rng(0).normal(0, 1, 100)
        sdr = compute_sdr(gt, recovered)
        assert sdr < 5.0

    def test_handles_nan(self):
        gt = np.array([1.0, np.nan, 3.0])
        recovered = np.array([1.0, np.nan, 3.1])
        sdr = compute_sdr(gt, recovered)
        assert np.isfinite(sdr)


class TestNMSE:
    def test_perfect(self):
        gt = np.array([1.0, 2.0, 3.0])
        assert compute_nmse(gt, gt) == 0.0

    def test_proportional_to_error(self):
        gt = np.ones(100)
        r1 = gt + 0.1
        r2 = gt + 0.5
        assert compute_nmse(gt, r1) < compute_nmse(gt, r2)

    def test_handles_nan(self):
        gt = np.array([1.0, np.nan, 3.0])
        recovered = np.array([1.1, np.nan, 3.1])
        nmse = compute_nmse(gt, recovered)
        assert np.isfinite(nmse)


class TestDeltaSDR:
    def test_improvement(self):
        gt = np.ones(100) * 5.0
        rng = np.random.default_rng(0)
        corrupted = gt + rng.normal(0, 1, 100)
        recovered = gt + rng.normal(0, 0.1, 100)
        delta = compute_delta_sdr(gt, corrupted, recovered)
        assert delta > 0  # should improve

    def test_no_improvement(self):
        gt = np.ones(100)
        corrupted = gt + 0.5
        delta = compute_delta_sdr(gt, corrupted, corrupted)
        assert abs(delta) < 0.01  # output == input → delta ≈ 0


# ---------------------------------------------------------------------------
# Biomarker bias
# ---------------------------------------------------------------------------

class TestBiomarkerBias:
    def test_zero_bias_on_perfect_recovery(self):
        """Bias should be near zero when recovered trace equals ground truth."""
        protocol = _make_protocol()
        T = int(protocol.end_time_sec() * protocol.fs)
        t = np.arange(T, dtype=float) / protocol.fs
        # Simple known signal: baseline 0, peak at 3% during flicker
        y = np.zeros(T)
        for cyc in protocol.cycles:
            fl_idx = np.where(cyc.flicker.contains(t))[0]
            y[fl_idx] = 3.0
            rec_idx = np.where(cyc.recovery.contains(t))[0]
            y[rec_idx[:len(rec_idx)//3]] = -1.0  # constriction

        biases = compute_biomarker_bias(y, y, t, protocol, vessel_type="artery")
        for cyc_biases in biases:
            for key, val in cyc_biases.items():
                if np.isfinite(val):
                    assert abs(val) < 0.01, f"Bias for {key}: {val:.4f}"

    def test_detects_amplitude_bias(self):
        """Should detect when recovered trace has different amplitude."""
        protocol = _make_protocol()
        T = int(protocol.end_time_sec() * protocol.fs)
        t = np.arange(T, dtype=float) / protocol.fs
        gt = np.zeros(T)
        recovered = np.zeros(T)
        for cyc in protocol.cycles:
            fl_idx = np.where(cyc.flicker.contains(t))[0]
            gt[fl_idx] = 3.0
            recovered[fl_idx] = 4.0  # overestimates by 1%

        biases = compute_biomarker_bias(gt, recovered, t, protocol, vessel_type="artery")
        # MD bias should be ~1.0
        assert abs(biases[0]["MD"] - 1.0) < 0.1

    def test_vein_no_constriction_biomarkers(self):
        """Venous evaluation should not include MC or DA."""
        protocol = _make_protocol()
        T = int(protocol.end_time_sec() * protocol.fs)
        t = np.arange(T, dtype=float) / protocol.fs
        y = np.zeros(T)
        for cyc in protocol.cycles:
            fl_idx = np.where(cyc.flicker.contains(t))[0]
            y[fl_idx] = 3.0

        biases = compute_biomarker_bias(y, y, t, protocol, vessel_type="vein")
        assert "MC" not in biases[0]
        assert "DA" not in biases[0]
        assert "MD" in biases[0]
        assert "tMAD30" in biases[0]


# ---------------------------------------------------------------------------
# Plateau stability
# ---------------------------------------------------------------------------

class TestPlateauStability:
    def test_stable_on_sharp_peak(self):
        """Sharp peak should have low instability fraction."""
        protocol = _make_protocol()
        T = int(protocol.end_time_sec() * protocol.fs)
        t = np.arange(T, dtype=float) / protocol.fs
        y = np.zeros(T)
        for cyc in protocol.cycles:
            # Sharp peak at flicker start + 10s
            peak_t = cyc.flicker.start_sec + 10.0
            y += 3.0 * np.exp(-((t - peak_t) ** 2) / (2 * 1.0 ** 2))

        frac = compute_plateau_stability(y, y, t, protocol)
        assert frac == 0.0  # perfect recovery → zero instability

    def test_detects_shift(self):
        """Should flag when peak timing shifts by > 0.5s."""
        protocol = _make_protocol()
        T = int(protocol.end_time_sec() * protocol.fs)
        t = np.arange(T, dtype=float) / protocol.fs
        gt = np.zeros(T)
        recovered = np.zeros(T)
        for cyc in protocol.cycles:
            peak_t_gt = cyc.flicker.start_sec + 10.0
            peak_t_rec = cyc.flicker.start_sec + 12.0  # shifted by 2s
            gt += 3.0 * np.exp(-((t - peak_t_gt) ** 2) / (2 * 1.0 ** 2))
            recovered += 3.0 * np.exp(-((t - peak_t_rec) ** 2) / (2 * 1.0 ** 2))

        frac = compute_plateau_stability(gt, recovered, t, protocol)
        assert frac > 0.5  # most cycles should be flagged
