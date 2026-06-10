"""Tests for dvanalysis.validation.noise — cardiac extraction and noise tiling."""
import numpy as np
import pytest

from dvanalysis.domain import TimeWindow, StimulusCycle, StimulusProtocol, SegmentSignal
from dvanalysis.validation.noise import (
    extract_cardiac,
    extract_baseline_noise,
    tile_noise,
)


def _make_protocol(fs=25.0):
    gb = TimeWindow("baseline", 0.0, 20.0)
    cycles = [StimulusCycle(index=i,
        baseline=TimeWindow("baseline", float(20+i*100), float(50+i*100)),
        flicker=TimeWindow("flicker", float(50+i*100), float(70+i*100)),
        recovery=TimeWindow("recovery", float(70+i*100), float(120+i*100)))
        for i in range(3)]
    return StimulusProtocol(name="test", fs=fs, global_baseline=gb, cycles=cycles)


def _make_signal(T=8000, P=20, fs=25.0, seed=42):
    """Synthetic signal with known cardiac component for testing."""
    rng = np.random.default_rng(seed)
    protocol = _make_protocol(fs)
    t = np.arange(T, dtype=float) / fs

    # Known components
    baseline = 130.0 + rng.normal(0, 0.5, (T, P))  # resting diameter ~130 MU
    cardiac_freq = 1.2  # Hz (~72 bpm)
    cardiac = 2.0 * np.sin(2 * np.pi * cardiac_freq * t)[:, None] * np.ones((1, P))
    noise = rng.normal(0, 0.3, (T, P))

    x = baseline + cardiac + noise
    m = np.ones((T, P), dtype=bool)
    # Some missing data
    m[100:105, 3] = False

    return SegmentSignal(t=t, x=x, m=m, protocol=protocol)


class TestExtractCardiac:
    def test_output_shape(self):
        sig = _make_signal()
        bl_window = sig.protocol.global_baseline
        cardiac = extract_cardiac(sig, bl_window)
        assert cardiac.shape == sig.x.shape

    def test_detects_heartbeat_frequency(self):
        """Extracted cardiac should oscillate near the known heart rate."""
        sig = _make_signal()
        bl_window = sig.protocol.global_baseline
        cardiac = extract_cardiac(sig, bl_window)
        # Check that the cardiac component has power near 1.2 Hz
        fs = sig.protocol.fs
        # Take mean across loci and compute FFT
        c_mean = np.nanmean(cardiac[:500, :], axis=1)
        freqs = np.fft.rfftfreq(500, d=1.0/fs)
        psd = np.abs(np.fft.rfft(c_mean))**2
        peak_freq = freqs[np.argmax(psd[1:]) + 1]  # skip DC
        assert abs(peak_freq - 1.2) < 0.3, f"Peak freq {peak_freq:.2f} Hz, expected ~1.2 Hz"

    def test_cardiac_removed_from_residual(self):
        """After removing cardiac, the residual should have less power at heartbeat freq."""
        sig = _make_signal()
        bl_window = sig.protocol.global_baseline
        cardiac = extract_cardiac(sig, bl_window)
        residual = sig.x.copy()
        residual[sig.m] -= cardiac[sig.m]
        # Power at heartbeat freq should be lower in residual
        fs = sig.protocol.fs
        bl_idx = np.where(bl_window.contains(sig.t))[0]
        orig_mean = np.nanmean(sig.x[bl_idx, :], axis=1)
        resid_mean = np.nanmean(residual[bl_idx, :], axis=1)
        # Crude check: std of residual should be less than std of original
        assert np.std(resid_mean) < np.std(orig_mean)


class TestExtractBaselineNoise:
    def test_output_shape(self):
        sig = _make_signal()
        bl_window = sig.protocol.global_baseline
        noise = extract_baseline_noise(sig, bl_window)
        bl_idx = np.where(bl_window.contains(sig.t))[0]
        assert noise.shape == (len(bl_idx), sig.P)

    def test_approximately_zero_mean(self):
        """Baseline noise should be approximately zero-mean per locus."""
        sig = _make_signal()
        bl_window = sig.protocol.global_baseline
        noise = extract_baseline_noise(sig, bl_window)
        for p in range(sig.P):
            col = noise[:, p]
            col = col[np.isfinite(col)]
            if col.size > 0:
                assert abs(np.mean(col)) < 1.0, f"Locus {p} mean={np.mean(col):.2f}"

    def test_cardiac_removed(self):
        """Noise should not contain the cardiac oscillation."""
        sig = _make_signal()
        bl_window = sig.protocol.global_baseline
        noise = extract_baseline_noise(sig, bl_window)
        # Check that the noise has much less power at 1.2 Hz than the raw signal
        fs = sig.protocol.fs
        noise_mean = np.nanmean(noise, axis=1)
        if noise_mean.size > 50:
            freqs = np.fft.rfftfreq(noise_mean.size, d=1.0/fs)
            psd = np.abs(np.fft.rfft(noise_mean))**2
            hb_idx = np.argmin(np.abs(freqs - 1.2))
            # Power at heartbeat should be small relative to total
            relative_power = psd[hb_idx] / np.sum(psd[1:])
            assert relative_power < 0.5, f"Cardiac still present: relative power={relative_power:.3f}"


class TestTileNoise:
    def test_output_length(self):
        block = np.random.randn(100, 5)
        tiled = tile_noise(block, target_length=350)
        assert tiled.shape == (350, 5)

    def test_preserves_statistics(self):
        """Tiled noise should have similar std as original block."""
        rng = np.random.default_rng(0)
        block = rng.normal(0, 2.0, (100, 5))
        tiled = tile_noise(block, target_length=500)
        np.testing.assert_allclose(np.std(tiled, axis=0), np.std(block, axis=0), rtol=0.3)

    def test_shorter_than_block(self):
        """If target is shorter than block, truncate."""
        block = np.random.randn(200, 3)
        tiled = tile_noise(block, target_length=50)
        assert tiled.shape == (50, 3)

    def test_exact_multiple(self):
        block = np.random.randn(100, 2)
        tiled = tile_noise(block, target_length=300)
        assert tiled.shape == (300, 2)
