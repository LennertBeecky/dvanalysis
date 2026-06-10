"""Tests for dvanalysis.validation.hybrid — hybrid sample assembly."""
import numpy as np
import pytest

from dvanalysis.domain import TimeWindow, StimulusCycle, StimulusProtocol, SegmentSignal
from dvanalysis.validation.hybrid import HybridSample, build_hybrid, HybridBuilder
from dvanalysis.validation.templates import sample_arterial_params, sample_venous_params


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
    # Realistic signal with cardiac + noise
    baseline = 130.0 + rng.normal(0, 0.5, (T, P))
    cardiac = 2.0 * np.sin(2 * np.pi * 1.2 * t)[:, None] * np.ones((1, P))
    x = baseline + cardiac
    m = np.ones((T, P), dtype=bool)
    m[100:105, 3] = False
    m[500:510, :] = False  # full-frame dropout
    return SegmentSignal(t=t, x=x, m=m, protocol=protocol)


class TestHybridSample:
    def test_dataclass_fields(self):
        sig = _make_signal()
        gt = np.zeros(sig.T)
        hs = HybridSample(signal=sig, ground_truth=gt, params={"test": True})
        assert hs.signal.T == sig.T
        assert hs.ground_truth.shape == (sig.T,)
        assert hs.params["test"] is True


class TestBuildHybrid:
    def test_output_type(self):
        sig = _make_signal()
        rng = np.random.default_rng(42)
        params = sample_arterial_params(rng)
        hs = build_hybrid(sig, vessel_type="artery", params=params, rng=rng)
        assert isinstance(hs, HybridSample)

    def test_mask_is_constructed(self):
        """Hybrid mask should combine residual NaN pattern with flicker dropout."""
        sig = _make_signal()
        rng = np.random.default_rng(42)
        params = sample_arterial_params(rng)
        hs = build_hybrid(sig, vessel_type="artery", params=params, rng=rng)
        m = hs.signal.m
        # Mask should be boolean and same shape as signal
        assert m.dtype == bool
        assert m.shape == sig.x.shape
        # During flicker, observation rate should be ~50% (every other frame masked)
        t = np.asarray(sig.t, float)
        for cyc in sig.protocol.cycles:
            fl_idx = np.where((t >= cyc.flicker.start_sec) & (t < cyc.flicker.end_sec))[0]
            if len(fl_idx) > 0:
                frac = m[fl_idx].mean()
                assert frac < 0.6, f"Flicker observation rate {frac:.2f} too high (expected ~0.5)"

    def test_ground_truth_shape(self):
        sig = _make_signal()
        rng = np.random.default_rng(42)
        params = sample_arterial_params(rng)
        hs = build_hybrid(sig, vessel_type="artery", params=params, rng=rng)
        assert hs.ground_truth.shape == (sig.T,)

    def test_ground_truth_has_response(self):
        """Ground truth should have non-zero values during flicker."""
        sig = _make_signal()
        rng = np.random.default_rng(42)
        params = sample_arterial_params(rng)
        hs = build_hybrid(sig, vessel_type="artery", params=params, rng=rng)
        protocol = sig.protocol
        fl_idx = np.where(protocol.cycles[0].flicker.contains(sig.t))[0]
        assert np.nanmax(np.abs(hs.ground_truth[fl_idx])) > 0.5

    def test_signal_differs_from_ground_truth(self):
        """The hybrid signal (with noise) should differ from the clean ground truth."""
        sig = _make_signal()
        rng = np.random.default_rng(42)
        params = sample_arterial_params(rng)
        hs = build_hybrid(sig, vessel_type="artery", params=params, rng=rng)
        # Aggregate hybrid signal to 1D
        x_agg = np.nanmedian(hs.signal.x, axis=1)
        # Should not be identical to ground truth
        finite = np.isfinite(x_agg) & np.isfinite(hs.ground_truth)
        assert not np.allclose(x_agg[finite], hs.ground_truth[finite], atol=0.01)

    def test_venous_template(self):
        sig = _make_signal()
        rng = np.random.default_rng(42)
        params = sample_venous_params(rng)
        hs = build_hybrid(sig, vessel_type="vein", params=params, rng=rng)
        assert isinstance(hs, HybridSample)
        assert hs.ground_truth.shape == (sig.T,)


class TestHybridBuilder:
    def test_generates_samples(self):
        sig = _make_signal()
        builder = HybridBuilder(n_configs=3, seed=42)
        samples = builder.build_for_segment(sig, vessel_type="artery")
        assert len(samples) == 3
        for hs in samples:
            assert isinstance(hs, HybridSample)

    def test_different_configs(self):
        """Each config should produce a different ground truth."""
        sig = _make_signal()
        builder = HybridBuilder(n_configs=5, seed=42)
        samples = builder.build_for_segment(sig, vessel_type="artery")
        gts = [s.ground_truth for s in samples]
        # Not all identical
        assert not all(np.allclose(gts[0], g) for g in gts[1:])

    def test_reproducible(self):
        sig = _make_signal()
        s1 = HybridBuilder(n_configs=3, seed=42).build_for_segment(sig, vessel_type="artery")
        s2 = HybridBuilder(n_configs=3, seed=42).build_for_segment(sig, vessel_type="artery")
        np.testing.assert_array_equal(s1[0].ground_truth, s2[0].ground_truth)
