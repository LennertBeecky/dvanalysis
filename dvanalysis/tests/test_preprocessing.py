"""Tests for dvanalysis.preprocessing — signal conditioning modules."""
import numpy as np
import pytest

from dvanalysis.domain import (
    TimeWindow, StimulusCycle, StimulusProtocol, SegmentSignal, Segment, Recording,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_protocol(fs=25.0):
    """Standard 3-cycle DVA protocol."""
    gb = TimeWindow("baseline", 0.0, 20.0)
    cycles = []
    t = 20.0
    for i in range(3):
        b = TimeWindow("baseline", t, t + 30.0)
        f = TimeWindow("flicker", t + 30.0, t + 50.0)
        r = TimeWindow("recovery", t + 50.0, t + 100.0)
        cycles.append(StimulusCycle(index=i, baseline=b, flicker=f, recovery=r))
        t += 100.0
    return StimulusProtocol(name="test_3cycle", fs=fs, global_baseline=gb, cycles=cycles)


def _make_signal(T=8000, P=20, fs=25.0, baseline_val=100.0, seed=42):
    """Synthetic segment signal with known structure."""
    protocol = _make_protocol(fs=fs)
    rng = np.random.default_rng(seed)
    t = np.arange(T, dtype=float) / fs
    x = np.full((T, P), baseline_val, dtype=float) + rng.normal(0, 1.0, (T, P))

    # Add a dilation response during flicker windows
    for cyc in protocol.cycles:
        idx = np.where(cyc.flicker.contains(t))[0]
        x[idx, :] += 3.0  # 3% dilation-like signal

    m = np.ones((T, P), dtype=bool)
    # Introduce some missing data (guard against small T or P)
    if T > 110 and P > 3:
        m[100:110, 3] = False
    if T > 520 and P > 7:
        m[500:520, 7] = False
    x[~m] = np.nan

    return SegmentSignal(t=t, x=x, m=m, protocol=protocol)


def _make_recording(fs=25.0, seed=42):
    """Synthetic recording with 2 artery and 2 vein segments."""
    rec = Recording(subject_id="001", visit_id="0")
    for i, (label, vtype) in enumerate([("A1", "artery"), ("A2", "artery"), ("V3", "vein"), ("V4", "vein")]):
        sig = _make_signal(fs=fs, seed=seed + i)
        rec.add_segment(Segment(segment_label=label, vessel_type=vtype, signal=sig))
    return rec


# ---------------------------------------------------------------------------
# PreprocessorConfig and PreprocessResult
# ---------------------------------------------------------------------------

class TestConfigs:
    def test_base_config(self):
        from dvanalysis.preprocessing.configs import PreprocessorConfig
        cfg = PreprocessorConfig()
        d = cfg.to_dict()
        assert "debug" in d
        assert d["debug"] is False

    def test_result_shape_validation(self):
        from dvanalysis.preprocessing.result import PreprocessResult
        sig = _make_signal(T=100, P=5)
        # Valid result
        res = PreprocessResult(
            preprocessor_name="test",
            preprocessor_version="0.0",
            config={},
            input_signal=sig,
            s_hat=np.zeros(100),
            m_used=np.ones((100, 5), dtype=bool),
        )
        assert res.T == 100
        assert res.P == 5

    def test_result_bad_s_hat_shape(self):
        from dvanalysis.preprocessing.result import PreprocessResult
        sig = _make_signal(T=100, P=5)
        with pytest.raises(ValueError, match="s_hat"):
            PreprocessResult(
                preprocessor_name="test",
                preprocessor_version="0.0",
                config={},
                input_signal=sig,
                s_hat=np.zeros(50),  # wrong length
                m_used=np.ones((100, 5), dtype=bool),
            )


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

class TestFiltering:
    def test_lowpass_filter(self):
        from dvanalysis.preprocessing.filtering import lowpass_filter
        # 25 Hz signal, low pass at 5 Hz should remove high-freq content
        t = np.arange(1000) / 25.0
        y = np.sin(2 * np.pi * 1.0 * t) + 0.5 * np.sin(2 * np.pi * 10.0 * t)
        y_lp = lowpass_filter(y, cutoff=5.0, sampling_rate=25.0)
        # High-freq component should be attenuated
        assert np.std(y_lp) < np.std(y)

    def test_lowpass_filter_nan_raises(self):
        from dvanalysis.preprocessing.filtering import lowpass_filter
        y = np.array([1.0, np.nan, 2.0])
        with pytest.raises(ValueError):
            lowpass_filter(y, cutoff=5.0, sampling_rate=25.0)

    def test_lowpass_filter_masked(self):
        from dvanalysis.preprocessing.filtering import lowpass_filter_masked
        rng = np.random.default_rng(0)
        y = rng.normal(0, 1, 200)
        m = np.ones(200, dtype=bool)
        m[50:60] = False
        y[50:60] = np.nan
        lp, hi = lowpass_filter_masked(y, m, cutoff=5.0, sampling_rate=25.0)
        # Where mask is False, output should be NaN
        assert np.all(np.isnan(lp[50:60]))
        # Where mask is True, output should be finite
        assert np.all(np.isfinite(lp[m]))

    def test_extract_heartbeat_psd(self):
        from dvanalysis.preprocessing.filtering import extract_heartbeat_psd
        # Synthesize a signal with a known heartbeat at 1.2 Hz (72 bpm)
        fs = 25.0
        t = np.arange(5000) / fs
        hb_freq = 1.2  # Hz
        signal = np.sin(2 * np.pi * hb_freq * t)
        bpm, amp = extract_heartbeat_psd(signal, timestep_ms=1000.0 / fs)
        assert abs(bpm - 72.0) < 5.0  # within 5 bpm

    def test_bandstop_filter(self):
        from dvanalysis.preprocessing.filtering import bandstop_filter
        fs = 25.0
        t = np.arange(1000) / fs
        y = np.sin(2 * np.pi * 1.0 * t) + np.sin(2 * np.pi * 5.0 * t)
        y_bs = bandstop_filter(y, lowcut=4.5, highcut=5.5, sampling_rate=fs)
        # 5 Hz component should be attenuated
        assert np.std(y_bs) < np.std(y)


# ---------------------------------------------------------------------------
# RPCA solver
# ---------------------------------------------------------------------------

class TestRPCA:
    def test_fit_basic(self):
        from dvanalysis.preprocessing.rpca import RobustPCA
        rng = np.random.default_rng(42)
        # Low-rank + sparse
        P, T = 5, 50
        S_true = np.outer(rng.normal(0, 1, P), np.sin(np.linspace(0, 2 * np.pi, T)))
        A_true = np.zeros((P, T))
        A_true[2, 25] = 10.0  # one sparse spike
        X = S_true + A_true

        rpca = RobustPCA(lmb=0.1, mu=1.0, rho=5.0, gamma=0.0, max_iter=100, tol_rel=1e-5)
        S, A, diag = rpca.fit(X)
        assert S.shape == (P, T)
        assert A.shape == (P, T)
        assert "n_iter" in diag
        # Sparse component should detect the spike
        assert abs(A[2, 25]) > 1.0

    def test_fit_with_mask(self):
        from dvanalysis.preprocessing.rpca import RobustPCA
        rng = np.random.default_rng(0)
        P, T = 4, 40
        X = rng.normal(0, 1, (P, T))
        M = np.ones((P, T), dtype=bool)
        M[1, 10:15] = False
        X[1, 10:15] = np.nan

        rpca = RobustPCA(lmb=0.1, mu=1.0, rho=5.0, gamma=0.0, max_iter=50)
        S, A, diag = rpca.fit(X, M)
        assert S.shape == (P, T)
        assert not np.any(np.isnan(S))  # solver should produce finite output

    def test_fit_with_temporal_smoothness(self):
        from dvanalysis.preprocessing.rpca import RobustPCA
        rng = np.random.default_rng(1)
        P, T = 3, 60
        X = rng.normal(0, 1, (P, T))
        rpca = RobustPCA(lmb=0.1, mu=1.0, rho=5.0, gamma=1.0, max_iter=100)
        S, A, diag = rpca.fit(X)
        # With temporal smoothness, S should be smoother than X
        diffs_S = np.diff(S, axis=1)
        diffs_X = np.diff(X, axis=1)
        assert np.std(diffs_S) < np.std(diffs_X)

    def test_convergence(self):
        from dvanalysis.preprocessing.rpca import RobustPCA
        rng = np.random.default_rng(42)
        P, T = 3, 30
        X = rng.normal(0, 1, (P, T))
        rpca = RobustPCA(lmb=0.5, mu=1.0, rho=5.0, gamma=0.0, max_iter=200, tol_rel=1e-4)
        S, A, diag = rpca.fit(X)
        # Should converge before max_iter
        assert diag["n_iter"] < 200


# ---------------------------------------------------------------------------
# Centering
# ---------------------------------------------------------------------------

class TestCentering:
    def test_baseline_centering_global(self):
        from dvanalysis.preprocessing.centering import baseline_center_and_scale
        rng = np.random.default_rng(42)
        T, P = 100, 5
        X = rng.normal(50.0, 2.0, (T, P))
        M = np.ones((T, P), dtype=bool)
        baseline_idx = np.arange(20)

        Z, mu, sd = baseline_center_and_scale(X, M, baseline_idx, scale_mode="global")
        assert Z.shape == (T, P)
        assert mu.shape == (1, P)
        assert sd.shape == (1, P)
        # Centered: baseline should have ~0 mean
        baseline_Z = Z[baseline_idx, :]
        assert abs(np.nanmean(baseline_Z)) < 0.5

    def test_baseline_centering_per_locus(self):
        from dvanalysis.preprocessing.centering import baseline_center_and_scale
        T, P = 100, 3
        X = np.zeros((T, P))
        X[:, 0] = 10.0
        X[:, 1] = 20.0
        X[:, 2] = 30.0
        M = np.ones((T, P), dtype=bool)
        baseline_idx = np.arange(20)

        Z, mu, sd = baseline_center_and_scale(X, M, baseline_idx, scale_mode="per_locus")
        # Each locus should be centered to ~0 in baseline
        for p in range(P):
            assert abs(np.mean(Z[baseline_idx, p])) < 1e-10


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

class TestAggregation:
    def test_aggregate_loci_median(self):
        from dvanalysis.preprocessing.aggregation import aggregate_loci
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        m = np.ones_like(x, dtype=bool)
        y = aggregate_loci(x, m, mode="median")
        np.testing.assert_array_equal(y, [2.0, 5.0])

    def test_aggregate_loci_with_mask(self):
        from dvanalysis.preprocessing.aggregation import aggregate_loci
        x = np.array([[1.0, 100.0, 3.0]])
        m = np.array([[True, False, True]])
        y = aggregate_loci(x, m, mode="mean")
        np.testing.assert_almost_equal(y, [2.0])

    def test_moving_average(self):
        from dvanalysis.preprocessing.aggregation import moving_average_cols
        X = np.zeros((10, 2))
        X[5, :] = 10.0  # spike
        out = moving_average_cols(X, window=3)
        # Spike should be smoothed
        assert out[5, 0] < 10.0
        assert out[5, 0] > 0.0


# ---------------------------------------------------------------------------
# Segment-level RPCA pipeline (rpca_denoise.py / MyMethodRPCA)
# ---------------------------------------------------------------------------

class TestRPCADenoise:
    def test_run_produces_result(self):
        from dvanalysis.preprocessing.rpca_denoise import MyMethodRPCA, MyMethodConfig
        sig = _make_signal(T=500, P=10, fs=25.0, seed=42)
        cfg = MyMethodConfig(
            rpca_lmb=0.1,
            rpca_max_iter=20,
            standardize=True,
            harmonize_output=False,
            heartbeat_filter=False,
            fill_missing=False,
        )
        proc = MyMethodRPCA(config=cfg)
        result = proc.run(sig)
        assert result.s_hat.shape == (500,)
        assert result.S_hat.shape == (500, 10)
        assert result.A_hat.shape == (500, 10)
        assert result.m_used.shape == (500, 10)

    def test_run_with_harmonize(self):
        from dvanalysis.preprocessing.rpca_denoise import MyMethodRPCA, MyMethodConfig
        # Need enough timepoints to cover the protocol (320s * 25Hz = 8000)
        sig = _make_signal(T=8000, P=15, fs=25.0, seed=42)
        cfg = MyMethodConfig(
            rpca_lmb=0.1,
            rpca_max_iter=20,
            standardize=True,
            harmonize_output=True,
            harmonize_percent_mode="delta_over_baseline",
            heartbeat_filter=False,
            fill_missing=False,
            support_min_valid_abs=5,
        )
        proc = MyMethodRPCA(config=cfg)
        result = proc.run(sig)
        assert result.s_hat.shape == (8000,)
        assert result.diagnostics.get("harmonize_output") is True


# ---------------------------------------------------------------------------
# Group RPCA (recording-level)
# ---------------------------------------------------------------------------

class TestGroupRPCA:
    def test_run_produces_results_per_segment(self):
        from dvanalysis.preprocessing.group_rpca import RPCARecordingPreprocessor, RPCARecordingConfig
        rec = _make_recording(fs=25.0, seed=42)
        cfg = RPCARecordingConfig(
            rpca_lmb=0.1,
            rpca_max_iter=20,
            smooth_enabled=False,
            rpca_path="",
        )
        proc = RPCARecordingPreprocessor(config=cfg)
        results = proc.run(rec)
        # Should have results for all 4 segments
        assert len(results) == 4
        for label, res in results.items():
            assert res.s_hat.shape[0] == rec.segments[label].signal.T


# ---------------------------------------------------------------------------
# Kotliar preprocessor
# ---------------------------------------------------------------------------

class TestKotliar:
    def test_run_cycle_template(self):
        from dvanalysis.preprocessing.kotliar import KotliarPreprocessor, KotliarConfig
        sig = _make_signal(T=8000, P=10, fs=25.0, seed=42)
        cfg = KotliarConfig(validate_protocol=True, mode="cycle_template")
        proc = KotliarPreprocessor(config=cfg)
        result = proc.run(sig)
        assert result.s_hat.shape == (8000,)
        # Cycle template mode: s_hat should have NaNs outside cycle windows
        assert np.any(np.isnan(result.s_hat))

    def test_run_smooth_only(self):
        from dvanalysis.preprocessing.kotliar import KotliarPreprocessor, KotliarConfig
        sig = _make_signal(T=8000, P=10, fs=25.0, seed=42)
        cfg = KotliarConfig(
            validate_protocol=True,
            mode="smooth_only",
            smooth_only_keep_outside_cycles=True,
        )
        proc = KotliarPreprocessor(config=cfg)
        result = proc.run(sig)
        assert result.s_hat.shape == (8000,)


# ---------------------------------------------------------------------------
# Gherghel preprocessor
# ---------------------------------------------------------------------------

class TestGherghel:
    def test_run_global(self):
        from dvanalysis.preprocessing.gherghel import GherghelPreprocessor, GherghelConfig
        sig = _make_signal(T=8000, P=10, fs=25.0, seed=42)
        cfg = GherghelConfig(fit_mode="global", validate_protocol=True)
        proc = GherghelPreprocessor(config=cfg)
        result = proc.run(sig)
        assert result.s_hat.shape == (8000,)
        assert np.all(np.isfinite(result.s_hat))  # global fit covers everything

    def test_run_per_cycle(self):
        from dvanalysis.preprocessing.gherghel import GherghelPreprocessor, GherghelConfig
        sig = _make_signal(T=8000, P=10, fs=25.0, seed=42)
        cfg = GherghelConfig(fit_mode="per_cycle", validate_protocol=True)
        proc = GherghelPreprocessor(config=cfg)
        result = proc.run(sig)
        assert result.s_hat.shape == (8000,)
