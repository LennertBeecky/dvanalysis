"""Tests for dvanalysis.validation.templates -- synthetic response waveforms."""
import numpy as np
import pytest

from dvanalysis.domain import TimeWindow, StimulusCycle, StimulusProtocol
from dvanalysis.validation.templates import (
    arterial_template,
    venous_template,
    sample_arterial_params,
    sample_venous_params,
    apply_intercycle_jitter,
    ArterialParams,
    VenousParams,
)


def _make_protocol(fs=25.0):
    gb = TimeWindow("baseline", 0.0, 20.0)
    cycles = []
    t = 20.0
    for i in range(3):
        b = TimeWindow("baseline", t, t + 30.0)
        f = TimeWindow("flicker", t + 30.0, t + 50.0)
        r = TimeWindow("recovery", t + 50.0, t + 100.0)
        cycles.append(StimulusCycle(index=i, baseline=b, flicker=f, recovery=r))
        t += 100.0
    return StimulusProtocol(name="test", fs=fs, global_baseline=gb, cycles=cycles)


@pytest.fixture
def t_and_protocol():
    protocol = _make_protocol(fs=25.0)
    T = int(protocol.end_time_sec() * protocol.fs)
    t = np.arange(T, dtype=float) / protocol.fs
    return t, protocol


# ---------------------------------------------------------------------------
# Arterial template
# ---------------------------------------------------------------------------

class TestArterialTemplate:
    def test_output_shape(self, t_and_protocol):
        t, protocol = t_and_protocol
        params = ArterialParams(dilation_amplitude=3.0, constriction_depth=1.0)
        y = arterial_template(t, protocol, params)
        assert y.shape == t.shape

    def test_baseline_near_zero(self, t_and_protocol):
        t, protocol = t_and_protocol
        params = ArterialParams(dilation_amplitude=3.0, constriction_depth=1.0)
        y = arterial_template(t, protocol, params)
        bl_idx = np.where(t < 20.0)[0]
        assert np.abs(np.mean(y[bl_idx])) < 0.1

    def test_peak_within_range(self, t_and_protocol):
        t, protocol = t_and_protocol
        rng = np.random.default_rng(0)
        for _ in range(20):
            params = sample_arterial_params(rng)
            y = arterial_template(t, protocol, params)
            peak = np.max(y)
            assert 0.5 <= peak <= 7.0, f"Peak {peak:.2f}% outside expected range"

    def test_has_constriction(self, t_and_protocol):
        t, protocol = t_and_protocol
        params = ArterialParams(dilation_amplitude=3.0, constriction_depth=1.0)
        y = arterial_template(t, protocol, params)
        rec = protocol.cycles[0].recovery
        rec_idx = np.where(rec.contains(t))[0]
        assert np.min(y[rec_idx]) < -0.1, "No constriction below baseline"

    def test_three_cycles_present(self, t_and_protocol):
        t, protocol = t_and_protocol
        params = ArterialParams(dilation_amplitude=3.0, constriction_depth=1.0)
        y = arterial_template(t, protocol, params)
        for cyc in protocol.cycles:
            fl_idx = np.where(cyc.flicker.contains(t))[0]
            assert np.max(y[fl_idx]) > 0.5, f"Cycle {cyc.index} has no dilation"

    def test_plateau_shape(self, t_and_protocol):
        t, protocol = t_and_protocol
        params = ArterialParams(dilation_amplitude=4.0, constriction_depth=1.5)
        y = arterial_template(t, protocol, params)
        cyc = protocol.cycles[0]
        w = TimeWindow("search", cyc.flicker.start_sec, cyc.flicker.end_sec + 5.0)
        w_idx = np.where(w.contains(t))[0]
        y_w = y[w_idx]
        peak = np.max(y_w)
        above_80 = np.sum(y_w >= 0.8 * peak) / protocol.fs
        assert above_80 >= 3.0, f"Plateau duration {above_80:.1f}s too short"


# ---------------------------------------------------------------------------
# Venous template
# ---------------------------------------------------------------------------

class TestVenousTemplate:
    def test_output_shape(self, t_and_protocol):
        t, protocol = t_and_protocol
        params = VenousParams(dilation_amplitude=4.0)
        y = venous_template(t, protocol, params)
        assert y.shape == t.shape

    def test_baseline_near_zero(self, t_and_protocol):
        t, protocol = t_and_protocol
        params = VenousParams(dilation_amplitude=4.0)
        y = venous_template(t, protocol, params)
        bl_idx = np.where(t < 20.0)[0]
        assert np.abs(np.mean(y[bl_idx])) < 0.1

    def test_peak_within_range(self, t_and_protocol):
        t, protocol = t_and_protocol
        rng = np.random.default_rng(0)
        for _ in range(20):
            params = sample_venous_params(rng)
            y = venous_template(t, protocol, params)
            peak = np.max(y)
            assert 1.0 <= peak <= 8.0, f"Peak {peak:.2f}% outside expected range"

    def test_monotonic_during_flicker(self, t_and_protocol):
        t, protocol = t_and_protocol
        params = VenousParams(dilation_amplitude=4.0)
        y = venous_template(t, protocol, params)
        cyc = protocol.cycles[0]
        fl_idx = np.where(cyc.flicker.contains(t))[0]
        y_fl = y[fl_idx]
        mid = len(y_fl) // 2
        assert np.mean(y_fl[mid:]) > np.mean(y_fl[:mid]), "Not increasing during flicker"

    def test_no_constriction(self, t_and_protocol):
        t, protocol = t_and_protocol
        rng = np.random.default_rng(0)
        for _ in range(20):
            params = sample_venous_params(rng)
            y = venous_template(t, protocol, params)
            assert np.min(y) >= -0.5, f"Venous trace dips to {np.min(y):.2f}%"

    def test_three_cycles_present(self, t_and_protocol):
        t, protocol = t_and_protocol
        params = VenousParams(dilation_amplitude=4.0)
        y = venous_template(t, protocol, params)
        for cyc in protocol.cycles:
            fl_idx = np.where(cyc.flicker.contains(t))[0]
            assert np.max(y[fl_idx]) > 0.5, f"Cycle {cyc.index} has no dilation"

    def test_peak_near_flicker_offset(self, t_and_protocol):
        t, protocol = t_and_protocol
        params = VenousParams(dilation_amplitude=4.0)
        y = venous_template(t, protocol, params)
        cyc = protocol.cycles[0]
        fl_end = cyc.flicker.end_sec
        w = TimeWindow("search", cyc.flicker.start_sec, fl_end + 10.0)
        w_idx = np.where(w.contains(t))[0]
        peak_t = t[w_idx[np.argmax(y[w_idx])]]
        assert abs(peak_t - fl_end) < 5.0, f"Peak at {peak_t:.1f}s, expected near {fl_end:.1f}s"


# ---------------------------------------------------------------------------
# Parameter sampling
# ---------------------------------------------------------------------------

class TestParameterSampling:
    def test_arterial_params_in_range(self):
        rng = np.random.default_rng(0)
        for _ in range(100):
            p = sample_arterial_params(rng)
            assert 1.0 <= p.dilation_amplitude <= 6.0
            assert p.constriction_depth > 0
            assert abs(p.time_shift) <= 3.0

    def test_venous_params_in_range(self):
        rng = np.random.default_rng(0)
        for _ in range(100):
            p = sample_venous_params(rng)
            assert 2.0 <= p.dilation_amplitude <= 7.0
            assert abs(p.time_shift) <= 3.0

    def test_intercycle_jitter_stays_in_bounds(self):
        rng = np.random.default_rng(0)
        base = sample_arterial_params(rng)
        for _ in range(100):
            jittered = apply_intercycle_jitter(base, rng)
            assert jittered.dilation_amplitude > 0
            assert jittered.constriction_depth >= 0

    def test_jitter_produces_variation(self):
        rng = np.random.default_rng(42)
        base = sample_arterial_params(rng)
        jittered = [apply_intercycle_jitter(base, rng) for _ in range(10)]
        amps = [j.dilation_amplitude for j in jittered]
        assert np.std(amps) > 0

    def test_reproducible_with_seed(self):
        p1 = sample_arterial_params(np.random.default_rng(42))
        p2 = sample_arterial_params(np.random.default_rng(42))
        assert p1.dilation_amplitude == p2.dilation_amplitude
