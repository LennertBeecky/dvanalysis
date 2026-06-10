"""Tests for dvanalysis.biomarkers."""
import numpy as np
import pytest

from dvanalysis.domain import TimeWindow, StimulusCycle, StimulusProtocol, SegmentSignal, Segment
from dvanalysis.biomarkers.windows import as_window, idx_for_window
from dvanalysis.biomarkers.library import (
    calc_dilation_max, calc_constr_max, calc_da, calc_baseline_diameter, calc_dilation_max_t,
)
from dvanalysis.biomarkers.definitions import make_default_definitions, make_paper_definitions
from dvanalysis.biomarkers.primitives import masked_trace_median
from dvanalysis.biomarkers.extraction import ParameterExtractor, ParameterExtractorConfig


def _make_protocol(fs=25.0):
    gb = TimeWindow("baseline", 0.0, 20.0)
    cycles = []
    t_start = 20.0
    for i in range(3):
        b = TimeWindow("baseline", t_start, t_start + 30.0)
        f = TimeWindow("flicker", t_start + 30.0, t_start + 50.0)
        r = TimeWindow("recovery", t_start + 50.0, t_start + 100.0)
        cycles.append(StimulusCycle(index=i, baseline=b, flicker=f, recovery=r))
        t_start += 100.0
    return StimulusProtocol(name="test", fs=fs, global_baseline=gb, cycles=cycles)


# Deterministic trace: baseline=100, flicker peak=105, recovery dip=97
def _make_trace():
    fs = 25.0
    T = 8000
    t = np.arange(T, dtype=float) / fs
    y = np.full(T, 100.0)
    protocol = _make_protocol(fs)
    for cyc in protocol.cycles:
        fi = np.where(cyc.flicker.contains(t))[0]
        ri = np.where(cyc.recovery.contains(t))[0]
        y[fi] = 105.0
        y[ri] = 97.0
    return t, y, protocol


class TestWindows:
    def test_as_window_from_tuple(self):
        w = as_window((10.0, 20.0))
        assert isinstance(w, TimeWindow)
        assert w.start_sec == 10.0

    def test_as_window_passthrough(self):
        w = TimeWindow("x", 1.0, 2.0)
        assert as_window(w) is w

    def test_idx_for_window(self):
        t = np.arange(0, 10, 0.1)
        idx = idx_for_window(t, (2.0, 3.0))
        assert idx[0] == 20
        assert t[idx[-1]] < 3.0

    def test_idx_for_window_strict_empty(self):
        t = np.arange(0, 5, 0.1)
        with pytest.raises(ValueError):
            idx_for_window(t, (10.0, 20.0), strict=True)


class TestLibrary:
    def test_calc_dilation_max(self):
        t, y, protocol = _make_trace()
        cyc = protocol.cycles[0]
        assert calc_dilation_max(y, t, cyc.flicker) == 105.0

    def test_calc_constr_max(self):
        t, y, protocol = _make_trace()
        cyc = protocol.cycles[0]
        assert calc_constr_max(y, t, cyc.recovery) == 97.0

    def test_calc_da(self):
        t, y, protocol = _make_trace()
        cyc = protocol.cycles[0]
        assert calc_da(y, t, cyc.flicker, cyc.recovery) == 8.0

    def test_calc_baseline_diameter(self):
        t, y, protocol = _make_trace()
        assert calc_baseline_diameter(y, t, protocol.global_baseline) == 100.0

    def test_calc_dilation_max_t(self):
        t, y, protocol = _make_trace()
        cyc = protocol.cycles[0]
        # All flicker values are equal so argmax returns first index
        tmax = calc_dilation_max_t(y, t, cyc.flicker, relative_to_start=True)
        assert tmax >= 0.0
        assert tmax < 20.0  # within flicker duration


class TestDefinitions:
    def test_default_has_3(self):
        defs = make_default_definitions()
        assert len(defs) == 3
        assert "max_dilation" in defs

    def test_paper_has_9(self):
        defs = make_paper_definitions()
        assert len(defs) == 9
        assert "time_to_max_dilation" in defs
        assert "time_to_max_dilation_30" in defs

    def test_default_compute_callable(self):
        t, y, protocol = _make_trace()
        defs = make_default_definitions()
        cyc = protocol.cycles[0]
        val = defs["max_dilation"].compute(y, t, cyc)
        assert val == 105.0


class TestPrimitives:
    def test_masked_trace_median(self):
        rng = np.random.default_rng(0)
        protocol = _make_protocol()
        t = np.arange(100, dtype=float) / 25.0
        x = rng.normal(50, 1, (100, 5))
        m = np.ones((100, 5), dtype=bool)
        sig = SegmentSignal(t=t, x=x, m=m, protocol=protocol)
        y = masked_trace_median(sig)
        assert y.shape == (100,)


class TestExtractor:
    def test_analyze_segment_per_cycle(self):
        t, y, protocol = _make_trace()
        T = len(t)
        P = 5
        x = np.tile(y[:, None], (1, P))
        m = np.ones((T, P), dtype=bool)
        sig = SegmentSignal(t=t, x=x, m=m, protocol=protocol)
        seg = Segment(segment_label="A1", vessel_type="artery", signal=sig)

        cfg = ParameterExtractorConfig(trace_source="raw", raw_agg="median", strict=False)
        ext = ParameterExtractor(config=cfg)
        rows = ext.analyze_segment(seg, sig, method_name="test")
        assert len(rows) == 3  # 3 cycles
        assert rows[0]["max_dilation"] == 105.0
        assert rows[0]["dilation_amplitude"] == 8.0
