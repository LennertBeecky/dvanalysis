"""Tests for dvanalysis.domain — data model classes."""
import numpy as np
import pytest

from dvanalysis.domain import (
    TimeWindow,
    StimulusCycle,
    StimulusProtocol,
    SegmentSignal,
    Segment,
    Recording,
    Dataset,
)


# ---------------------------------------------------------------------------
# TimeWindow
# ---------------------------------------------------------------------------

class TestTimeWindow:
    def test_contains(self):
        w = TimeWindow(phase="flicker", start_sec=10.0, end_sec=30.0)
        t = np.array([5.0, 10.0, 20.0, 30.0, 35.0])
        mask = w.contains(t)
        np.testing.assert_array_equal(mask, [False, True, True, False, False])

    def test_to_indices(self):
        w = TimeWindow(phase="baseline", start_sec=0.0, end_sec=2.0)
        t = np.arange(0, 5, 0.5)
        idx = w.to_indices(t)
        np.testing.assert_array_equal(idx, [0, 1, 2, 3])

    def test_to_indices_from_fs(self):
        w = TimeWindow(phase="flicker", start_sec=1.0, end_sec=3.0)
        idx = w.to_indices_from_fs(fs=10.0)
        np.testing.assert_array_equal(idx, np.arange(10, 30))

    def test_frozen(self):
        w = TimeWindow(phase="x", start_sec=0.0, end_sec=1.0)
        with pytest.raises(AttributeError):
            w.phase = "y"


# ---------------------------------------------------------------------------
# StimulusCycle
# ---------------------------------------------------------------------------

class TestStimulusCycle:
    def test_all_windows_with_baseline(self):
        b = TimeWindow("baseline", 0, 30)
        f = TimeWindow("flicker", 30, 50)
        r = TimeWindow("recovery", 50, 100)
        cyc = StimulusCycle(index=0, baseline=b, flicker=f, recovery=r)
        assert len(cyc.all_windows()) == 3

    def test_all_windows_without_baseline(self):
        f = TimeWindow("flicker", 30, 50)
        r = TimeWindow("recovery", 50, 100)
        cyc = StimulusCycle(index=0, baseline=None, flicker=f, recovery=r)
        assert len(cyc.all_windows()) == 2


# ---------------------------------------------------------------------------
# StimulusProtocol
# ---------------------------------------------------------------------------

def _make_protocol():
    """Standard 3-cycle DVA protocol for testing."""
    gb = TimeWindow("baseline", 0.0, 20.0)
    cycles = []
    t = 20.0
    for i in range(3):
        b = TimeWindow("baseline", t, t + 30.0)
        f = TimeWindow("flicker", t + 30.0, t + 50.0)
        r = TimeWindow("recovery", t + 50.0, t + 100.0)
        cycles.append(StimulusCycle(index=i, baseline=b, flicker=f, recovery=r))
        t += 100.0
    return StimulusProtocol(name="test_3cycle", fs=25.0, global_baseline=gb, cycles=cycles)


class TestStimulusProtocol:
    def test_flicker_windows(self):
        p = _make_protocol()
        fw = p.flicker_windows()
        assert len(fw) == 3
        assert fw[0].start_sec == 50.0

    def test_baseline_windows_includes_global(self):
        p = _make_protocol()
        bw = p.baseline_windows()
        assert len(bw) == 4  # global + 3 cycle baselines

    def test_end_time(self):
        p = _make_protocol()
        assert p.end_time_sec() == 320.0

    def test_mask_for_phase(self):
        p = _make_protocol()
        t = np.arange(0, 320, 1.0 / 25.0)
        m = p.mask_for_phase(t, "flicker")
        # 3 flicker windows of 20s each at 25 Hz = 3*500 = 1500 True values
        assert m.sum() == 1500


# ---------------------------------------------------------------------------
# SegmentSignal
# ---------------------------------------------------------------------------

def _make_signal(T=100, P=5, fs=25.0):
    """Create a synthetic SegmentSignal for testing."""
    t = np.arange(T, dtype=float) / fs
    rng = np.random.default_rng(42)
    x = rng.normal(100.0, 2.0, size=(T, P))
    m = np.ones((T, P), dtype=bool)
    # introduce some missing data
    m[10:15, 2] = False
    x[10:15, 2] = np.nan

    protocol = StimulusProtocol(
        name="mini", fs=fs,
        global_baseline=TimeWindow("baseline", 0.0, 1.0),
        cycles=[
            StimulusCycle(
                index=0,
                baseline=TimeWindow("baseline", 0.0, 1.0),
                flicker=TimeWindow("flicker", 1.0, 2.0),
                recovery=TimeWindow("recovery", 2.0, 4.0),
            )
        ],
    )
    return SegmentSignal(t=t, x=x, m=m, protocol=protocol)


class TestSegmentSignal:
    def test_shape_properties(self):
        sig = _make_signal(T=100, P=5)
        assert sig.T == 100
        assert sig.P == 5

    def test_mask_enforces_nan(self):
        sig = _make_signal()
        # where mask is False, x should be NaN
        assert np.all(np.isnan(sig.x[~sig.m]))

    def test_arrays_read_only(self):
        sig = _make_signal()
        with pytest.raises(ValueError):
            sig.t[0] = 999.0

    def test_frozen(self):
        sig = _make_signal()
        with pytest.raises(AttributeError):
            sig.units = "mm"

    def test_valid_fraction(self):
        sig = _make_signal(T=100, P=5)
        # 5 missing out of 500
        expected = 1.0 - 5.0 / 500.0
        assert abs(sig.valid_fraction() - expected) < 1e-10

    def test_slice_window(self):
        sig = _make_signal(T=100, P=5, fs=25.0)
        w = TimeWindow("test", 0.0, 1.0)
        sliced = sig.slice_window(w)
        assert sliced.T == 25

    def test_bad_shapes_raise(self):
        with pytest.raises(ValueError):
            SegmentSignal(
                t=np.arange(10.0),
                x=np.ones((5, 3)),  # T mismatch
                m=np.ones((5, 3), dtype=bool),
                protocol=_make_protocol(),
            )

    def test_aggregate_over_loci(self):
        sig = _make_signal(T=50, P=4)
        y = sig.aggregate_over_loci(agg="median")
        assert y.shape == (50,)
        y2 = sig.aggregate_over_loci(agg="mean")
        assert y2.shape == (50,)


# ---------------------------------------------------------------------------
# Segment
# ---------------------------------------------------------------------------

class TestSegment:
    def test_creation(self):
        sig = _make_signal()
        seg = Segment(segment_label="A1", vessel_type="artery", signal=sig)
        assert seg.segment_label == "A1"
        assert seg.vessel_type == "artery"

    def test_summary(self):
        sig = _make_signal()
        seg = Segment(segment_label="V3", vessel_type="vein", signal=sig)
        s = seg.summary()
        assert "V3" in s
        assert "vein" in s


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

class TestRecording:
    def test_add_and_iter_segments(self):
        sig = _make_signal()
        rec = Recording(subject_id="001", visit_id="0")
        rec.add_segment(Segment(segment_label="A1", vessel_type="artery", signal=sig))
        rec.add_segment(Segment(segment_label="V3", vessel_type="vein", signal=sig))
        assert len(list(rec.iter_segments())) == 2
        assert "A1" in rec.segments


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TestDataset:
    def test_iter_segments(self):
        sig = _make_signal()
        ds = Dataset(name="test")
        rec = Recording(subject_id="001", visit_id="0")
        rec.add_segment(Segment(segment_label="A1", vessel_type="artery", signal=sig))
        ds.add_recording(rec)
        pairs = list(ds.iter_segments())
        assert len(pairs) == 1
        assert pairs[0][1].segment_label == "A1"

    def test_get_recording(self):
        sig = _make_signal()
        ds = Dataset(name="test")
        rec = Recording(subject_id="001", visit_id="0")
        rec.add_segment(Segment(segment_label="A1", vessel_type="artery", signal=sig))
        ds.add_recording(rec)
        found = ds.get_recording("001", "0")
        assert found is rec

    def test_get_recording_missing_raises(self):
        ds = Dataset(name="test")
        with pytest.raises(KeyError):
            ds.get_recording("nope", "nope")
