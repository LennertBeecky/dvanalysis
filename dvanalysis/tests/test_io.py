"""Tests for dvanalysis.io — data readers."""
import numpy as np
import pytest

from dvanalysis.domain import TimeWindow, StimulusCycle, StimulusProtocol
from dvanalysis.io.readers import ImedosReader, ImedosReaderSettings, DataReaderConfig


def _make_protocol(fs=25.0):
    gb = TimeWindow("baseline", 0.0, 20.0)
    cycles = [StimulusCycle(index=0,
        baseline=TimeWindow("baseline", 20.0, 50.0),
        flicker=TimeWindow("flicker", 50.0, 70.0),
        recovery=TimeWindow("recovery", 70.0, 120.0))]
    return StimulusProtocol(name="test", fs=fs, global_baseline=gb, cycles=cycles)


def _make_reader():
    config = DataReaderConfig(protocol=_make_protocol())
    return ImedosReader(config=config)


class TestVesselTypeInference:
    def test_artery_A1(self):
        r = _make_reader()
        assert r._infer_vessel_type("A1") == "artery"

    def test_artery_Art1(self):
        r = _make_reader()
        assert r._infer_vessel_type("Art1") == "artery"

    def test_vein_V3(self):
        r = _make_reader()
        assert r._infer_vessel_type("V3") == "vein"

    def test_vein_Ven2(self):
        r = _make_reader()
        assert r._infer_vessel_type("Ven2") == "vein"

    def test_unknown(self):
        r = _make_reader()
        assert r._infer_vessel_type("X1") == "unknown"


class TestFilenameParser:
    def test_standard_format(self):
        r = _make_reader()
        info = r._parse_filename("001_0_A1.rtf")
        assert info == {"subject": "001", "visit": "0", "segment": "A1"}

    def test_no_match(self):
        r = _make_reader()
        assert r._parse_filename("random_file.txt") is None


class TestSafeFloat:
    def test_normal(self):
        r = _make_reader()
        assert r._safe_float("3.14") == pytest.approx(3.14)

    def test_comma_decimal(self):
        r = _make_reader()
        assert r._safe_float("3,14") == pytest.approx(3.14)

    def test_empty(self):
        r = _make_reader()
        assert np.isnan(r._safe_float(""))

    def test_junk(self):
        r = _make_reader()
        assert np.isnan(r._safe_float("abc"))


class TestNormalizeText:
    def test_tab_token(self):
        r = _make_reader()
        assert "\t" in r._normalize_imedos_text(r"1.0\tab2.0")

    def test_par_token(self):
        r = _make_reader()
        assert "\n" in r._normalize_imedos_text(r"line1\par")


class TestReadFromFile:
    def test_read_synthetic_directory(self, tmp_path):
        """Write synthetic data files, read them back."""
        protocol = _make_protocol(fs=25.0)
        T = 200
        P = 5
        rng = np.random.default_rng(42)

        for label in ["A1", "V3"]:
            data = rng.normal(100, 1, (T, P))
            filepath = tmp_path / f"001_0_{label}.txt"
            lines = ["H1\n", "H2\n", "H3\n", "H4\n"]
            for row in data:
                lines.append("\t".join(f"{v:.4f}" for v in row) + "\n")
            filepath.write_text("".join(lines))

        config = DataReaderConfig(protocol=protocol)
        reader = ImedosReader(config=config)
        ds = reader.read(tmp_path)

        assert len(ds.recordings) == 1
        rec = ds.recordings[0]
        assert rec.subject_id == "001"
        assert "A1" in rec.segments
        assert "V3" in rec.segments
        assert rec.segments["A1"].vessel_type == "artery"
        assert rec.segments["V3"].vessel_type == "vein"
        assert rec.segments["A1"].signal.T == T
        assert rec.segments["A1"].signal.P == P
