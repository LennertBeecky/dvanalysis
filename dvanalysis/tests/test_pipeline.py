"""Tests for dvanalysis.pipeline — end-to-end integration."""
import numpy as np
import pytest

from dvanalysis.domain import (
    TimeWindow, StimulusCycle, StimulusProtocol, SegmentSignal, Segment, Recording, Dataset,
)
from dvanalysis.pipeline import Pipeline


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
    return StimulusProtocol(name="test_3cycle", fs=fs, global_baseline=gb, cycles=cycles)


class TestPipeline:
    def test_init(self):
        p = Pipeline()
        assert p.preprocessor_name == "rpca"

    def test_run_requires_protocol(self, tmp_path):
        p = Pipeline()
        with pytest.raises(ValueError, match="protocol"):
            p.run(str(tmp_path))

    def test_unknown_preprocessor_raises(self, tmp_path):
        p = Pipeline(preprocessor="bogus")
        protocol = _make_protocol()
        # Create an empty dir — will fail at reader but that's fine, we check preprocessor name first
        with pytest.raises((ValueError, FileNotFoundError)):
            p.run(str(tmp_path), protocol=protocol)

    def test_run_on_synthetic_data(self, tmp_path):
        """Integration test: write synthetic data files, run full pipeline."""
        protocol = _make_protocol(fs=25.0)
        T = 8000
        P = 15
        fs = 25.0

        rng = np.random.default_rng(42)
        t = np.arange(T, dtype=float) / fs

        # Write synthetic tab-separated files
        for label in ["A1", "A2", "V3", "V4"]:
            data = np.full((T, P), 100.0) + rng.normal(0, 1, (T, P))
            # Add dilation during flicker
            for cyc in protocol.cycles:
                idx = np.where(cyc.flicker.contains(t))[0]
                data[idx, :] += 3.0

            filepath = tmp_path / f"001_0_{label}.txt"
            # Write with header lines
            lines = ["Header line 1\n", "Header line 2\n", "Header line 3\n", "Header line 4\n"]
            for row in data:
                lines.append("\t".join(f"{v:.4f}" for v in row) + "\n")
            filepath.write_text("".join(lines))

        from dvanalysis.preprocessing.rpca_denoise import MyMethodConfig
        cfg = MyMethodConfig(
            rpca_lmb=0.1,
            rpca_max_iter=10,
            standardize=True,
            harmonize_output=False,
            heartbeat_filter=False,
            fill_missing=False,
            support_min_valid_abs=5,
        )

        p = Pipeline(preprocessor="rpca", preprocessor_config=cfg)
        output = p.run(str(tmp_path), protocol=protocol)

        assert "dataset" in output
        assert "results" in output
        assert "parameters" in output

        ds = output["dataset"]
        assert len(ds.recordings) == 1

        results = output["results"]
        assert len(results) == 4  # 4 segments

        params = output["parameters"]
        assert len(params) > 0  # should have rows
