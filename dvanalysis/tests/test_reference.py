"""Reference output comparison — verify the refactored library produces identical results.

These tests load real DVA data, run preprocessing, and compare against saved reference outputs.
If reference outputs don't exist, the tests are skipped (they must be generated first).

Run with: conda run -n dvanalysis python -m pytest tests/test_reference.py -v -s
"""
from pathlib import Path

import numpy as np
import pytest

from dvanalysis.domain import TimeWindow, StimulusCycle, StimulusProtocol
from dvanalysis.io.readers import ImedosReader, ImedosReaderSettings, DataReaderConfig
from dvanalysis.preprocessing.rpca_denoise import MyMethodRPCA, MyMethodConfig
from dvanalysis.preprocessing.kotliar import KotliarPreprocessor, KotliarConfig

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = _PROJECT_ROOT / "data" / "Healthy_Volunteers"
REF_DIR = _PROJECT_ROOT / "data" / "reference_outputs"

def _discover_subjects(ref_dir):
    """Take subject IDs from whatever reference files exist locally, so no
    identifiers are hardcoded in the repo. Empty (e.g. no data) -> tests skip."""
    if not ref_dir.exists():
        return []
    return sorted({p.name.split("_")[0] for p in ref_dir.glob("*_rpca_trace.npy")})


TEST_SUBJECTS = _discover_subjects(REF_DIR)
SEGMENTS = ["A1", "A2", "V3", "V4"]


def _make_protocol():
    gb = TimeWindow("baseline", 0.0, 20.0)
    cycles = []
    for i, (b_s, f_s, r_s, r_e) in enumerate([(20, 50, 70, 120), (120, 150, 170, 220), (220, 250, 270, 320)]):
        cycles.append(StimulusCycle(index=i,
            baseline=TimeWindow("baseline", float(b_s), float(f_s)),
            flicker=TimeWindow("flicker", float(f_s), float(r_s)),
            recovery=TimeWindow("recovery", float(r_s), float(r_e))))
    return StimulusProtocol(name="DVA_3cycle_custom", fs=25.0, global_baseline=gb, cycles=cycles)


@pytest.fixture(scope="module")
def dataset():
    if not DATA_DIR.exists():
        pytest.skip(f"Data directory not found: {DATA_DIR}")
    protocol = _make_protocol()
    config = DataReaderConfig(protocol=protocol)
    reader = ImedosReader(config=config)
    return reader.read(DATA_DIR)


@pytest.fixture(scope="module")
def rpca_config():
    return MyMethodConfig(
        rpca_lmb=0.01, rpca_mu=1.0, rpca_rho=5.0, rpca_gamma=0.0,
        rpca_max_iter=200, rpca_tol_rel=1e-4,
        standardize=True, harmonize_output=True,
        harmonize_percent_mode="delta_over_baseline",
        heartbeat_filter=False, fill_missing=False,
        support_min_valid_abs=12,
    )


class TestReferenceRPCA:
    """Verify RPCA outputs match saved reference to numerical precision."""

    @pytest.mark.parametrize("subject_id", TEST_SUBJECTS)
    @pytest.mark.parametrize("seg_label", SEGMENTS)
    def test_rpca_s_hat_matches_reference(self, dataset, rpca_config, subject_id, seg_label):
        ref_path = REF_DIR / f"{subject_id}_0_{seg_label}_rpca_trace.npy"
        if not ref_path.exists():
            pytest.skip(f"Reference not found: {ref_path}")

        rec = dataset.get_recording(subject_id, "0")
        if seg_label not in rec.segments:
            pytest.skip(f"Segment {seg_label} not in recording {subject_id}")

        sig = rec.segments[seg_label].signal
        rpca = MyMethodRPCA(config=rpca_config)
        result = rpca.run(sig)

        ref = np.load(ref_path)
        # Both should have same shape
        assert result.s_hat.shape == ref.shape, (
            f"Shape mismatch: got {result.s_hat.shape}, ref {ref.shape}"
        )

        # Compare on finite entries (NaN positions should match too)
        nan_match = np.isnan(result.s_hat) == np.isnan(ref)
        assert np.all(nan_match), (
            f"NaN pattern mismatch: {np.sum(~nan_match)} positions differ"
        )

        finite = np.isfinite(ref) & np.isfinite(result.s_hat)
        if finite.sum() == 0:
            return  # both all-NaN, that's a match

        np.testing.assert_allclose(
            result.s_hat[finite], ref[finite], atol=1e-10,
            err_msg=f"RPCA s_hat mismatch for {subject_id}_{seg_label}"
        )

    @pytest.mark.parametrize("subject_id", TEST_SUBJECTS)
    @pytest.mark.parametrize("seg_label", SEGMENTS)
    def test_rpca_S_hat_matches_reference(self, dataset, rpca_config, subject_id, seg_label):
        ref_path = REF_DIR / f"{subject_id}_0_{seg_label}_rpca_loci.npy"
        if not ref_path.exists():
            pytest.skip(f"Reference not found: {ref_path}")

        rec = dataset.get_recording(subject_id, "0")
        if seg_label not in rec.segments:
            pytest.skip(f"Segment {seg_label} not in recording {subject_id}")

        sig = rec.segments[seg_label].signal
        rpca = MyMethodRPCA(config=rpca_config)
        result = rpca.run(sig)

        ref = np.load(ref_path)
        assert result.S_hat.shape == ref.shape

        finite = np.isfinite(ref) & np.isfinite(result.S_hat)
        if finite.sum() == 0:
            return

        np.testing.assert_allclose(
            result.S_hat[finite], ref[finite], atol=1e-10,
            err_msg=f"RPCA S_hat mismatch for {subject_id}_{seg_label}"
        )


class TestReferenceKotliar:
    """Verify Kotliar outputs match saved reference."""

    @pytest.mark.parametrize("subject_id", TEST_SUBJECTS)
    @pytest.mark.parametrize("seg_label", SEGMENTS)
    def test_kotliar_s_hat_matches_reference(self, dataset, subject_id, seg_label):
        ref_path = REF_DIR / f"{subject_id}_0_{seg_label}_kotliar_s_hat.npy"
        if not ref_path.exists():
            pytest.skip(f"Reference not found: {ref_path}")

        rec = dataset.get_recording(subject_id, "0")
        if seg_label not in rec.segments:
            pytest.skip(f"Segment {seg_label} not in recording {subject_id}")

        sig = rec.segments[seg_label].signal
        kot = KotliarPreprocessor(config=KotliarConfig(
            mode="smooth_only", smooth_only_keep_outside_cycles=True))
        result = kot.run(sig)

        ref = np.load(ref_path)
        assert result.s_hat.shape == ref.shape

        nan_match = np.isnan(result.s_hat) == np.isnan(ref)
        assert np.all(nan_match)

        finite = np.isfinite(ref) & np.isfinite(result.s_hat)
        if finite.sum() == 0:
            return

        np.testing.assert_allclose(
            result.s_hat[finite], ref[finite], atol=1e-10,
            err_msg=f"Kotliar s_hat mismatch for {subject_id}_{seg_label}"
        )
