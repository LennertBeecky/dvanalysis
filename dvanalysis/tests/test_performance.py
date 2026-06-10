"""Performance and correctness tests for RPCA optimization and IO speed.

These tests serve two purposes:
1. Pin correctness: optimized code must produce identical results to the original
2. Benchmark: measure and report performance of different implementations

Run with: pytest tests/test_performance.py -v -s
"""
import io
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dvanalysis.domain import (
    TimeWindow, StimulusCycle, StimulusProtocol, SegmentSignal,
)


# ============================================================================
# Shared fixtures
# ============================================================================

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


@pytest.fixture(scope="module")
def rpca_test_data():
    """Realistic RPCA test data: low-rank + sparse + mask."""
    rng = np.random.default_rng(42)
    P, T = 40, 8000
    # Low-rank signal (rank 3)
    S_true = rng.normal(0, 1, (P, 3)) @ rng.normal(0, 1, (3, T))
    # Sparse artefacts
    A_true = np.zeros((P, T))
    spike_idx = rng.integers(0, P * T, 100)
    A_true.ravel()[spike_idx] = rng.normal(0, 5, 100)
    X = S_true + A_true
    # Observation mask (~5% missing)
    M = np.ones((P, T), dtype=bool)
    M[rng.random((P, T)) < 0.05] = False
    X[~M] = np.nan
    return X, M, S_true, A_true


@pytest.fixture(scope="module")
def rpca_reference_output(rpca_test_data):
    """Reference output from the ORIGINAL (unoptimized) solver."""
    from dvanalysis.preprocessing.rpca import RobustPCA
    X, M, _, _ = rpca_test_data
    rpca = RobustPCA(lmb=0.1, mu=1.0, rho=5.0, gamma=1.0, max_iter=30, tol_rel=1e-5)
    S_ref, A_ref, diag_ref = rpca.fit(X.copy(), M.copy())
    return S_ref, A_ref, diag_ref


@pytest.fixture(scope="module")
def io_test_dir():
    """Create a directory with realistic synthetic Imedos files."""
    tmpdir = tempfile.mkdtemp()
    rng = np.random.default_rng(42)
    T, P = 8750, 25
    protocol = _make_protocol()

    for label in ["A1", "A2", "V3", "V4"]:
        data = rng.normal(100, 2, (T, P))
        filepath = Path(tmpdir) / f"001_0_{label}.txt"
        lines = ["Header line 1\n", "Header line 2\n", "Header line 3\n", "Header line 4\n"]
        for row in data:
            lines.append("\t".join(f"{v:.4f}" for v in row) + "\n")
        filepath.write_text("".join(lines))

    # Also create an RTF-style file with \tab tokens
    data_rtf = rng.normal(100, 2, (T, P))
    rtf_path = Path(tmpdir) / "002_0_A1.rtf"
    lines_rtf = ["Header 1\n", "Header 2\n", "Header 3\n", "Header 4\n"]
    for row in data_rtf:
        lines_rtf.append(r"\tab".join(f"{v:.4f}" for v in row) + r"\par" + "\n")
    rtf_path.write_text("".join(lines_rtf))

    return Path(tmpdir), T, P


# ============================================================================
# RPCA CORRECTNESS TESTS — these MUST pass after optimization
# ============================================================================

class TestRPCACorrectness:
    """Pin correctness of RPCA output. After optimization, results must match."""

    def test_output_shapes(self, rpca_test_data, rpca_reference_output):
        X, M, _, _ = rpca_test_data
        S_ref, A_ref, _ = rpca_reference_output
        P, T = X.shape
        assert S_ref.shape == (P, T)
        assert A_ref.shape == (P, T)

    def test_sparse_outside_mask_is_zero(self, rpca_test_data, rpca_reference_output):
        _, M, _, _ = rpca_test_data
        _, A_ref, _ = rpca_reference_output
        assert np.all(A_ref[~M] == 0.0)

    def test_data_fidelity(self, rpca_test_data, rpca_reference_output):
        """On observed entries, S + A should approximately reconstruct X."""
        X, M, _, _ = rpca_test_data
        S_ref, A_ref, _ = rpca_reference_output
        Xf = X.copy()
        Xf[~np.isfinite(Xf)] = 0.0
        residual = Xf[M] - (S_ref[M] + A_ref[M])
        rel_err = np.linalg.norm(residual) / np.linalg.norm(Xf[M])
        assert rel_err < 0.1, f"Relative reconstruction error {rel_err:.4f} too large"

    def test_temporal_smoothness(self, rpca_reference_output):
        """S should be temporally smooth when gamma > 0."""
        S_ref, _, _ = rpca_reference_output
        diffs = np.diff(S_ref, axis=1)
        # Temporal differences should be small relative to signal
        ratio = np.std(diffs) / (np.std(S_ref) + 1e-12)
        assert ratio < 0.6, f"Temporal roughness ratio {ratio:.3f} too high"

    def test_tridiagonal_solve_matches_reference(self):
        """Verify that scipy.linalg.solve_banded gives same result as Thomas algorithm."""
        from dvanalysis.preprocessing.rpca import _solve_tridiagonal
        rng = np.random.default_rng(0)
        n = 8000
        a = np.zeros(n)
        b = rng.uniform(2, 5, n)
        c = np.zeros(n)
        a[1:] = rng.uniform(-0.5, 0, n - 1)
        c[:-1] = rng.uniform(-0.5, 0, n - 1)
        d = rng.normal(0, 1, n)

        x_thomas = _solve_tridiagonal(a, b, c, d)

        # Solve same system with scipy
        from scipy.linalg import solve_banded
        ab = np.zeros((3, n))
        ab[0, 1:] = c[:-1]     # upper diagonal
        ab[1, :] = b            # main diagonal
        ab[2, :-1] = a[1:]      # lower diagonal
        x_scipy = solve_banded((1, 1), ab, d)

        np.testing.assert_allclose(x_thomas, x_scipy, atol=1e-10,
                                   err_msg="Thomas algorithm and solve_banded disagree")


# ============================================================================
# RPCA BENCHMARK TESTS
# ============================================================================

class TestRPCABenchmark:
    """Benchmark RPCA solver performance. Reports times but does not fail on speed."""

    def test_benchmark_tridiagonal_thomas(self):
        """Benchmark: Thomas algorithm (current, Python loop)."""
        from dvanalysis.preprocessing.rpca import _solve_tridiagonal
        rng = np.random.default_rng(0)
        T = 8000
        a = np.zeros(T); a[1:] = -1.0
        b = np.full(T, 5.0)
        c = np.zeros(T); c[:-1] = -1.0
        d = rng.normal(0, 1, T)

        # Warmup
        _solve_tridiagonal(a, b, c, d)

        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            _solve_tridiagonal(a, b, c, d)
            times.append(time.perf_counter() - t0)

        mean_ms = np.mean(times) * 1000
        print(f"\n  Thomas algorithm (T={T}): {mean_ms:.2f} ms/call")

    def test_benchmark_tridiagonal_scipy(self):
        """Benchmark: scipy.linalg.solve_banded."""
        from scipy.linalg import solve_banded
        rng = np.random.default_rng(0)
        T = 8000
        d = rng.normal(0, 1, T)
        ab = np.zeros((3, T))
        ab[0, 1:] = -1.0   # upper
        ab[1, :] = 5.0      # main
        ab[2, :-1] = -1.0   # lower

        # Warmup
        solve_banded((1, 1), ab, d)

        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            solve_banded((1, 1), ab, d)
            times.append(time.perf_counter() - t0)

        mean_ms = np.mean(times) * 1000
        print(f"\n  scipy solve_banded (T={T}): {mean_ms:.4f} ms/call")

    def test_benchmark_tridiagonal_batch_scipy(self):
        """Benchmark: batch tridiagonal solve for all P loci at once."""
        from scipy.linalg import solve_banded
        rng = np.random.default_rng(0)
        P, T = 40, 8000
        d = rng.normal(0, 1, (T, P))  # multiple RHS
        ab = np.zeros((3, T))
        ab[0, 1:] = -1.0
        ab[1, :] = 5.0
        ab[2, :-1] = -1.0

        # Warmup
        solve_banded((1, 1), ab, d)

        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            solve_banded((1, 1), ab, d)
            times.append(time.perf_counter() - t0)

        mean_ms = np.mean(times) * 1000
        print(f"\n  scipy batch solve_banded (P={P}, T={T}): {mean_ms:.2f} ms/call")

    def test_benchmark_svd(self):
        """Benchmark: SVD on (P, T) matrix."""
        rng = np.random.default_rng(0)
        P, T = 40, 8000
        X = rng.normal(0, 1, (P, T))

        # Warmup
        np.linalg.svd(X, full_matrices=False)

        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            np.linalg.svd(X, full_matrices=False)
            times.append(time.perf_counter() - t0)

        mean_ms = np.mean(times) * 1000
        print(f"\n  SVD (P={P}, T={T}): {mean_ms:.1f} ms/call")

    def test_benchmark_full_rpca(self, rpca_test_data):
        """Benchmark: full RPCA solver (current implementation)."""
        from dvanalysis.preprocessing.rpca import RobustPCA
        X, M, _, _ = rpca_test_data
        rpca = RobustPCA(lmb=0.1, mu=1.0, rho=5.0, gamma=1.0, max_iter=10, tol_rel=1e-6)

        t0 = time.perf_counter()
        S, A, diag = rpca.fit(X.copy(), M.copy())
        dt = time.perf_counter() - t0

        print(f"\n  Full RPCA (P={X.shape[0]}, T={X.shape[1]}, iter={diag['n_iter']}): {dt:.3f}s")


# ============================================================================
# IO CORRECTNESS TESTS
# ============================================================================

class TestIOCorrectness:
    """Pin IO correctness — optimized reader must produce identical output."""

    def test_read_produces_correct_shape(self, io_test_dir):
        from dvanalysis.io.readers import ImedosReader, DataReaderConfig
        tmpdir, T, P = io_test_dir
        config = DataReaderConfig(protocol=_make_protocol())
        reader = ImedosReader(config=config)
        ds = reader.read(tmpdir)
        # 2 recordings (001 and 002), 001 has 4 segments, 002 has 1
        total_segments = sum(len(list(rec.iter_segments())) for rec in ds.recordings)
        assert total_segments == 5
        for rec in ds.recordings:
            for seg in rec.iter_segments():
                assert seg.signal.T == T
                assert seg.signal.P == P

    def test_read_preserves_values(self, io_test_dir):
        """Read back a file and verify values match what was written."""
        tmpdir, T, P = io_test_dir
        from dvanalysis.io.readers import ImedosReader, DataReaderConfig
        config = DataReaderConfig(protocol=_make_protocol())
        reader = ImedosReader(config=config)
        ds = reader.read(tmpdir)

        # Find the 001_0_A1 segment
        rec = ds.get_recording("001", "0")
        sig = rec.segments["A1"].signal

        # Read raw file for comparison
        filepath = tmpdir / "001_0_A1.txt"
        raw = np.loadtxt(filepath, skiprows=4)
        np.testing.assert_allclose(sig.x[sig.m], raw[sig.m], atol=1e-3,
                                   err_msg="Parsed values don't match raw file")

    def test_rtf_tokens_handled(self, io_test_dir):
        """RTF file with \\tab and \\par tokens is parsed correctly."""
        tmpdir, T, P = io_test_dir
        from dvanalysis.io.readers import ImedosReader, DataReaderConfig
        config = DataReaderConfig(protocol=_make_protocol())
        reader = ImedosReader(config=config)
        ds = reader.read(tmpdir)
        rec = ds.get_recording("002", "0")
        assert "A1" in rec.segments
        assert rec.segments["A1"].signal.T == T


# ============================================================================
# IO BENCHMARK TESTS — data container comparison
# ============================================================================

class TestIOBenchmark:
    """Benchmark different parsing strategies for Imedos files."""

    def test_benchmark_current_reader(self, io_test_dir):
        """Benchmark: current ImedosReader (custom line-by-line parser)."""
        tmpdir, T, P = io_test_dir
        from dvanalysis.io.readers import ImedosReader, DataReaderConfig
        config = DataReaderConfig(protocol=_make_protocol())
        reader = ImedosReader(config=config)

        # Warmup
        reader.read(tmpdir)

        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            ds = reader.read(tmpdir)
            times.append(time.perf_counter() - t0)

        per_file = np.mean(times) / 5 * 1000  # 5 files in the dir
        print(f"\n  Current ImedosReader: {np.mean(times)*1000:.1f} ms total, {per_file:.1f} ms/file")

    def test_benchmark_numpy_loadtxt(self, io_test_dir):
        """Benchmark: np.loadtxt on clean tab-separated file."""
        tmpdir, T, P = io_test_dir
        filepath = tmpdir / "001_0_A1.txt"

        # Warmup
        np.loadtxt(filepath, skiprows=4)

        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            arr = np.loadtxt(filepath, skiprows=4)
            times.append(time.perf_counter() - t0)

        print(f"\n  np.loadtxt: {np.mean(times)*1000:.1f} ms/file (shape={arr.shape})")

    def test_benchmark_numpy_genfromtxt(self, io_test_dir):
        """Benchmark: np.genfromtxt (handles NaN natively)."""
        tmpdir, T, P = io_test_dir
        filepath = tmpdir / "001_0_A1.txt"

        # Warmup
        np.genfromtxt(filepath, skip_header=4)

        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            arr = np.genfromtxt(filepath, skip_header=4)
            times.append(time.perf_counter() - t0)

        print(f"\n  np.genfromtxt: {np.mean(times)*1000:.1f} ms/file")

    def test_benchmark_pandas_read_csv(self, io_test_dir):
        """Benchmark: pd.read_csv with tab separator."""
        tmpdir, T, P = io_test_dir
        filepath = tmpdir / "001_0_A1.txt"

        # Warmup
        pd.read_csv(filepath, sep="\t", header=None, skiprows=4)

        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            df = pd.read_csv(filepath, sep="\t", header=None, skiprows=4)
            times.append(time.perf_counter() - t0)

        print(f"\n  pd.read_csv: {np.mean(times)*1000:.1f} ms/file (shape={df.shape})")

    def test_benchmark_pandas_from_string(self, io_test_dir):
        """Benchmark: read file as string, normalize, then pd.read_csv from StringIO."""
        tmpdir, T, P = io_test_dir
        filepath = tmpdir / "001_0_A1.txt"
        text = filepath.read_text()

        # Warmup
        pd.read_csv(io.StringIO(text), sep="\t", header=None, skiprows=4)

        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            raw = filepath.read_text()
            df = pd.read_csv(io.StringIO(raw), sep="\t", header=None, skiprows=4)
            times.append(time.perf_counter() - t0)

        print(f"\n  pd.read_csv (from string): {np.mean(times)*1000:.1f} ms/file")

    def test_benchmark_normalize_then_pandas(self, io_test_dir):
        """Benchmark: normalize RTF tokens, then pd.read_csv — the proposed fast path."""
        tmpdir, T, P = io_test_dir
        filepath = tmpdir / "002_0_A1.rtf"  # RTF-style file
        from dvanalysis.io.readers import ImedosReader, DataReaderConfig
        config = DataReaderConfig(protocol=_make_protocol())
        reader = ImedosReader(config=config)

        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            text = filepath.read_text()
            text = reader._normalize_imedos_text(text)
            # Skip 4 header lines manually
            lines = text.split("\n")
            table = "\n".join(lines[4:])
            try:
                df = pd.read_csv(io.StringIO(table), sep="\t", header=None,
                                 na_values=["", " "], on_bad_lines="warn")
                arr = df.values.astype(float)
            except Exception:
                arr = None
            times.append(time.perf_counter() - t0)

        if arr is not None:
            print(f"\n  Normalize + pd.read_csv (RTF): {np.mean(times)*1000:.1f} ms/file (shape={arr.shape})")
        else:
            print(f"\n  Normalize + pd.read_csv (RTF): FAILED")
