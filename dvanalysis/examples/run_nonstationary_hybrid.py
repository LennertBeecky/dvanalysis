"""Non-stationary hybrid evaluation. Additive extension to the existing pipeline.

This script adds two pieces to the existing hybrid evaluation without
modifying any existing code, experiment, table, or figure:

1. A non-stationary hybrid experiment running SMS, SMS-avg and RPCA on a
   hybrid set whose three cycles have amplitude multipliers bootstrapped
   per-recording from an empirical SMS-derived inter-cycle distribution.
   Outputs:
     - nonstationary_results.csv  per-cycle biomarker bias, both
       conditions, all three methods, evaluated against per-cycle GT.
     - nonstationary_fidelity.csv per-sample SDR / NMSE, both conditions,
       all three methods.
     - fig_nonstationary_hybrid.{png,pdf}  Figure 5 for the manuscript:
       arterial and venous MD bias panels, three methods, signed bias
       with IQR whiskers, with median |bias| annotated for SMS-avg.
     - table5_biomarker_bias.csv  ready-to-paste content for Table 5
       (signed bias and median |bias| for amplitude biomarkers, both
       conditions, all three methods).

2. Five paired statistical comparisons against per-cycle ground truth
   (Wilcoxon + bootstrap CI, 5000 resamples, fixed seed):
       (a) RPCA vs SMS-avg, stationary
       (b) RPCA vs SMS-avg, non-stationary
       (c) RPCA vs SMS,     stationary
       (d) RPCA vs SMS,     non-stationary
       (e) SMS-avg stationary vs SMS-avg non-stationary, amplitude
           biomarkers (MD, MC, DA)
   Output: nonstationary_paired_stats.csv, intended for Appendix F.

3. An empirical CV histogram for Appendix C documenting the
   non-stationary construction:
     - empirical_amp_triplets_sms.csv  the bootstrap pool.
     - empirical_cv_summary.csv        median / mean CV per vessel.
     - fig_appendix_empirical_cv.{png,pdf}  histogram.

Implementation notes
--------------------
The non-stationary build is implemented by reproducing the body of
``dvanalysis.validation.hybrid.build_hybrid`` inline, with one
substitution: cycle parameters are constructed by applying supplied
amplitude multipliers to the base parameters. The stationary build
uses ``cycle_jitter=False`` (identical cycles).
All other assembly steps (residual tiling, masking, locus structure,
sparse-artefact handling) reuse the public helpers from
``dvanalysis.validation.{templates,noise}`` so behaviour matches the
existing build exactly aside from the per-cycle amplitude multipliers.

Usage
-----
    python dvanalysis/examples/run_nonstationary_hybrid.py \\
        --data-dir ./data \\
        --out-dir ./outputs/nonstationary \\
        [--clinical-triplets-from PATH]   # skip Stage 1 if precomputed
        [--n-segments N --n-configs K]    # subset for quick drafts
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

from dvanalysis.domain import (
    SegmentSignal, StimulusCycle, StimulusProtocol, TimeWindow,
)
from dvanalysis.io import DataReaderConfig, ImedosReader
from dvanalysis.preprocessing.rpca_denoise import MyMethodConfig, MyMethodRPCA
from dvanalysis.validation.hybrid import (
    HybridSample, SparseArtefactConfig, build_hybrid, inject_sparse_artefacts,
)
from dvanalysis.validation.templates import (
    ArterialParams, VenousParams,
    arterial_template, venous_template,
    sample_arterial_params, sample_venous_params,
)
from dvanalysis.validation.noise import (
    build_flicker_mask, extract_baseline_residual, tile_residual,
)
from dvanalysis.validation.metrics import (
    compute_biomarker_bias, compute_nmse, compute_sdr,
)


# ── Protocol (DVA_3cycle_custom) ──────────────────────────────────────
PROTOCOL = StimulusProtocol(
    name="DVA_3cycle_custom",
    fs=25.0,
    global_baseline=TimeWindow("baseline", 0.0, 20.0),
    cycles=[
        StimulusCycle(
            index=i,
            baseline=TimeWindow("baseline", float(b), float(f)),
            flicker=TimeWindow("flicker", float(f), float(r)),
            recovery=TimeWindow("recovery", float(r), float(e)),
        )
        for i, (b, f, r, e) in enumerate(
            [(20, 50, 70, 120), (120, 150, 170, 220), (220, 250, 270, 320)]
        )
    ],
)
FLICKER_DUR = 20.0
CYCLE_BASELINES = {0: (20.0, 50.0), 1: (120.0, 150.0), 2: (220.0, 250.0)}
FLICKER_ONSETS = {0: 50.0, 1: 150.0, 2: 250.0}

RPCA_CFG = MyMethodConfig(
    rpca_lmb=0.55, rpca_gamma=1000.0, rpca_max_iter=70, rpca_tol_rel=1e-3,
    standardize=True, harmonize_output=True,
    harmonize_percent_mode="delta_over_baseline",
    harmonize_baseline_source="protocol_global",
    harmonize_aggregation_order="percent_then_aggregate",
    harmonize_baseline_per_locus=True,
    support_min_valid_frac=0.75, support_min_valid_abs=12,
    hampel_enable=True,
)


# ── CLI ───────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Non-stationary hybrid evaluation, paired stats and "
                    "Appendix C empirical-CV histogram. Strictly additive "
                    "to the existing pipeline.",
    )
    p.add_argument("--data-dir", type=Path, default=Path("./data"),
                   help="Clinical DVA recordings directory (default: ./data).")
    p.add_argument("--out-dir", type=Path, default=Path("./outputs/nonstationary"),
                   help="Output directory.")
    p.add_argument("--clinical-triplets-from", type=Path, default=None,
                   help="Optional precomputed empirical triplets CSV "
                        "(seg_key, vessel_type, m0, m1, m2). Skips Stage 1.")
    p.add_argument("--subjects-json", type=Path, default=None,
                   help="JSON of retained subject IDs per segment label.")
    p.add_argument("--n-segments", type=int, default=None,
                   help="Subset clinical segments (default: all retained).")
    p.add_argument("--n-configs", type=int, default=10,
                   help="Hybrid amplitude configurations per segment "
                        "(default: 10, matching the existing experiment).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for the hybrid bootstrap.")
    p.add_argument("--paired-stats-seed", type=int, default=0,
                   help="Random seed for the bootstrap CI on paired stats.")
    p.add_argument("--n-boot", type=int, default=5000,
                   help="Bootstrap resamples for paired-stats CIs.")
    return p.parse_args()


# ── Stage 1: empirical triplets via SMS ───────────────────────────────

def _load_subjects(path):
    if path is None:
        path = Path(__file__).resolve().parent / "cycle_quality_subjects.json"
    if not path.exists():
        return None
    with open(path) as f:
        raw = json.load(f)
    return {
        seg: {str(s) for s in ids}
        for seg, ids in raw.items()
        if not seg.startswith("_")
    }


def _sms_locus_median_pct(sig, baseline_window, window_sec=4.0):
    """SMS trace: locus median + sliding median filter, % change from baseline."""
    x = np.asarray(sig.x, float).copy()
    x[~np.asarray(sig.m, bool)] = np.nan
    y_abs = np.nanmedian(x, axis=1)
    fs = float(sig.protocol.fs)
    win = max(1, int(round(window_sec * fs)))
    T = y_abs.size
    y_smooth = np.full(T, np.nan, float)
    for i in range(T):
        lo = max(0, i - win // 2)
        hi = min(T, i + win // 2 + 1)
        y_smooth[i] = np.nanmedian(y_abs[lo:hi])
    bl_idx = np.where(baseline_window.contains(np.asarray(sig.t, float)))[0]
    b0 = float(np.nanmedian(y_smooth[bl_idx]))
    if not np.isfinite(b0) or b0 == 0:
        return y_smooth
    return 100.0 * (y_smooth - b0) / b0


def _md_per_cycle(trace_1d, t_abs):
    """Baseline-corrected MD for each of the three cycles. Returns (md0, md1, md2) or None."""
    out = []
    for ci, fl_start in FLICKER_ONSETS.items():
        bl_start, bl_end = CYCLE_BASELINES[ci]
        fl_end = fl_start + FLICKER_DUR
        bl_idx = np.where((t_abs >= bl_start) & (t_abs < bl_end))[0]
        bl_vals = trace_1d[bl_idx]
        bl_vals = bl_vals[np.isfinite(bl_vals)]
        if len(bl_vals) == 0:
            return None
        bd = float(np.median(bl_vals))
        fl_idx = np.where((t_abs >= fl_start) & (t_abs < fl_end + 10.0))[0]
        if len(fl_idx) == 0:
            return None
        seg = trace_1d[fl_idx]
        valid = np.isfinite(seg)
        if valid.sum() < 5:
            return None
        out.append(float(np.nanmax(seg) - bd))
    return tuple(out)


def extract_empirical_triplets(data_dir, retained_subjects, out_dir, *, n_segments=None):
    """Run SMS on retained clinical segments, normalise per-cycle MD triplets so
    each has mean 1, write to CSV. Returns the DataFrame."""
    print(f"Stage 1: extracting empirical SMS-derived triplets from {data_dir}")
    ds = ImedosReader(config=DataReaderConfig(protocol=PROTOCOL)).read(data_dir)

    rows = []
    try:
        from tqdm import tqdm
        rec_iter = tqdm(list(ds.recordings), desc="Stage 1 SMS", unit="rec")
    except ImportError:
        rec_iter = ds.recordings

    for rec in rec_iter:
        for seg_label in ("A1", "V3"):
            if seg_label not in rec.segments:
                continue
            if (
                retained_subjects is not None
                and str(rec.subject_id) not in retained_subjects.get(seg_label, set())
            ):
                continue
            seg = rec.segments[seg_label]
            sig = seg.signal
            bl_window = sig.protocol.global_baseline or sig.protocol.cycles[0].baseline
            try:
                sms_trace = _sms_locus_median_pct(sig, bl_window)
            except Exception:
                continue
            triplet = _md_per_cycle(np.asarray(sms_trace), np.asarray(sig.t, float))
            if triplet is None:
                continue
            mds = np.asarray(triplet, float)
            # Floor non-positive per-cycle MDs at 5% of the mean of positive cycles
            # so that all retained segments contribute. Heavy SMS-trace noise or
            # vasomotion drift can occasionally drag a cycle's max below baseline;
            # excluding the segment entirely would discard real inter-cycle variation.
            pos = mds[mds > 0]
            if pos.size == 0:
                continue
            mds = np.maximum(mds, 0.05 * float(pos.mean()))
            mean_md = float(mds.mean())
            mults = mds / mean_md
            rows.append({
                "seg_key": f"{rec.subject_id}_{rec.visit_id}_{seg_label}",
                "vessel_type": seg.vessel_type,
                "mean_md": mean_md,
                "m0": float(mults[0]),
                "m1": float(mults[1]),
                "m2": float(mults[2]),
                "cv": float(mds.std(ddof=1) / mean_md),
            })
            if n_segments is not None and len(rows) >= n_segments:
                break
        if n_segments is not None and len(rows) >= n_segments:
            break

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "empirical_amp_triplets_sms.csv"
    df.to_csv(out_path, index=False)
    print(f"  wrote {out_path} ({len(df)} triplets)")
    return df


# ── Stage 2: build hybrids and run methods ────────────────────────────

def _scale_params(base, multiplier):
    """Return a copy of `base` with amplitude(s) scaled by `multiplier`."""
    multiplier = max(0.01, float(multiplier))
    if isinstance(base, ArterialParams):
        return ArterialParams(
            dilation_amplitude=base.dilation_amplitude * multiplier,
            constriction_depth=base.constriction_depth * multiplier,
            time_shift=base.time_shift,
        )
    if isinstance(base, VenousParams):
        return VenousParams(
            dilation_amplitude=base.dilation_amplitude * multiplier,
            time_shift=base.time_shift,
        )
    raise TypeError(f"Unknown params type: {type(base)}")


def _build_nonstationary(
    sig, vessel_type, base_params, rng, multipliers,
    locus_variability=0.0, locus_jitter_samples=2, sparse_cfg=None,
):
    """Replicate build_hybrid with custom per-cycle amplitude multipliers.

    Mirrors `dvanalysis.validation.hybrid.build_hybrid` step-for-step,
    except the per-cycle parameter list is constructed by scaling
    `base_params` by `multipliers[i]` instead of by calling
    `apply_intercycle_jitter`. All other assembly is identical and uses
    the same public helpers.
    """
    t = np.asarray(sig.t, float)
    T, P = sig.T, sig.P
    protocol = sig.protocol

    if len(multipliers) != len(protocol.cycles):
        raise ValueError(
            f"multipliers length {len(multipliers)} does not match "
            f"number of cycles {len(protocol.cycles)}"
        )

    baseline_windows = []
    if protocol.global_baseline is not None:
        baseline_windows.append(protocol.global_baseline)
    for cyc in protocol.cycles:
        if cyc.baseline is not None:
            baseline_windows.append(cyc.baseline)

    # 1. Generate ground-truth response with per-cycle multipliers
    cycle_params = [_scale_params(base_params, m) for m in multipliers]
    if vessel_type.lower().startswith("a"):
        gt_pct = arterial_template(t, protocol, base_params, cycle_params=cycle_params)
    else:
        gt_pct = venous_template(t, protocol, base_params, cycle_params=cycle_params)

    # 2. Real baseline residual + per-locus resting diameter
    residual_block = extract_baseline_residual(sig, baseline_windows)
    x_real = np.asarray(sig.x, float).copy()
    m_real = np.asarray(sig.m, bool)
    x_real[~m_real] = np.nan
    all_bl_idx = np.concatenate(
        [np.where(w.contains(t))[0] for w in baseline_windows]
    )
    baselines = np.nanmedian(x_real[all_bl_idx, :], axis=0)
    residual_tiled = tile_residual(residual_block, target_length=T, rng=rng)

    # 3. Per-locus response variation (rank-1 by default)
    if locus_variability > 0.0:
        locus_scale = 1.0 + rng.normal(0, locus_variability, size=P)
    else:
        locus_scale = np.ones(P)
    gt_pct_loci = gt_pct[:, np.newaxis] * locus_scale[np.newaxis, :]

    locus_shifts = np.zeros(P, dtype=int)
    if locus_jitter_samples > 0:
        locus_shifts = rng.integers(
            -locus_jitter_samples, locus_jitter_samples + 1, size=P
        )
        for p in range(P):
            s = int(locus_shifts[p])
            if s != 0:
                gt_pct_loci[:, p] = np.roll(gt_pct_loci[:, p], s)
                if s > 0:
                    gt_pct_loci[:s, p] = 0.0
                else:
                    gt_pct_loci[s:, p] = 0.0

    # 4. Assemble hybrid signal
    response_factor = 1.0 + gt_pct_loci / 100.0
    x_hybrid = baselines[np.newaxis, :] * response_factor + residual_tiled

    # 5. Observation mask
    residual_mask = np.isfinite(residual_tiled)
    flicker_mask_1d = build_flicker_mask(t, protocol, fs=protocol.fs)
    flicker_mask = flicker_mask_1d[:, np.newaxis] & np.ones(P, dtype=bool)[np.newaxis, :]
    m_hybrid = residual_mask & flicker_mask
    x_hybrid[~m_hybrid] = np.nan

    # 6. Ground truth (matches build_hybrid logic)
    if locus_variability > 0.0 or locus_jitter_samples > 0:
        ground_truth = np.full(T, np.nan)
        for i_t in range(T):
            valid = m_hybrid[i_t, :]
            if valid.sum() > 0:
                ground_truth[i_t] = np.nanmedian(gt_pct_loci[i_t, valid])
    else:
        ground_truth = gt_pct.copy()
        no_obs = m_hybrid.sum(axis=1) == 0
        ground_truth[no_obs] = np.nan

    # 7. Optional sparse artefacts (off by default, matches build_hybrid)
    if sparse_cfg is None:
        sparse_cfg = SparseArtefactConfig(enabled=False)
    A_true = inject_sparse_artefacts(t, P, rng, sparse_cfg)
    if sparse_cfg.enabled:
        x_hybrid[m_hybrid] += A_true[m_hybrid]

    hybrid_sig = SegmentSignal(
        t=t, x=x_hybrid, m=m_hybrid, protocol=protocol,
        units=sig.units, meta={**sig.meta, "hybrid": True},
    )

    return HybridSample(
        signal=hybrid_sig,
        ground_truth=ground_truth,
        params={
            "vessel_type": vessel_type,
            "template_params": base_params.__dict__,
            "cycle_jitter": False,
            "cycle_amp_multipliers": tuple(float(m) for m in multipliers),
            "gt_pct_clean": gt_pct.copy(),
            "locus_variability": locus_variability,
            "locus_scale": locus_scale,
            "locus_jitter_samples": locus_jitter_samples,
            "locus_shifts": locus_shifts,
            "sparse_cfg": (
                sparse_cfg.__dict__ if sparse_cfg.enabled else {"enabled": False}
            ),
            "sparse_n_nonzero": int(np.count_nonzero(A_true)),
        },
    )


def _sms_trace(sig, baseline_window, window_sec=4.0):
    """SMS for evaluation: locus median + sliding median filter, % change."""
    return _sms_locus_median_pct(sig, baseline_window, window_sec)


def _sms_avg_trace(sig, baseline_window, window_sec=4.0):
    """Cycle-averaged SMS, tiled back to all three cycle positions."""
    sms = _sms_trace(sig, baseline_window, window_sec)
    t = np.asarray(sig.t, float)
    cycles = sig.protocol.cycles
    if not cycles:
        return sms
    cycle_traces = []
    for cyc in cycles:
        bl_start = (
            cyc.baseline.start_sec if cyc.baseline is not None
            else cyc.flicker.start_sec - 30.0
        )
        cyc_end = cyc.recovery.end_sec
        idx = np.where((t >= bl_start) & (t < cyc_end))[0]
        if idx.size > 0:
            cycle_traces.append(sms[idx])
    if not cycle_traces:
        return sms
    L = min(len(c) for c in cycle_traces)
    stack = np.stack([c[:L] for c in cycle_traces], axis=0)
    avg_cycle = np.nanmedian(stack, axis=0)
    y_tiled = np.full_like(sms, np.nan, float)
    for cyc in cycles:
        bl_start = (
            cyc.baseline.start_sec if cyc.baseline is not None
            else cyc.flicker.start_sec - 30.0
        )
        cyc_end = cyc.recovery.end_sec
        idx = np.where((t >= bl_start) & (t < cyc_end))[0]
        n = min(idx.size, L)
        y_tiled[idx[:n]] = avg_cycle[:n]
    return y_tiled


def _run_methods(hs, rpca):
    sig = hs.signal
    bl_window = sig.protocol.global_baseline or sig.protocol.cycles[0].baseline
    out = {
        "sms": _sms_trace(sig, bl_window),
        "sms_avg": _sms_avg_trace(sig, bl_window),
    }
    try:
        out["rpca"] = rpca.run(sig).s_hat
    except Exception:
        out["rpca"] = np.full(sig.T, np.nan)
    return out


def run_stage2(triplets_df, *, data_dir, retained_subjects,
               n_segments, n_configs, seed, out_dir):
    """Run all three methods on stationary and non-stationary hybrid sets.

    Stationary uses ``build_hybrid`` (default 5% jitter); non-stationary
    uses ``_build_nonstationary`` with multipliers bootstrapped from the
    SMS-derived per-recording triplets.
    """
    print("Stage 2: running stationary and non-stationary conditions through "
          "SMS, SMS-avg, RPCA")
    ds = ImedosReader(config=DataReaderConfig(protocol=PROTOCOL)).read(data_dir)
    rpca = MyMethodRPCA(config=RPCA_CFG)

    pool_by_vessel = {
        v: triplets_df[triplets_df.vessel_type == v][["m0", "m1", "m2"]].to_numpy()
        for v in ("artery", "vein")
    }
    if any(arr.shape[0] == 0 for arr in pool_by_vessel.values()):
        raise RuntimeError("Empirical triplet pool empty for at least one vessel.")

    rng = np.random.default_rng(seed)
    bias_rows = []
    fid_rows = []

    seg_count = 0
    rec_iter = list(ds.recordings)
    try:
        from tqdm import tqdm
        rec_iter = tqdm(rec_iter, desc="Stage 2", unit="rec")
    except ImportError:
        pass

    for rec in rec_iter:
        for seg_label in ("A1", "V3"):
            if seg_label not in rec.segments:
                continue
            if (
                retained_subjects is not None
                and str(rec.subject_id) not in retained_subjects.get(seg_label, set())
            ):
                continue
            seg = rec.segments[seg_label]
            vessel_type = seg.vessel_type
            seg_key = f"{rec.subject_id}_{rec.visit_id}_{seg_label}"

            for cfg_idx in range(n_configs):
                if vessel_type == "artery":
                    base_params = sample_arterial_params(rng)
                else:
                    base_params = sample_venous_params(rng)

                pool = pool_by_vessel[vessel_type]
                triplet = pool[rng.integers(0, pool.shape[0])]

                # Stationary: identical cycles (cycle_jitter=False)
                hs_stat = build_hybrid(
                    seg.signal, vessel_type, base_params, rng=rng,
                    cycle_jitter=False, locus_variability=0.0,
                    locus_jitter_samples=2,
                )
                # Non-stationary: same construction but per-cycle multipliers
                hs_ns = _build_nonstationary(
                    seg.signal, vessel_type, base_params, rng,
                    multipliers=tuple(triplet),
                    locus_variability=0.0, locus_jitter_samples=2,
                )

                for condition, hs in (("stationary", hs_stat),
                                      ("non_stationary", hs_ns)):
                    recovered = _run_methods(hs, rpca)
                    for method, y_hat in recovered.items():
                        try:
                            sdr = float(compute_sdr(hs.ground_truth, y_hat))
                        except Exception:
                            sdr = float("nan")
                        try:
                            nmse = float(compute_nmse(hs.ground_truth, y_hat))
                        except Exception:
                            nmse = float("nan")
                        fid_rows.append(dict(
                            condition=condition, seg_key=seg_key,
                            vessel_type=vessel_type, config_idx=cfg_idx,
                            method=method, sdr=sdr, nmse=nmse,
                        ))

                        biases = compute_biomarker_bias(
                            hs.ground_truth, y_hat,
                            np.asarray(hs.signal.t, float),
                            hs.signal.protocol, vessel_type=vessel_type,
                        )
                        for cycle_idx, cycle_bias in enumerate(biases):
                            for bm_name, bias_val in cycle_bias.items():
                                bias_rows.append(dict(
                                    condition=condition, seg_key=seg_key,
                                    vessel_type=vessel_type, config_idx=cfg_idx,
                                    method=method, cycle_idx=cycle_idx,
                                    biomarker=bm_name, bias=bias_val,
                                    amp_m0=float(triplet[0]),
                                    amp_m1=float(triplet[1]),
                                    amp_m2=float(triplet[2]),
                                ))

            seg_count += 1
            if n_segments is not None and seg_count >= n_segments:
                break
        if n_segments is not None and seg_count >= n_segments:
            break

    out_dir.mkdir(parents=True, exist_ok=True)
    bias_df = pd.DataFrame(bias_rows)
    fid_df = pd.DataFrame(fid_rows)
    bias_df.to_csv(out_dir / "nonstationary_results.csv", index=False)
    fid_df.to_csv(out_dir / "nonstationary_fidelity.csv", index=False)
    print(f"  wrote {out_dir / 'nonstationary_results.csv'} "
          f"({len(bias_df)} rows, {seg_count} segments)")
    print(f"  wrote {out_dir / 'nonstationary_fidelity.csv'} "
          f"({len(fid_df)} rows)")
    return bias_df, fid_df


# ── Stage 3: paired statistical comparisons (5 comparisons) ───────────

def _paired_diff_stats(x, y, *, n_boot=5000, seed=0):
    """Median per-sample paired difference, 95% bootstrap CI, paired Wilcoxon p."""
    from scipy.stats import wilcoxon
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    diff = x[valid] - y[valid]
    if diff.size < 5:
        return dict(n=int(diff.size), median=float("nan"),
                    ci_low=float("nan"), ci_high=float("nan"),
                    p_wilcoxon=float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    boot_med = np.median(diff[idx], axis=1)
    try:
        _, p = wilcoxon(diff)
    except ValueError:
        p = float("nan")
    return dict(
        n=int(diff.size),
        median=float(np.median(diff)),
        ci_low=float(np.percentile(boot_med, 2.5)),
        ci_high=float(np.percentile(boot_med, 97.5)),
        p_wilcoxon=float(p),
    )


def _holm_correct(pvals):
    """Holm-Bonferroni step-down adjusted p-values, in input order."""
    p = np.asarray(pvals, float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m, float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


def _paired_cluster_stats(pivot, col_a, col_b, *, n_boot=5000, seed=0, absolute=False):
    """Subject-level cluster bootstrap + Wilcoxon on per-subject median diffs.

    `pivot` is indexed by (seg_key, config_idx[, cycle_idx]); within a vessel
    seg_key is one segment per subject, so it is the cluster unit. The paired
    difference col_a - col_b is reduced to one median per subject; the point
    estimate is the median across subjects; the 95% CI resamples whole subjects
    (carrying all their configs/cycles) 5000 times.
    """
    from scipy.stats import wilcoxon
    if col_a not in pivot.columns or col_b not in pivot.columns:
        return None
    a = pivot[col_a].astype(float)
    b = pivot[col_b].astype(float)
    if absolute:
        a, b = a.abs(), b.abs()
    diff = (a - b).dropna()
    if diff.empty:
        return None
    subj = diff.index.get_level_values("seg_key")
    vals = diff.groupby(subj).median().to_numpy()   # one per-subject median diff
    n_subj = vals.size
    if n_subj < 3:
        return dict(n_subjects=int(n_subj), n_samples=int(diff.size),
                    median=float("nan"), ci_low=float("nan"),
                    ci_high=float("nan"), p_wilcoxon=float("nan"))
    rng = np.random.default_rng(seed)
    # Resample whole subjects with replacement. A drawn subject carries all its
    # pooled rows, so its per-subject median is unchanged by the draw; resampling
    # the per-subject medians is therefore exactly equivalent and vectorisable.
    boot = np.median(vals[rng.integers(0, n_subj, size=(n_boot, n_subj))], axis=1)
    try:
        _, p = wilcoxon(vals)
    except ValueError:
        p = float("nan")
    return dict(
        n_subjects=int(n_subj),
        n_samples=int(diff.size),
        median=float(np.median(vals)),
        ci_low=float(np.percentile(boot, 2.5)),
        ci_high=float(np.percentile(boot, 97.5)),
        p_wilcoxon=float(p),
    )


# Five comparisons specified in the briefing.
_AMPLITUDE_BIOMARKERS = ("MD", "MC", "DA")


def compute_paired_stats(bias_df, fid_df, out_dir, *, n_boot=5000, seed=0):
    print("Stage 3: paired statistical comparisons (Wilcoxon + bootstrap CI)")
    rows = []

    # (a)–(d): RPCA vs SMS-avg and RPCA vs SMS, within each condition.
    within_pairs = [("rpca", "sms_avg"), ("rpca", "sms")]
    for cond in ("stationary", "non_stationary"):
        sub_fid = fid_df[fid_df.condition == cond]
        for vessel in ("artery", "vein"):
            sub_fv = sub_fid[sub_fid.vessel_type == vessel]
            for metric in ("sdr", "nmse"):
                pivot = sub_fv.pivot_table(
                    index=["seg_key", "config_idx"],
                    columns="method", values=metric,
                )
                for m1, m2 in within_pairs:
                    s = _paired_cluster_stats(pivot, m1, m2, n_boot=n_boot, seed=seed)
                    if s is None:
                        continue
                    rows.append(dict(
                        comparison=f"{m1} vs {m2} ({cond})",
                        condition=cond, vessel_type=vessel,
                        metric=metric.upper(),
                        **s,
                    ))

        sub_bias = bias_df[bias_df.condition == cond]
        for vessel in ("artery", "vein"):
            sub_bv = sub_bias[sub_bias.vessel_type == vessel]
            for biomarker in sub_bv["biomarker"].unique():
                bsub = sub_bv[sub_bv.biomarker == biomarker]
                pivot = bsub.pivot_table(
                    index=["seg_key", "config_idx", "cycle_idx"],
                    columns="method", values="bias",
                )
                for m1, m2 in within_pairs:
                    s = _paired_cluster_stats(pivot, m1, m2, n_boot=n_boot, seed=seed)
                    if s is not None:
                        rows.append(dict(
                            comparison=f"{m1} vs {m2} ({cond})",
                            condition=cond, vessel_type=vessel,
                            metric=f"bias_{biomarker}",
                            **s,
                        ))
                    if biomarker in _AMPLITUDE_BIOMARKERS:
                        sa = _paired_cluster_stats(
                            pivot, m1, m2, n_boot=n_boot, seed=seed, absolute=True,
                        )
                        if sa is not None:
                            rows.append(dict(
                                comparison=f"{m1} vs {m2} ({cond})",
                                condition=cond, vessel_type=vessel,
                                metric=f"abs_bias_{biomarker}",
                                **sa,
                            ))

    # (e): SMS-avg stationary vs non-stationary, on amplitude biomarkers.
    for vessel in ("artery", "vein"):
        sub = bias_df[
            (bias_df.method == "sms_avg")
            & (bias_df.vessel_type == vessel)
            & (bias_df.biomarker.isin(_AMPLITUDE_BIOMARKERS))
        ]
        for biomarker in sub["biomarker"].unique():
            bsub = sub[sub.biomarker == biomarker]
            pivot = bsub.pivot_table(
                index=["seg_key", "config_idx", "cycle_idx"],
                columns="condition", values="bias",
            )
            s = _paired_cluster_stats(
                pivot, "stationary", "non_stationary", n_boot=n_boot, seed=seed,
            )
            if s is None:
                continue
            rows.append(dict(
                comparison="sms_avg stationary vs non-stationary",
                condition="cross",
                vessel_type=vessel,
                metric=f"bias_{biomarker}",
                **s,
            ))
            s_abs = _paired_cluster_stats(
                pivot, "stationary", "non_stationary",
                n_boot=n_boot, seed=seed, absolute=True,
            )
            rows.append(dict(
                comparison="sms_avg stationary vs non-stationary (|bias|)",
                condition="cross",
                vessel_type=vessel,
                metric=f"abs_bias_{biomarker}",
                **s_abs,
            ))

    df = pd.DataFrame(rows)

    # Holm correction over the primary family: RPCA vs SMS-avg, non-stationary,
    # |bias| on MD/MC/DA (artery) and MD (vein).
    primary = (
        (df.comparison == "rpca vs sms_avg (non_stationary)")
        & (
            ((df.vessel_type == "artery")
             & df.metric.isin(["abs_bias_MD", "abs_bias_MC", "abs_bias_DA"]))
            | ((df.vessel_type == "vein") & (df.metric == "abs_bias_MD"))
        )
    )
    df["primary_family"] = primary
    df["p_holm"] = np.nan
    if primary.any():
        df.loc[primary, "p_holm"] = _holm_correct(
            df.loc[primary, "p_wilcoxon"].to_numpy()
        )

    out_path = out_dir / "nonstationary_paired_stats.csv"
    df.to_csv(out_path, index=False)
    print(f"  wrote {out_path} ({len(df)} pairwise comparisons, "
          f"{int(primary.sum())} in primary family)")
    return df


# ── Stage 4: Table 5 + figures ────────────────────────────────────────

METHOD_ORDER = ["sms", "sms_avg", "rpca"]
METHOD_LABEL = {"sms": "SMS", "sms_avg": "SMS-avg", "rpca": "RPCA"}
# Vessel-coloured palettes: light → dark = SMS → SMS-avg → RPCA, so that the
# panel's vessel colour (artery red #B2182B, vein blue #2166AC, used throughout
# the manuscript figures) is the dominant visual cue and methods are
# distinguished by saturation within the panel.
VESSEL_METHOD_COLOR = {
    "artery": {"sms": "#FCBBA1", "sms_avg": "#FB6A4A", "rpca": "#B2182B"},
    "vein":   {"sms": "#C6DBEF", "sms_avg": "#6BAED6", "rpca": "#2166AC"},
}
VESSEL_HISTO_COLOR = {"artery": "#B2182B", "vein": "#2166AC"}
CONDITION_ORDER = ["stationary", "non_stationary"]
CONDITION_LABEL = {"stationary": "Stationary", "non_stationary": "Non-stationary"}


def _summarise_md(bias_df):
    md = bias_df[bias_df.biomarker == "MD"].copy()
    md["abs_bias"] = md["bias"].abs()
    out = (
        md.groupby(["vessel_type", "condition", "method"])
        .agg(median=("bias", "median"),
             q25=("bias", lambda s: float(np.nanpercentile(s, 25))),
             q75=("bias", lambda s: float(np.nanpercentile(s, 75))),
             median_abs=("abs_bias", "median"),
             n=("bias", "count"))
        .reset_index()
    )
    return out


def _build_table5(bias_df):
    """Construct Table 5 content: signed bias and median |bias| for amplitude
    biomarkers, both conditions, all three methods."""
    bm_amp = ("MD", "MC", "DA")
    bm_timing = ("tMAD", "tMAD30")
    rows = []
    for vessel in ("artery", "vein"):
        for method in METHOD_ORDER:
            for condition in CONDITION_ORDER:
                row = dict(vessel_type=vessel, method=method, condition=condition)
                sub = bias_df[
                    (bias_df.vessel_type == vessel)
                    & (bias_df.method == method)
                    & (bias_df.condition == condition)
                ]
                for bm in bm_amp + bm_timing:
                    bsub = sub[sub.biomarker == bm]
                    signed = bsub["bias"].median() if len(bsub) else np.nan
                    row[f"{bm}_signed"] = round(float(signed), 3) if pd.notna(signed) else np.nan
                    if bm in bm_amp:
                        absmed = bsub["bias"].abs().median() if len(bsub) else np.nan
                        row[f"{bm}_abs"] = round(float(absmed), 3) if pd.notna(absmed) else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def _draw_md_panel(ax, summary, *, vessel_type, title):
    sub = summary[summary.vessel_type == vessel_type]
    n_methods = len(METHOD_ORDER)
    bar_width = 0.24
    x_centres = np.arange(len(CONDITION_ORDER))
    palette = VESSEL_METHOD_COLOR[vessel_type]

    handles = []
    for j, method in enumerate(METHOD_ORDER):
        offsets = (j - (n_methods - 1) / 2.0) * bar_width
        med, err_lo, err_hi = [], [], []
        for cond in CONDITION_ORDER:
            row = sub[(sub.condition == cond) & (sub.method == method)]
            if row.empty:
                med.append(np.nan); err_lo.append(0); err_hi.append(0)
            else:
                m = float(row["median"].iloc[0])
                lo = float(row["q25"].iloc[0])
                hi = float(row["q75"].iloc[0])
                med.append(m)
                err_lo.append(max(0, m - lo))
                err_hi.append(max(0, hi - m))
        ax.bar(x_centres + offsets, med, width=bar_width * 0.94,
               color=palette[method], edgecolor="white", linewidth=0.4,
               yerr=[err_lo, err_hi], capsize=2,
               error_kw=dict(lw=0.6, ecolor="#666666"),
               zorder=3)
        handles.append(Patch(facecolor=palette[method], edgecolor="white",
                             linewidth=0.4, label=METHOD_LABEL[method]))

    ax.axhline(0, color="#cccccc", linewidth=0.5, linestyle=":", zorder=1)
    ax.set_xticks(x_centres)
    ax.set_xticklabels([CONDITION_LABEL[c] for c in CONDITION_ORDER])
    ax.set_ylabel("MD bias (% change)", fontsize=8)
    ax.set_title(title, loc="left", fontsize=10)
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(handles=handles, loc="upper left", fontsize=8,
              framealpha=0.95, edgecolor="#dddddd", fancybox=False)


def make_figure(bias_df, triplets_df, out_dir):
    print("Stage 4: rendering figures and Table 5 content")
    summary = _summarise_md(bias_df)

    plt.rcParams.update({
        "font.size": 9, "figure.dpi": 200,
        "savefig.dpi": 200, "savefig.bbox": "tight",
    })

    # Body figure: arterial + venous MD panels, all three methods.
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))
    _draw_md_panel(axes[0], summary, vessel_type="artery",
                   title=r"$\bf{A}$  Artery")
    _draw_md_panel(axes[1], summary, vessel_type="vein",
                   title=r"$\bf{B}$  Vein")
    # Share y-range for visual comparison
    y0, y1 = zip(*[ax.get_ylim() for ax in axes])
    for ax in axes:
        ax.set_ylim(min(y0), max(y1))
    fig.tight_layout(w_pad=1.5)

    body_png = out_dir / "fig_nonstationary_hybrid.png"
    body_pdf = out_dir / "fig_nonstationary_hybrid.pdf"
    fig.savefig(body_png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(body_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {body_png}")
    print(f"  wrote {body_pdf}")

    # Appendix C figure: empirical CV histogram (SMS-derived).
    figA, ax = plt.subplots(figsize=(5.0, 3.2))
    bins = np.linspace(0, 0.8, 25)
    for vessel, color in VESSEL_HISTO_COLOR.items():
        sub = triplets_df[triplets_df.vessel_type == vessel]
        if sub.empty:
            continue
        ax.hist(sub["cv"], bins=bins, color=color, alpha=0.65,
                edgecolor=color, linewidth=0.9,
                label=f"{vessel.capitalize()} (median = {sub.cv.median():.2f})",
                zorder=3)
    ax.set_xlabel("Within-recording amplitude CV (SMS-derived)", fontsize=8)
    ax.set_ylabel("Recordings", fontsize=8)
    ax.set_title("Empirical inter-cycle amplitude variation",
                 loc="left", fontsize=10)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=8, loc="upper right",
              framealpha=0.95, edgecolor="#dddddd", fancybox=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    figA.tight_layout()
    app_png = out_dir / "fig_appendix_empirical_cv.png"
    app_pdf = out_dir / "fig_appendix_empirical_cv.pdf"
    figA.savefig(app_png, dpi=200, bbox_inches="tight", facecolor="white")
    figA.savefig(app_pdf, bbox_inches="tight", facecolor="white")
    plt.close(figA)
    print(f"  wrote {app_png}")
    print(f"  wrote {app_pdf}")

    # Table 5 content
    table5 = _build_table5(bias_df)
    table5_path = out_dir / "table5_biomarker_bias.csv"
    table5.to_csv(table5_path, index=False)
    print(f"  wrote {table5_path}")

    return summary


# ── Main ──────────────────────────────────────────────────────────────

def main():
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    retained = _load_subjects(args.subjects_json)

    if args.clinical_triplets_from is not None:
        triplets_df = pd.read_csv(args.clinical_triplets_from)
        print(f"Loaded {len(triplets_df)} pre-computed triplets from "
              f"{args.clinical_triplets_from}")
    else:
        triplets_df = extract_empirical_triplets(
            args.data_dir, retained, args.out_dir,
            n_segments=args.n_segments,
        )

    cv_summary = (
        triplets_df.groupby("vessel_type")["cv"]
        .agg(median="median", mean="mean", n="count")
        .reset_index()
    )
    print("Empirical within-recording amplitude CV (SMS-derived):")
    print(cv_summary.to_string(index=False))
    cv_summary.to_csv(args.out_dir / "empirical_cv_summary.csv", index=False)

    bias_df, fid_df = run_stage2(
        triplets_df,
        data_dir=args.data_dir,
        retained_subjects=retained,
        n_segments=args.n_segments,
        n_configs=args.n_configs,
        seed=args.seed,
        out_dir=args.out_dir,
    )

    compute_paired_stats(
        bias_df, fid_df, args.out_dir,
        n_boot=args.n_boot, seed=args.paired_stats_seed,
    )
    make_figure(bias_df, triplets_df, args.out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
