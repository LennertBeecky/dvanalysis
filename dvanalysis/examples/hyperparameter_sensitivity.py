"""
Hyperparameter Sensitivity Analysis for Masked Smooth RPCA
==========================================================

This script runs a grid search over (lambda, gamma) on hybrid ground-truth
data and produces:
  1. 2D heatmaps of SDR, NMSE, and timing bias vs (lambda, gamma)
  2. 1D marginal plots (lambda sweep at fixed gamma, gamma sweep at fixed lambda)
  3. A summary table of the top hyperparameter pairs

The tuning procedure used real-data proxy criteria (heartbeat suppression,
peak timing stability, amplitude preservation). This script provides an
independent sensitivity analysis on hybrid data with known ground truth.

Usage:
    conda run -n dvanalysis python dvanalysis/examples/hyperparameter_sensitivity.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from dvanalysis.domain import TimeWindow, StimulusCycle, StimulusProtocol, SegmentSignal
from dvanalysis.io import ImedosReader, DataReaderConfig
from dvanalysis.preprocessing.rpca_denoise import MyMethodRPCA, MyMethodConfig
from dvanalysis.biomarkers.library import calc_dilation_max, calc_da, calc_dilation_max_t
from dvanalysis.biomarkers.windows import idx_for_window

# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "Healthy_Volunteers"
OUT_DIR = Path(__file__).resolve().parent / "sensitivity_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Protocol
PROTOCOL = StimulusProtocol(
    name="DVA_3cycle_custom", fs=25.0,
    global_baseline=TimeWindow("baseline", 0.0, 20.0),
    cycles=[
        StimulusCycle(index=i,
            baseline=TimeWindow("baseline", float(b), float(f)),
            flicker=TimeWindow("flicker", float(f), float(r)),
            recovery=TimeWindow("recovery", float(r), float(e)))
        for i, (b, f, r, e) in enumerate([(20,50,70,120),(120,150,170,220),(220,250,270,320)])
    ],
)

# Grid: log-spaced around the chosen values
LAMBDA_GRID = np.array([0.01, 0.1, 0.3, 0.55, 1.0, 3.0])
GAMMA_GRID = np.array([0.0, 10.0, 100.0, 1000.0, 5000.0, 10000.0])

# Theoretical RPCA default for reference
# lambda_default = 1 / sqrt(max(T, P)) ~ 1/sqrt(8750) ~ 0.0107

# Number of test subjects (use a subset for speed)
N_SUBJECTS = 5

# Fixed RPCA parameters (not being swept)
FIXED_CONFIG = dict(
    fill_missing=False,
    heartbeat_filter=False,
    standardize=True,
    rpca_max_iter=70,
    rpca_tol_rel=1e-3,
    rpca_mu=1.0,
    rpca_rho=5.0,
    locus_agg="median",
    harmonize_output=True,
    harmonize_percent_mode="delta_over_baseline",
    harmonize_baseline_source="protocol_global",
    harmonize_aggregation_order="percent_then_aggregate",
    harmonize_baseline_per_locus=True,
    support_min_valid_frac=0.75,
    support_min_valid_abs=12,
    hampel_enable=True,
)


# ============================================================================
# Hybrid ground-truth generation (simplified)
# ============================================================================

def make_synthetic_response(t: np.ndarray, protocol: StimulusProtocol, seed: int = 0) -> np.ndarray:
    """Generate a synthetic percent-change stimulus response on time grid t."""
    rng = np.random.default_rng(seed)
    y = np.zeros_like(t)

    for cyc in protocol.cycles:
        f_start = cyc.flicker.start_sec
        f_end = cyc.flicker.end_sec
        r_end = cyc.recovery.end_sec

        # Dilation amplitude varies per cycle
        amp = rng.uniform(2.0, 5.0)
        rise_tau = rng.uniform(3.0, 8.0)
        decay_tau = rng.uniform(5.0, 15.0)
        constr_depth = rng.uniform(0.5, 2.0)

        for i, ti in enumerate(t):
            if f_start <= ti < f_end:
                # Rising sigmoid during flicker
                x = (ti - f_start) / rise_tau
                y[i] += amp * (1.0 / (1.0 + np.exp(-x + 2.0)))
            elif f_end <= ti < r_end:
                # Exponential decay + constriction during recovery
                dt = ti - f_end
                y[i] += amp * np.exp(-dt / decay_tau) - constr_depth * (1 - np.exp(-dt / 5.0)) * np.exp(-dt / 30.0)

    return y


def make_hybrid_sample(
    sig: SegmentSignal,
    seed: int = 0,
) -> Tuple[SegmentSignal, np.ndarray]:
    """Create a hybrid sample: synthetic response + real noise structure.

    Returns (corrupted_signal, ground_truth_trace).
    """
    rng = np.random.default_rng(seed)
    t = np.asarray(sig.t, float)
    T, P = sig.T, sig.P
    M = np.asarray(sig.m, bool)

    # Ground truth: synthetic response replicated across loci
    gt_trace = make_synthetic_response(t, sig.protocol, seed=seed)

    # Per-locus baseline offsets from real data
    b_idx = np.where(sig.protocol.global_baseline.contains(t))[0]
    x_real = np.asarray(sig.x, float).copy()
    x_real[~M] = np.nan
    baselines = np.nanmedian(x_real[b_idx, :], axis=0)  # (P,)

    # Build synthetic X: baseline + response + real residual noise
    X_clean = baselines[np.newaxis, :] * (1.0 + gt_trace[:, np.newaxis] / 100.0)

    # Residual noise from real baseline
    baseline_residual = x_real[b_idx, :] - np.nanmedian(x_real[b_idx, :], axis=0, keepdims=True)
    noise_blocks = []
    while sum(b.shape[0] for b in noise_blocks) < T:
        noise_blocks.append(baseline_residual)
    noise = np.concatenate(noise_blocks, axis=0)[:T, :]
    noise[~np.isfinite(noise)] = 0.0

    X_hybrid = X_clean + noise
    X_hybrid[~M] = np.nan

    hybrid_sig = SegmentSignal(t=t, x=X_hybrid, m=M, protocol=sig.protocol,
                               units=sig.units, meta=sig.meta)
    return hybrid_sig, gt_trace


# ============================================================================
# Evaluation metrics
# ============================================================================

def compute_sdr(gt: np.ndarray, recovered: np.ndarray) -> float:
    """Signal-to-distortion ratio in dB."""
    finite = np.isfinite(gt) & np.isfinite(recovered)
    if finite.sum() < 10:
        return float("nan")
    g = gt[finite]
    r = recovered[finite]
    sig_power = np.sum(g ** 2)
    err_power = np.sum((g - r) ** 2)
    if err_power == 0:
        return 60.0  # cap
    return float(10.0 * np.log10(sig_power / err_power))


def compute_nmse(gt: np.ndarray, recovered: np.ndarray) -> float:
    finite = np.isfinite(gt) & np.isfinite(recovered)
    if finite.sum() < 10:
        return float("nan")
    g = gt[finite]
    r = recovered[finite]
    return float(np.sum((g - r) ** 2) / np.sum(g ** 2))


def compute_timing_bias(
    gt: np.ndarray, recovered: np.ndarray, t: np.ndarray, protocol: StimulusProtocol,
) -> float:
    """Median timing bias across cycles (seconds)."""
    biases = []
    for cyc in protocol.cycles:
        w = TimeWindow("search", cyc.flicker.start_sec, cyc.flicker.end_sec + 10.0)
        try:
            t_gt = calc_dilation_max_t(gt, t, w, relative_to_start=True)
            t_rec = calc_dilation_max_t(recovered, t, w, relative_to_start=True)
            if np.isfinite(t_gt) and np.isfinite(t_rec):
                biases.append(t_rec - t_gt)
        except Exception:
            pass
    return float(np.median(biases)) if biases else float("nan")


# ============================================================================
# Main
# ============================================================================

def main():
    print("Loading data...")
    reader = ImedosReader(config=DataReaderConfig(protocol=PROTOCOL))
    ds = reader.read(DATA_DIR)

    # Select a subset of recordings
    recs = ds.recordings[:N_SUBJECTS]
    print(f"Using {len(recs)} recordings")

    # Build hybrid samples (one per segment)
    hybrids = []
    for rec in recs:
        for seg_label, seg in rec.segments.items():
            try:
                hybrid_sig, gt_trace = make_hybrid_sample(seg.signal, seed=hash(seg_label) % 2**31)
                hybrids.append((seg_label, seg.vessel_type, hybrid_sig, gt_trace))
            except Exception as e:
                print(f"  Skipped {rec.subject_id}_{seg_label}: {e}")

    print(f"Built {len(hybrids)} hybrid samples")

    # Grid search
    results = []
    total = len(LAMBDA_GRID) * len(GAMMA_GRID)
    print(f"\nRunning {total} grid points x {len(hybrids)} samples...")

    for li, lam in enumerate(LAMBDA_GRID):
        for gi, gam in enumerate(GAMMA_GRID):
            cfg = MyMethodConfig(rpca_lmb=float(lam), rpca_gamma=float(gam), **FIXED_CONFIG)
            rpca = MyMethodRPCA(config=cfg)

            sdrs, nmses, timing_biases = [], [], []

            t0 = time.perf_counter()
            for seg_label, vtype, hybrid_sig, gt_trace in hybrids:
                try:
                    result = rpca.run(hybrid_sig)
                    s_hat = result.s_hat

                    sdrs.append(compute_sdr(gt_trace, s_hat))
                    nmses.append(compute_nmse(gt_trace, s_hat))
                    timing_biases.append(compute_timing_bias(
                        gt_trace, s_hat, np.asarray(hybrid_sig.t, float), PROTOCOL))
                except Exception:
                    pass

            dt = time.perf_counter() - t0

            row = {
                "lambda": lam, "gamma": gam,
                "med_sdr": float(np.nanmedian(sdrs)),
                "med_nmse": float(np.nanmedian(nmses)),
                "med_timing_bias": float(np.nanmedian(timing_biases)),
                "abs_timing_bias": float(np.nanmedian(np.abs(timing_biases))),
                "n_valid": sum(1 for s in sdrs if np.isfinite(s)),
                "time_s": dt,
            }
            results.append(row)
            print(f"  [{li*len(GAMMA_GRID)+gi+1:3d}/{total}] "
                  f"lam={lam:.3f} gam={gam:.0f}: "
                  f"SDR={row['med_sdr']:.1f}dB, "
                  f"NMSE={row['med_nmse']:.4f}, "
                  f"timing={row['med_timing_bias']:+.2f}s "
                  f"({dt:.1f}s)")

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "grid_results.csv", index=False)
    print(f"\nResults saved to {OUT_DIR / 'grid_results.csv'}")

    # ── Plot heatmaps ──────────────────────────────────────────────────────

    # Mark the chosen operating point
    chosen_lam, chosen_gam = 0.55, 1000.0

    for metric, label, cmap, vmin, vmax, better in [
        ("med_sdr", "SDR (dB)", "RdYlGn", None, None, "high"),
        ("med_nmse", "NMSE", "RdYlGn_r", None, None, "low"),
        ("abs_timing_bias", "|Timing bias| (s)", "RdYlGn_r", 0, 5, "low"),
    ]:
        pivot = df.pivot(index="gamma", columns="lambda", values=metric)

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(
            pivot.values, aspect="auto", cmap=cmap,
            extent=[
                np.log10(LAMBDA_GRID[0]), np.log10(LAMBDA_GRID[-1]),
                np.log10(max(GAMMA_GRID[-1], 1)), np.log10(max(GAMMA_GRID[0], 0.1)),
            ],
            vmin=vmin, vmax=vmax,
        )
        plt.colorbar(im, ax=ax, label=label)

        # Mark chosen point
        ax.plot(np.log10(chosen_lam), np.log10(chosen_gam), "k*", markersize=15, zorder=10)

        # Mark theoretical default
        lam_default = 1.0 / np.sqrt(8750)
        if LAMBDA_GRID[0] <= lam_default <= LAMBDA_GRID[-1]:
            ax.axvline(np.log10(lam_default), color="white", ls="--", lw=1, alpha=0.7)
            ax.text(np.log10(lam_default), np.log10(max(GAMMA_GRID[-1], 1)) * 0.95,
                    r"$\lambda_{RPCA}$", color="white", fontsize=8, ha="center")

        ax.set_xlabel(r"$\log_{10}(\lambda)$", fontsize=11)
        ax.set_ylabel(r"$\log_{10}(\gamma)$", fontsize=11)
        ax.set_title(f"Sensitivity: {label}\n(star = chosen operating point)", fontsize=12)

        # Add value annotations
        for _, row in df.iterrows():
            x = np.log10(row["lambda"])
            y = np.log10(max(row["gamma"], 0.1))
            val = row[metric]
            if np.isfinite(val):
                txt = f"{val:.1f}" if "sdr" in metric.lower() or "bias" in metric.lower() else f"{val:.3f}"
                ax.text(x, y, txt, ha="center", va="center", fontsize=6, color="black", alpha=0.8)

        plt.tight_layout()
        fig.savefig(OUT_DIR / f"heatmap_{metric}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {OUT_DIR / f'heatmap_{metric}.png'}")

    # ── 1D marginal plots ──────────────────────────────────────────────────

    # Lambda sweep at fixed gamma
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    df_gam = df[df["gamma"] == chosen_gam].sort_values("lambda")

    for ax, metric, label in zip(axes,
        ["med_sdr", "med_nmse", "abs_timing_bias"],
        ["SDR (dB)", "NMSE", "|Timing bias| (s)"]
    ):
        ax.semilogx(df_gam["lambda"], df_gam[metric], "o-", color="#2166AC")
        ax.axvline(chosen_lam, color="#B2182B", ls="--", lw=1.5, label=f"chosen={chosen_lam}")
        ax.axvline(1/np.sqrt(8750), color="#999999", ls=":", lw=1, label=r"$1/\sqrt{T}$")
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
        ax.set_title(f"{label} vs $\\lambda$ ($\\gamma$={chosen_gam:.0f})")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "marginal_lambda.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Gamma sweep at fixed lambda
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    df_lam = df[df["lambda"] == chosen_lam].sort_values("gamma")

    for ax, metric, label in zip(axes,
        ["med_sdr", "med_nmse", "abs_timing_bias"],
        ["SDR (dB)", "NMSE", "|Timing bias| (s)"]
    ):
        gamma_vals = df_lam["gamma"].values
        gamma_plot = np.where(gamma_vals > 0, gamma_vals, 0.1)  # avoid log(0)
        ax.semilogx(gamma_plot, df_lam[metric], "o-", color="#2166AC")
        ax.axvline(chosen_gam, color="#B2182B", ls="--", lw=1.5, label=f"chosen={chosen_gam:.0f}")
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
        ax.set_title(f"{label} vs $\\gamma$ ($\\lambda$={chosen_lam})")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "marginal_gamma.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nAll plots saved to {OUT_DIR}/")

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("Top 10 hyperparameter pairs by SDR:")
    print("=" * 80)
    top = df.nlargest(10, "med_sdr")[["lambda", "gamma", "med_sdr", "med_nmse", "abs_timing_bias"]]
    print(top.to_string(index=False))

    print(f"\nChosen operating point: lambda={chosen_lam}, gamma={chosen_gam}")
    chosen_row = df[(df["lambda"] == chosen_lam) & (df["gamma"] == chosen_gam)]
    if len(chosen_row) > 0:
        r = chosen_row.iloc[0]
        print(f"  SDR={r['med_sdr']:.1f} dB, NMSE={r['med_nmse']:.4f}, "
              f"|timing bias|={r['abs_timing_bias']:.2f} s")

    print(f"\nTheoretical default lambda=1/sqrt(T)={1/np.sqrt(8750):.4f}")
    theory_row = df[df["lambda"] == LAMBDA_GRID[np.argmin(np.abs(LAMBDA_GRID - 1/np.sqrt(8750)))]]
    if len(theory_row) > 0:
        for _, r in theory_row.iterrows():
            if r["gamma"] == chosen_gam:
                print(f"  At gamma={chosen_gam}: SDR={r['med_sdr']:.1f} dB, "
                      f"|timing bias|={r['abs_timing_bias']:.2f} s")


if __name__ == "__main__":
    main()
