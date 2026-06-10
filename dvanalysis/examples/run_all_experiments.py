"""
Run ALL experiments for the manuscript: hybrid evaluation + appendix analyses.

This is the single script to run on the server. It:
  1. Loads data once
  2. Runs the main hybrid evaluation (results tables)
  3. Regenerates the corrected summary (split by vessel type, tMAD30 included)
  4. Runs all four appendix analyses (templates, noise, drift, rank-1)

Usage:
    conda activate dvanalysis
    python dvanalysis/examples/run_all_experiments.py

    # Or selective:
    python dvanalysis/examples/run_all_experiments.py --skip-hybrid     # appendix only
    python dvanalysis/examples/run_all_experiments.py --skip-appendix   # hybrid only

Output:
    dvanalysis/examples/results/           (hybrid evaluation)
    dvanalysis/examples/appendix_output/   (appendix analyses)

Estimated runtime: ~4-6 hours on 4 CPUs with 32GB RAM.
"""
import warnings
warnings.filterwarnings("ignore")

import argparse
import re
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from dvanalysis.domain import (
    Segment,
    TimeWindow,
    StimulusCycle,
    StimulusProtocol,
)
from dvanalysis.io import ImedosReader, DataReaderConfig
from dvanalysis.validation.runner import run_hybrid_evaluation

# ── Protocol ─────────────────────────────────────────────────────────────
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
        for i, (b, f, r, e) in enumerate([
            (20, 50, 70, 120),
            (120, 150, 170, 220),
            (220, 250, 270, 320),
        ])
    ],
)

# ── Paths ────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "Healthy_Volunteers"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
APPENDIX_DIR = Path(__file__).resolve().parent / "appendix_output"

# ── Configuration ────────────────────────────────────────────────────────
N_CONFIGS = 10
SEED = 42
RPCA_MAX_ITER = 70
RPCA_LMB = 0.55
RPCA_GAMMA = 1000.0
SEGMENT_LABELS = {"A1", "V3"}


# ═══════════════════════════════════════════════════════════════════════════
# Data loading (shared across all experiments)
# ═══════════════════════════════════════════════════════════════════════════

def load_segments() -> Dict[str, Segment]:
    """Load DVA recordings and extract A1/V3 segments."""
    print("Loading data...")
    ds = ImedosReader(config=DataReaderConfig(protocol=PROTOCOL)).read(DATA_DIR)
    print(f"Loaded {len(ds.recordings)} recordings")

    segments = {}
    for rec in ds.recordings:
        for seg_label, seg in rec.segments.items():
            if seg_label in SEGMENT_LABELS:
                key = f"{rec.subject_id}_{rec.visit_id}_{seg_label}"
                segments[key] = seg

    n_art = sum(1 for s in segments.values() if s.vessel_type == "artery")
    n_ven = sum(1 for s in segments.values() if s.vessel_type == "vein")
    print(f"Segments: {len(segments)} total ({n_art} artery, {n_ven} vein)")
    return segments


# ═══════════════════════════════════════════════════════════════════════════
# Helper: correct summary with vessel-type split and exact column matching
# ═══════════════════════════════════════════════════════════════════════════

def _get_bias_vals(sub, bias_cols, bio_name):
    """Get flattened finite bias values for a specific biomarker."""
    pattern = re.compile(rf"^bias_{re.escape(bio_name)}_c\d+$")
    cols = [c for c in bias_cols if pattern.match(c)]
    if not cols:
        return np.array([])
    vals = sub[cols].values.flatten()
    return vals[np.isfinite(vals)]


def generate_summary(df: pd.DataFrame, out_dir: Path) -> None:
    """Generate the corrected summary split by vessel type."""
    bias_cols = [c for c in df.columns if c.startswith("bias_")]
    bio_names = sorted(set(
        re.match(r"bias_(.+)_c\d+$", c).group(1)
        for c in bias_cols
        if re.match(r"bias_(.+)_c\d+$", c)
    ))

    ARTERY_ONLY = {"MC", "DA"}

    # Print summary
    print("\n" + "=" * 80)
    print("SIGNAL-LEVEL METRICS (median [IQR])")
    print("=" * 80)

    for method in sorted(df["method"].unique()):
        sub = df[df["method"] == method]
        sdr = sub["SDR"]
        nmse = sub["NMSE"]
        dsdr = sub["delta_SDR"]
        print(f"\n{method} (n={len(sub)}):")
        print(f"  SDR:   {sdr.median():.1f} [{sdr.quantile(0.25):.1f}, {sdr.quantile(0.75):.1f}] dB")
        print(f"  NMSE:  {nmse.median():.4f} [{nmse.quantile(0.25):.4f}, {nmse.quantile(0.75):.4f}]")
        print(f"  ΔSDR:  {dsdr.median():.1f} [{dsdr.quantile(0.25):.1f}, {dsdr.quantile(0.75):.1f}] dB")

    print("\n" + "=" * 80)
    print("BIOMARKER BIAS BY VESSEL TYPE (median [IQR])")
    print("=" * 80)

    for vtype in ["artery", "vein"]:
        print(f"\n--- {vtype.upper()} ---")
        df_vt = df[df["vessel_type"] == vtype]
        for method in sorted(df_vt["method"].unique()):
            sub = df_vt[df_vt["method"] == method]
            print(f"\n  {method} (n={len(sub)}):")
            for bio in bio_names:
                if vtype == "vein" and bio in ARTERY_ONLY:
                    continue
                vals = _get_bias_vals(sub, bias_cols, bio)
                if len(vals) > 0:
                    print(f"    {bio:8s}: {np.median(vals):+.3f} [{np.percentile(vals, 25):+.3f}, {np.percentile(vals, 75):+.3f}]")

    print("\n" + "=" * 80)
    print("PLATEAU STABILITY BY VESSEL TYPE (fraction |tMAD error| > 0.5s)")
    print("=" * 80)

    for vtype in ["artery", "vein"]:
        print(f"\n  {vtype}:")
        df_vt = df[df["vessel_type"] == vtype]
        for method in sorted(df_vt["method"].unique()):
            sub = df_vt[df_vt["method"] == method]
            ps = sub["plateau_stability"]
            ps_valid = ps[np.isfinite(ps)]
            if len(ps_valid) > 0:
                print(f"    {method}: {ps_valid.mean():.3f}")

    # Save CSV
    summary_rows = []
    for vtype in ["all", "artery", "vein"]:
        df_vt = df if vtype == "all" else df[df["vessel_type"] == vtype]
        for method in sorted(df_vt["method"].unique()):
            sub = df_vt[df_vt["method"] == method]
            row = {
                "vessel_type": vtype,
                "method": method,
                "n_samples": len(sub),
                "SDR_median": sub["SDR"].median(),
                "SDR_q25": sub["SDR"].quantile(0.25),
                "SDR_q75": sub["SDR"].quantile(0.75),
                "NMSE_median": sub["NMSE"].median(),
                "NMSE_q25": sub["NMSE"].quantile(0.25),
                "NMSE_q75": sub["NMSE"].quantile(0.75),
                "delta_SDR_median": sub["delta_SDR"].median(),
                "delta_SDR_q25": sub["delta_SDR"].quantile(0.25),
                "delta_SDR_q75": sub["delta_SDR"].quantile(0.75),
            }
            for bio in bio_names:
                if vtype == "vein" and bio in ARTERY_ONLY:
                    continue
                vals = _get_bias_vals(sub, bias_cols, bio)
                if len(vals) > 0:
                    row[f"bias_{bio}_median"] = np.median(vals)
                    row[f"bias_{bio}_q25"] = np.percentile(vals, 25)
                    row[f"bias_{bio}_q75"] = np.percentile(vals, 75)

            ps = sub["plateau_stability"]
            ps_valid = ps[np.isfinite(ps)]
            if len(ps_valid) > 0:
                row["plateau_stability_mean"] = ps_valid.mean()

            summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "hybrid_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"\nSummary saved: {summary_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Part 1: Main hybrid evaluation
# ═══════════════════════════════════════════════════════════════════════════

def run_hybrid(segments: Dict[str, Segment]) -> None:
    """Run the main hybrid evaluation."""
    print("\n" + "=" * 80)
    print("PART 1: HYBRID EVALUATION")
    print("=" * 80)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    n_total = len(segments) * N_CONFIGS
    print(f"Segments: {len(segments)}, configs/segment: {N_CONFIGS}, total: {n_total}")
    print(f"Methods: rpca, rpca_gamma0, sms, sms_avg, sms_avg_fair")
    print(f"RPCA: lmb={RPCA_LMB}, gamma={RPCA_GAMMA}, max_iter={RPCA_MAX_ITER}")
    print()

    t0 = time.perf_counter()

    df = run_hybrid_evaluation(
        segments=segments,
        n_configs=N_CONFIGS,
        seed=SEED,
        rpca_max_iter=RPCA_MAX_ITER,
        rpca_lmb=RPCA_LMB,
        rpca_gamma=RPCA_GAMMA,
    )

    elapsed = time.perf_counter() - t0
    print(f"\nHybrid evaluation done in {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"Total rows: {len(df)}")

    # Save full results
    out_path = RESULTS_DIR / "hybrid_results.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    # Generate corrected summary
    generate_summary(df, RESULTS_DIR)


# ═══════════════════════════════════════════════════════════════════════════
# Part 2: Appendix analyses
# ═══════════════════════════════════════════════════════════════════════════

def run_appendix(segments: Dict[str, Segment]) -> None:
    """Run all appendix analyses."""
    print("\n" + "=" * 80)
    print("PART 2: APPENDIX ANALYSES")
    print("=" * 80)

    # Import the appendix script functions
    from dvanalysis.examples.run_appendix_analyses import (
        plot_templates,
        run_noise_stationarity,
        run_vasomotion_drift,
        run_rank1_ablation,
    )

    APPENDIX_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Templates (no data needed)
    t0 = time.perf_counter()
    plot_templates(APPENDIX_DIR)
    print(f"  Templates done in {time.perf_counter() - t0:.0f}s")

    # 2. Noise stationarity
    t0 = time.perf_counter()
    run_noise_stationarity(segments, PROTOCOL, APPENDIX_DIR)
    print(f"  Noise stationarity done in {time.perf_counter() - t0:.0f}s")

    # 3. Vasomotion drift
    t0 = time.perf_counter()
    run_vasomotion_drift(segments, PROTOCOL, APPENDIX_DIR)
    print(f"  Vasomotion drift done in {time.perf_counter() - t0:.0f}s")

    # 4. Rank-1 ablation
    t0 = time.perf_counter()
    run_rank1_ablation(segments, PROTOCOL, APPENDIX_DIR)
    print(f"  Rank-1 ablation done in {time.perf_counter() - t0:.0f}s")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all DVA manuscript experiments")
    parser.add_argument("--skip-hybrid", action="store_true",
                        help="Skip the main hybrid evaluation")
    parser.add_argument("--skip-appendix", action="store_true",
                        help="Skip the appendix analyses")
    parser.add_argument("--summary-only", action="store_true",
                        help="Only regenerate summary from existing hybrid_results.csv")
    args = parser.parse_args()

    t_start = time.perf_counter()

    if args.summary_only:
        # Just regenerate summary from existing results
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = RESULTS_DIR / "hybrid_results.csv"
        if not csv_path.exists():
            print(f"ERROR: {csv_path} not found. Run hybrid evaluation first.")
            exit(1)
        print(f"Loading existing results from {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows")
        generate_summary(df, RESULTS_DIR)
    else:
        # Load data once
        segments = load_segments()

        if not args.skip_hybrid:
            run_hybrid(segments)

        if not args.skip_appendix:
            run_appendix(segments)

    total = time.perf_counter() - t_start
    print(f"\n{'=' * 80}")
    print(f"ALL DONE in {total:.0f}s ({total / 60:.1f} min, {total / 3600:.1f} h)")
    print(f"{'=' * 80}")
