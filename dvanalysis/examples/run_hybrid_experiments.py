"""
Run the full hybrid evaluation on all retained segments.

Usage:
    conda run -n dvanalysis python dvanalysis/examples/run_hybrid_experiments.py

Output:
    dvanalysis/examples/results/hybrid_results.csv
    dvanalysis/examples/results/hybrid_summary.csv
"""
import warnings
warnings.filterwarnings("ignore")

import time
from pathlib import Path

import numpy as np
import pandas as pd

from dvanalysis.domain import TimeWindow, StimulusCycle, StimulusProtocol
from dvanalysis.io import ImedosReader, DataReaderConfig
from dvanalysis.preprocessing.rpca_denoise import MyMethodRPCA, MyMethodConfig
from dvanalysis.validation.runner import run_hybrid_evaluation

# ── Protocol ─────────────────────────────────────────────────────────────
protocol = StimulusProtocol(
    name="DVA_3cycle_custom", fs=25.0,
    global_baseline=TimeWindow("baseline", 0.0, 20.0),
    cycles=[
        StimulusCycle(index=i,
            baseline=TimeWindow("baseline", float(b), float(f)),
            flicker=TimeWindow("flicker", float(f), float(r)),
            recovery=TimeWindow("recovery", float(r), float(e)))
        for i, (b, f, r, e) in enumerate([
            (20, 50, 70, 120), (120, 150, 170, 220), (220, 250, 270, 320)
        ])
    ],
)

# ── Paths ────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "Healthy_Volunteers"
OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Configuration ────────────────────────────────────────────────────────
N_CONFIGS = 10          # response configurations per segment
SEED = 42
RPCA_MAX_ITER = 70
RPCA_LMB = 0.55
RPCA_GAMMA = 1000.0

# ── Load data ────────────────────────────────────────────────────────────
print("Loading data...")
ds = ImedosReader(config=DataReaderConfig(protocol=protocol)).read(DATA_DIR)
print(f"Loaded {len(ds.recordings)} recordings")

# ── Collect retained segments (A1 and V3 only) ──────────────────────────
SEGMENT_LABELS = {"A1", "V3"}

segments = {}
for rec in ds.recordings:
    for seg_label, seg in rec.segments.items():
        if seg_label in SEGMENT_LABELS:
            key = f"{rec.subject_id}_{rec.visit_id}_{seg_label}"
            segments[key] = seg

print(f"Total segments: {len(segments)}")
n_artery = sum(1 for s in segments.values() if s.vessel_type == "artery")
n_vein = sum(1 for s in segments.values() if s.vessel_type == "vein")
print(f"  Arteries: {n_artery}")
print(f"  Veins: {n_vein}")

# ── Run hybrid evaluation ────────────────────────────────────────────────
print(f"\nRunning hybrid evaluation: {N_CONFIGS} configs per segment...")
print(f"  Expected samples: {len(segments)} segments × {N_CONFIGS} configs = {len(segments) * N_CONFIGS}")
print(f"  Methods: rpca, rpca_gamma0, sms, sms_avg, sms_avg_fair")
print(f"  RPCA: lmb={RPCA_LMB}, gamma={RPCA_GAMMA}, max_iter={RPCA_MAX_ITER}")
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
print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f} min)")
print(f"Total rows: {len(df)}")

# ── Save full results ────────────────────────────────────────────────────
out_path = OUT_DIR / "hybrid_results.csv"
df.to_csv(out_path, index=False)
print(f"Saved: {out_path}")

# ── Helper: extract bias values for a specific biomarker ─────────────────
def _get_bias_vals(sub, bias_cols, bio_name):
    """Get flattened finite bias values for a specific biomarker.

    Uses exact column matching to avoid tMAD matching tMAD30.
    """
    import re
    pattern = re.compile(rf"^bias_{re.escape(bio_name)}_c\d+$")
    cols = [c for c in bias_cols if pattern.match(c)]
    if not cols:
        return np.array([])
    vals = sub[cols].values.flatten()
    return vals[np.isfinite(vals)]


# ── Summary statistics ───────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SUMMARY: Signal-level metrics (median [IQR]) — all segments")
print("=" * 80)

for method in sorted(df["method"].unique()):
    sub = df[df["method"] == method]
    sdr = sub["SDR"]
    nmse = sub["NMSE"]
    dsdr = sub["delta_SDR"]
    print(f"\n{method} (n={len(sub)}):")
    print(f"  SDR:       {sdr.median():.1f} [{sdr.quantile(0.25):.1f} – {sdr.quantile(0.75):.1f}] dB")
    print(f"  NMSE:      {nmse.median():.4f} [{nmse.quantile(0.25):.4f} – {nmse.quantile(0.75):.4f}]")
    print(f"  ΔSDR:      {dsdr.median():.1f} [{dsdr.quantile(0.25):.1f} – {dsdr.quantile(0.75):.1f}] dB")

bias_cols = [c for c in df.columns if c.startswith("bias_")]
# Extract unique biomarker names using regex to avoid tMAD/tMAD30 collision
import re
bio_names = sorted(set(
    re.match(r"bias_(.+)_c\d+$", c).group(1)
    for c in bias_cols
    if re.match(r"bias_(.+)_c\d+$", c)
))

# Biomarkers that apply to both vessel types vs arteries only
BOTH_TYPES = {"MD", "tMAD", "tMAD30"}
ARTERY_ONLY = {"MC", "DA"}

print("\n" + "=" * 80)
print("SUMMARY: Biomarker bias by vessel type (median [IQR])")
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
print("SUMMARY: Plateau stability by vessel type (fraction |tMAD error| > 0.5s)")
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

# ── Save summary (split by vessel type) ─────────────────────────────────
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
            "delta_SDR_median": sub["delta_SDR"].median(),
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
summary_path = OUT_DIR / "hybrid_summary.csv"
df_summary.to_csv(summary_path, index=False)
print(f"\nSummary saved: {summary_path}")

# ═══════════════════════════════════════════════════════════════════════════
# ATTENUATED-CYCLE ANALYSIS (main text result)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("ATTENUATED-CYCLE ANALYSIS")
print("=" * 80)

from dvanalysis.validation.hybrid import build_hybrid, inject_sparse_artefacts, SparseArtefactConfig
from dvanalysis.validation.templates import arterial_template, venous_template, apply_intercycle_jitter
from dvanalysis.validation.templates import ArterialParams, VenousParams, sample_arterial_params, sample_venous_params
from dvanalysis.validation.noise import extract_baseline_residual, tile_residual, build_flicker_mask
from dvanalysis.validation.runner import _sms_trace, _sms_avg_trace
from dvanalysis.validation.metrics import compute_sdr, compute_nmse, compute_biomarker_bias
from dvanalysis.domain import SegmentSignal

ATTENUATION_LEVELS = [1.0, 0.7, 0.5, 0.3, 0.0]
N_CONFIGS_ATTEN = 10

rpca_method = MyMethodRPCA(MyMethodConfig(
    rpca_lmb=RPCA_LMB, rpca_gamma=RPCA_GAMMA, rpca_max_iter=RPCA_MAX_ITER,
    rpca_tol_rel=1e-3, standardize=True, harmonize_output=True,
    harmonize_percent_mode="delta_over_baseline",
    harmonize_baseline_source="protocol_global",
    harmonize_aggregation_order="percent_then_aggregate",
    harmonize_baseline_per_locus=True, support_min_valid_frac=0.75,
    support_min_valid_abs=12, hampel_enable=True,
))

atten_rows = []
t0_atten = time.perf_counter()

try:
    from tqdm import tqdm
    atten_seg_iter = tqdm(segments.items(), desc="Attenuated-cycle analysis", unit="seg")
except ImportError:
    atten_seg_iter = segments.items()

for seg_key, seg in atten_seg_iter:
    vessel_type = seg.vessel_type
    sig = seg.signal
    t_sig = np.asarray(sig.t, float)
    T_sig, P_sig = sig.T, sig.P

    baseline_windows = [protocol.global_baseline] + [
        c.baseline for c in protocol.cycles if c.baseline
    ]

    for atten in ATTENUATION_LEVELS:
        for cfg_i in range(N_CONFIGS_ATTEN):
            rng = np.random.default_rng(SEED + cfg_i + int(atten * 1000))

            # Sample template params
            if vessel_type == "artery":
                base_params = sample_arterial_params(rng)
            else:
                base_params = sample_venous_params(rng)

            # Per-cycle jitter
            cycle_params = [apply_intercycle_jitter(base_params, rng) for _ in protocol.cycles]

            # Attenuate cycle 3
            cp2 = cycle_params[2]
            if vessel_type == "artery":
                cycle_params[2] = ArterialParams(
                    dilation_amplitude=cp2.dilation_amplitude * atten,
                    constriction_depth=cp2.constriction_depth * atten,
                    time_shift=cp2.time_shift,
                )
            else:
                cycle_params[2] = VenousParams(
                    dilation_amplitude=cp2.dilation_amplitude * atten,
                    time_shift=cp2.time_shift,
                )

            # Generate template
            if vessel_type == "artery":
                gt_pct = arterial_template(t_sig, protocol, base_params, cycle_params=cycle_params)
            else:
                gt_pct = venous_template(t_sig, protocol, base_params, cycle_params=cycle_params)

            # Build hybrid signal (reuse noise extraction logic)
            residual_block = extract_baseline_residual(sig, baseline_windows)
            x_real = np.asarray(sig.x, float).copy()
            m_real = np.asarray(sig.m, bool)
            x_real[~m_real] = np.nan
            all_bl_idx = np.concatenate([np.where(w.contains(t_sig))[0] for w in baseline_windows])
            baselines = np.nanmedian(x_real[all_bl_idx, :], axis=0)

            residual_tiled = tile_residual(residual_block, target_length=T_sig, rng=rng)

            gt_pct_loci = gt_pct[:, np.newaxis] * np.ones(P_sig)[np.newaxis, :]
            response_factor = 1.0 + gt_pct_loci / 100.0
            x_hybrid = baselines[np.newaxis, :] * response_factor + residual_tiled * response_factor

            residual_mask = np.isfinite(residual_tiled)
            flicker_mask_1d = build_flicker_mask(t_sig, protocol, fs=protocol.fs)
            flicker_mask = flicker_mask_1d[:, np.newaxis] & np.ones(P_sig, dtype=bool)[np.newaxis, :]
            m_hybrid = residual_mask & flicker_mask
            x_hybrid[~m_hybrid] = np.nan

            ground_truth = gt_pct.copy()
            ground_truth[m_hybrid.sum(axis=1) == 0] = np.nan

            # Sparse artefacts
            A = inject_sparse_artefacts(t_sig, P_sig, rng, SparseArtefactConfig())
            x_hybrid[m_hybrid] += A[m_hybrid]

            hybrid_sig = SegmentSignal(
                t=t_sig, x=x_hybrid, m=m_hybrid, protocol=protocol,
                units=sig.units, meta={**sig.meta, "hybrid": True},
            )

            # Run methods
            bl = protocol.global_baseline
            sms = _sms_trace(hybrid_sig, bl)
            sms_avg = _sms_avg_trace(hybrid_sig, bl)
            try:
                rpca_t = rpca_method.run(hybrid_sig).s_hat
            except Exception:
                rpca_t = np.full(T_sig, np.nan)

            # Support gate
            frac = m_hybrid.sum(axis=1) / m_hybrid.shape[1]
            low = frac < 0.75
            gt_g = ground_truth.copy(); gt_g[low] = np.nan

            # Per-cycle SDR (cycle 3 = the attenuated one)
            c3 = protocol.cycles[2]
            c3_idx = np.where((t_sig >= c3.baseline.start_sec) & (t_sig < c3.recovery.end_sec))[0]

            for method, trace in [("rpca", rpca_t), ("sms", sms), ("sms_avg", sms_avg)]:
                tr_g = trace.copy(); tr_g[low] = np.nan
                sdr_all = compute_sdr(gt_g, tr_g)
                sdr_c3 = compute_sdr(gt_g[c3_idx], tr_g[c3_idx])
                nmse_all = compute_nmse(gt_g, tr_g)

                atten_rows.append({
                    "segment_key": seg_key,
                    "vessel_type": vessel_type,
                    "attenuation": atten,
                    "config_index": cfg_i,
                    "method": method,
                    "SDR_all": sdr_all,
                    "SDR_cycle3": sdr_c3,
                    "NMSE_all": nmse_all,
                })

df_atten = pd.DataFrame(atten_rows)
atten_path = OUT_DIR / "hybrid_attenuated_cycle.csv"
df_atten.to_csv(atten_path, index=False)

elapsed_atten = time.perf_counter() - t0_atten
print(f"Done in {elapsed_atten:.0f}s ({elapsed_atten/60:.1f} min)")
print(f"Saved: {atten_path}")

# Summary
print("\n--- Overall SDR by attenuation ---")
pivot = df_atten.groupby(["attenuation", "method"])["SDR_all"].median().unstack()
print(pivot.to_string())

print("\n--- Cycle 3 SDR by attenuation ---")
pivot_c3 = df_atten.groupby(["attenuation", "method"])["SDR_cycle3"].median().unstack()
print(pivot_c3.to_string())

# Summary CSV
atten_summary_rows = []
for atten in ATTENUATION_LEVELS:
    for method in ["rpca", "sms", "sms_avg"]:
        sub = df_atten[(df_atten.attenuation == atten) & (df_atten.method == method)]
        atten_summary_rows.append({
            "attenuation": atten,
            "method": method,
            "n": len(sub),
            "SDR_all_median": sub.SDR_all.median(),
            "SDR_all_q25": sub.SDR_all.quantile(0.25),
            "SDR_all_q75": sub.SDR_all.quantile(0.75),
            "SDR_c3_median": sub.SDR_cycle3.median(),
            "SDR_c3_q25": sub.SDR_cycle3.quantile(0.25),
            "SDR_c3_q75": sub.SDR_cycle3.quantile(0.75),
            "NMSE_all_median": sub.NMSE_all.median(),
        })

df_atten_summary = pd.DataFrame(atten_summary_rows)
atten_summary_path = OUT_DIR / "hybrid_attenuated_summary.csv"
df_atten_summary.to_csv(atten_summary_path, index=False)
print(f"\nAttenuated summary saved: {atten_summary_path}")
print(f"\nAll done.")
