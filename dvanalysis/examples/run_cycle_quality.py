"""
Generate cycle quality figure and rank distribution CSV.

Runs RPCA on all retained segments, extracts per-cycle MD,
ranks cycles (highest/middle/lowest MD), and produces:
  - fig_cycle_quality_artery.png
  - fig_cycle_quality_vein.png
  - cycle_rank_distribution.csv

Usage:
    python dvanalysis/examples/run_cycle_quality.py \
        --data-dir ./data \
        --out-dir ./output/cycle_quality \
        [--pickle-dir PATH]   # skip RPCA, load pre-computed pickles instead
"""
import argparse
import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from dvanalysis.domain import TimeWindow, StimulusCycle, StimulusProtocol
from dvanalysis.io import ImedosReader, DataReaderConfig
from dvanalysis.preprocessing.rpca_denoise import MyMethodRPCA, MyMethodConfig


def _parse_args():
    p = argparse.ArgumentParser(
        description="Generate cycle quality figure and rank distribution CSV.",
    )
    p.add_argument(
        "--data-dir", type=Path, default=Path("./data"),
        help="Directory containing the DVA recordings (default: ./data).",
    )
    p.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output directory for figures and CSV "
             "(default: dvanalysis/examples/figures next to this script).",
    )
    p.add_argument(
        "--pickle-dir", type=Path, default=None,
        help="Optional directory of pre-computed RPCA pickles "
             "(*_rpca.pkl.gz). If provided, RPCA is skipped.",
    )
    p.add_argument(
        "--subjects-json", type=Path, default=None,
        help="Path to a JSON file mapping segment labels to subject ID lists "
             "(see cycle_quality_subjects.example.json). Defaults to "
             "cycle_quality_subjects.json next to this script if it exists; "
             "if no file is found, all subjects are kept.",
    )
    return p.parse_args()


_args = _parse_args()

# ── Paths ─────────────────────────────────────────────────────────────
DATA_DIR = _args.data_dir
OUT_DIR = (
    _args.out_dir
    if _args.out_dir is not None
    else Path(__file__).resolve().parent / "figures"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

PICKLE_DIR = _args.pickle_dir
USE_PICKLES = PICKLE_DIR is not None and PICKLE_DIR.exists()

# ── Protocol ──────────────────────────────────────────────────────────
PROTOCOL = StimulusProtocol(
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

FLICKER_ONSETS = {0: 50.0, 1: 150.0, 2: 250.0}
FLICKER_DUR = 20.0
# Per-cycle baseline windows (same as protocol definition)
CYCLE_BASELINES = {0: (20.0, 50.0), 1: (120.0, 150.0), 2: (220.0, 250.0)}
PRE_SEC = 15
POST_SEC = 50
FS = 25.0
T_COMMON = np.arange(-PRE_SEC, POST_SEC, 1.0 / FS)
Y_LIM = (-3.5, 5.5)

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

RANK_STYLE = {
    "artery": {
        "highest": dict(color="#B2182B", ls="-",  label="Highest MD"),
        "middle":  dict(color="#E08080", ls="--", label="Middle MD"),
        "lowest":  dict(color="#D4B5B5", ls=":",  label="Lowest MD"),
    },
    "vein": {
        "highest": dict(color="#2166AC", ls="-",  label="Highest MD"),
        "middle":  dict(color="#67A9CF", ls="--", label="Middle MD"),
        "lowest":  dict(color="#B0C4DE", ls=":",  label="Lowest MD"),
    },
}

PANEL_LETTER = {"artery": "A", "vein": "B"}

def _load_retained_subjects(path):
    """Load {segment_label: set(subject_id)} from a JSON file.

    Returns None if no file is found, in which case all subjects are kept.
    """
    if path is None:
        path = Path(__file__).resolve().parent / "cycle_quality_subjects.json"
    if not path.exists():
        return None
    with open(path) as f:
        raw = json.load(f)
    return {
        seg: set(str(s) for s in ids)
        for seg, ids in raw.items()
        if not seg.startswith("_")
    }


RETAINED_SUBJECTS = _load_retained_subjects(_args.subjects_json)
if RETAINED_SUBJECTS is None:
    print("No subject inclusion list found; keeping all subjects.")
else:
    n_total = sum(len(v) for v in RETAINED_SUBJECTS.values())
    print(f"Loaded subject inclusion list: {n_total} entries across "
          f"{len(RETAINED_SUBJECTS)} segments.")

# ── Helper ────────────────────────────────────────────────────────────

def compute_md_bc(s_hat, t_abs, ci):
    """Compute baseline-corrected max dilation, matching calc_dilation_max_bc.

    Subtracts per-cycle baseline median from s_hat, then takes nanmax
    over the extended flicker window (flicker onset to flicker end + 10 s).
    """
    bl_start, bl_end = CYCLE_BASELINES[ci]
    fl_start = FLICKER_ONSETS[ci]
    fl_end = fl_start + FLICKER_DUR

    # Per-cycle baseline median
    bl_idx = np.where((t_abs >= bl_start) & (t_abs < bl_end))[0]
    bl_vals = s_hat[bl_idx]
    bl_vals = bl_vals[np.isfinite(bl_vals)]
    if len(bl_vals) == 0:
        return None
    bd = float(np.median(bl_vals))

    # Extended flicker window (flicker + 10 s into recovery)
    fl_idx = np.where((t_abs >= fl_start) & (t_abs < fl_end + 10.0))[0]
    if len(fl_idx) == 0:
        return None
    s_cycle = s_hat[fl_idx]
    valid = np.isfinite(s_cycle)
    if valid.sum() < 5:
        return None

    return float(np.nanmax(s_cycle) - bd)


def extract_cycle_trace(s_hat, t_abs, flicker_start):
    """Cut one cycle, baseline-normalise, interpolate, resample."""
    t_start = flicker_start - PRE_SEC
    t_end = flicker_start + POST_SEC

    mask = (t_abs >= t_start) & (t_abs < t_end)
    t_rel = t_abs[mask] - flicker_start
    y = np.asarray(s_hat[mask], dtype=float)

    bl_mask = t_rel < 0
    bl_vals = y[bl_mask]
    bl_vals = bl_vals[np.isfinite(bl_vals)]
    if len(bl_vals) == 0:
        return None
    y -= np.median(bl_vals)

    valid = np.isfinite(y)
    if valid.sum() < 3:
        return None
    if (~valid).any():
        f = interp1d(t_rel[valid], y[valid], kind="linear",
                     bounds_error=False, fill_value=np.nan)
        y = f(t_rel)

    valid2 = np.isfinite(y)
    if valid2.sum() < 3:
        return None
    f2 = interp1d(t_rel[valid2], y[valid2], kind="linear",
                  bounds_error=False, fill_value=np.nan)
    return f2(T_COMMON)

# ── Load data + extract per-cycle MD + traces ─────────────────────────

cycle_data = []

if USE_PICKLES:
    # ── Server mode: load pre-computed RPCA pickles ───────────────────
    import gzip, pickle

    print(f"Loading RPCA pickles from {PICKLE_DIR} ...")
    rpca_results = {}
    for fp in sorted(PICKLE_DIR.glob("*_rpca.pkl.gz")):
        seg_key = fp.name.replace("_rpca.pkl.gz", "")
        with gzip.open(fp, "rb") as f:
            rpca_results[seg_key] = pickle.load(f)
    print(f"  Loaded {len(rpca_results)} segments.")

    for seg_key, payload in rpca_results.items():
        # Parse seg_key: e.g. "<subject>_0_A1"
        parts = seg_key.split("_")
        if len(parts) < 3:
            continue
        subject_id = parts[0]
        seg_label = parts[-1]

        if seg_label not in ("A1", "V3"):
            continue
        if (
            RETAINED_SUBJECTS is not None
            and subject_id not in RETAINED_SUBJECTS.get(seg_label, set())
        ):
            continue

        vessel_type = "artery" if seg_label == "A1" else "vein"
        t_abs = np.asarray(payload["t"], dtype=float)
        s_hat = np.asarray(payload["s_hat"], dtype=float)

        for ci, fl_start in FLICKER_ONSETS.items():
            md = compute_md_bc(s_hat, t_abs, ci)
            if md is None:
                continue
            trace = extract_cycle_trace(s_hat, t_abs, fl_start)

            cycle_data.append({
                "seg_key": seg_key,
                "vessel_type": vessel_type,
                "cycle_index": ci,
                "md": md,
                "trace": trace,
            })

else:
    # ── Local mode: run RPCA from raw data ────────────────────────────
    print("Loading data...")
    ds = ImedosReader(config=DataReaderConfig(protocol=PROTOCOL)).read(DATA_DIR)
    print(f"Loaded {len(ds.recordings)} recordings")

    rpca = MyMethodRPCA(config=RPCA_CFG)

    try:
        from tqdm import tqdm
        rec_iter = tqdm(list(ds.recordings), desc="Processing segments", unit="rec")
    except ImportError:
        rec_iter = ds.recordings

    for rec in rec_iter:
        for seg_label in ["A1", "V3"]:
            if seg_label not in rec.segments:
                continue
            if (
                RETAINED_SUBJECTS is not None
                and str(rec.subject_id) not in RETAINED_SUBJECTS.get(seg_label, set())
            ):
                continue
            seg = rec.segments[seg_label]
            vessel_type = seg.vessel_type
            sig = seg.signal
            seg_key = f"{rec.subject_id}_{rec.visit_id}_{seg_label}"

            try:
                result = rpca.run(sig)
            except Exception:
                continue

            s_hat = result.s_hat
            t_abs = np.asarray(sig.t, float)

            for ci, fl_start in FLICKER_ONSETS.items():
                md = compute_md_bc(s_hat, t_abs, ci)
                if md is None:
                    continue
                trace = extract_cycle_trace(s_hat, t_abs, fl_start)

                cycle_data.append({
                    "seg_key": seg_key,
                    "vessel_type": vessel_type,
                    "cycle_index": ci,
                    "md": md,
                    "trace": trace,
                })

print(f"\nExtracted {len(cycle_data)} cycle entries")

# ── Rank cycles ───────────────────────────────────────────────────────
df_cycles = pd.DataFrame([
    {k: v for k, v in d.items() if k != "trace"}
    for d in cycle_data
])

ranked_rows = []
for seg_key, group in df_cycles.groupby("seg_key"):
    sorted_g = group.sort_values("md", ascending=False).reset_index(drop=True)
    labels = {0: "highest", 1: "middle", 2: "lowest"}
    for i, (_, row) in enumerate(sorted_g.iterrows()):
        ranked_rows.append({
            "seg_key": seg_key,
            "vessel_type": row["vessel_type"],
            "cycle_index": row["cycle_index"],
            "md": row["md"],
            "rank": labels.get(i, "extra"),
        })
df_ranked = pd.DataFrame(ranked_rows)

# Build trace lookup
trace_lookup = {}
for d in cycle_data:
    trace_lookup[(d["seg_key"], d["cycle_index"])] = d["trace"]

# ── Cycle rank distribution CSV ───────────────────────────────────────
rank_rows = []
for vt in ["artery", "vein"]:
    sub = df_ranked[df_ranked.vessel_type == vt]
    for rank in ["highest", "middle", "lowest"]:
        rank_sub = sub[sub["rank"] == rank]
        total = len(rank_sub)
        for ci in [0, 1, 2]:
            count = int((rank_sub["cycle_index"] == ci).sum())
            frac = count / total if total > 0 else 0
            rank_rows.append({
                "vessel_type": vt,
                "rank": rank,
                "cycle_index": ci + 1,
                "count": count,
                "total": total,
                "fraction": round(frac, 3),
            })

df_rank_dist = pd.DataFrame(rank_rows)
csv_path = OUT_DIR / "cycle_rank_distribution.csv"
df_rank_dist.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}")
print(df_rank_dist.to_string(index=False))

# ── Collect traces per vessel type x rank ─────────────────────────────
traces = {}  # (vessel_type, rank) -> list of traces

for _, row in df_ranked.iterrows():
    key = (row["vessel_type"], row["rank"])
    trace = trace_lookup.get((row["seg_key"], row["cycle_index"]))
    if trace is not None:
        traces.setdefault(key, []).append(trace)

# ── Plot ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 10, "figure.dpi": 200,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})

for vt in ["artery", "vein"]:
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.axvspan(0, FLICKER_DUR, alpha=0.08, color="#888888", zorder=0)
    ax.axhline(0, color="#cccccc", linewidth=0.5, linestyle=":", zorder=1)
    ax.axvline(0, color="#999999", linewidth=0.6, linestyle="--", zorder=2)
    ax.axvline(FLICKER_DUR, color="#999999", linewidth=0.6, linestyle="--", zorder=2)

    styles = RANK_STYLE[vt]
    for rank in ("highest", "middle", "lowest"):
        arr_list = traces.get((vt, rank), [])
        if len(arr_list) == 0:
            continue

        mat = np.vstack(arr_list)
        n = mat.shape[0]
        mean = np.nanmean(mat, axis=0)
        sem = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(n)

        style = styles[rank]
        ax.plot(T_COMMON, mean,
                color=style["color"], linestyle=style["ls"], linewidth=1.4,
                label=f'{style["label"]} (n={n})', zorder=4)
        ax.fill_between(T_COMMON, mean - sem, mean + sem,
                        color=style["color"], alpha=0.15, zorder=3)

    ax.set_xlim(-PRE_SEC, POST_SEC)
    ax.set_ylim(Y_LIM)
    ax.set_xlabel("Time relative to flicker onset (s)", fontsize=9)
    ax.set_ylabel("% change from baseline", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.95,
              edgecolor="#dddddd", fancybox=False)
    ax.set_title(f"$\\bf{{{PANEL_LETTER[vt]}}}$  {vt.capitalize()}",
                 fontsize=10, loc="left")
    ax.text(FLICKER_DUR / 2, Y_LIM[1] - (Y_LIM[1] - Y_LIM[0]) * 0.04,
            "Flicker", ha="center", va="top", fontsize=8,
            color="#888888", style="italic")

    fig.tight_layout()
    out_path = OUT_DIR / f"fig_cycle_quality_{vt}.png"
    fig.savefig(out_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

print("\nDone.")
