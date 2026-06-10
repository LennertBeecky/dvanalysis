"""
Population-level cycle overlay stratified by max_dilation quality.

Fully standalone – loads RPCA results from saved pickle files and
dva_parameters.xlsx.  No notebook or methods_by_segment dependency.

Usage:
    python dvanalysis/fig_cycle_quality.py \
        --out-dir ./output \
        --fig-dir ./output/figures
"""

import argparse
import json
from pathlib import Path
import gzip
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def _parse_args():
    p = argparse.ArgumentParser(
        description="Population-level cycle overlay stratified by max_dilation quality.",
    )
    p.add_argument(
        "--out-dir", type=Path, default=Path("./output"),
        help="Directory containing preprocessed_segment/ pickles and "
             "dva_parameters.xlsx (default: ./output).",
    )
    p.add_argument(
        "--fig-dir", type=Path, default=None,
        help="Output directory for figures (default: <out-dir>/figures).",
    )
    p.add_argument(
        "--subjects-json", type=Path, default=None,
        help="Path to a JSON file mapping segment labels to subject ID lists "
             "(see dvanalysis/examples/cycle_quality_subjects.example.json). "
             "Defaults to dvanalysis/examples/cycle_quality_subjects.json if "
             "it exists; if no file is found, all subjects are kept.",
    )
    return p.parse_args()


def _load_retained_subjects(path):
    """Load {segment_label: set(int subject_id)} from JSON, or None."""
    if path is None:
        path = Path(__file__).resolve().parent / "examples" / "cycle_quality_subjects.json"
    if not path.exists():
        return None
    with open(path) as f:
        raw = json.load(f)
    return {
        seg: {int(s) for s in ids}
        for seg, ids in raw.items()
        if not seg.startswith("_")
    }


_args = _parse_args()

# ── paths ──────────────────────────────────────────────────────────────────
OUT_DIR = _args.out_dir
PICKLE_DIR = OUT_DIR / "preprocessed_segment"
PARAM_PATH = OUT_DIR / "dva_parameters.xlsx"

FIG_DIR = _args.fig_dir if _args.fig_dir is not None else OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── protocol: flicker onset times (seconds) per cycle ──────────────────────
# From DVA_3cycle_custom: 3 cycles with flicker at 50, 150, 270 s
FLICKER_ONSETS = {0: 50.0, 1: 150.0, 2: 250.0}
FLICKER_DUR = 20.0  # seconds

# ── configuration ──────────────────────────────────────────────────────────
PRE_SEC = 15
POST_SEC = 50
FS = 25.0
T_COMMON = np.arange(-PRE_SEC, POST_SEC, 1.0 / FS)

RETAINED_SUBJECTS = _load_retained_subjects(_args.subjects_json)
if RETAINED_SUBJECTS is None:
    print("No subject inclusion list found; keeping all subjects.")
else:
    n_total = sum(len(v) for v in RETAINED_SUBJECTS.values())
    print(f"Loaded subject inclusion list: {n_total} entries across "
          f"{len(RETAINED_SUBJECTS)} segments.")

VESSEL_LABELS = {"A1": "artery", "V3": "vein"}
VESSEL_PANEL = {"A1": "Artery", "V3": "Vein"}
PANEL_LETTER = {"A1": "A", "V3": "B"}
Y_LIM = (-3.5, 5.5)

RANK_STYLE = {
    "A1": {
        "best":   dict(color="#B2182B", ls="-",  label="Highest MD"),
        "middle": dict(color="#E08080", ls="--", label="Middle MD"),
        "worst":  dict(color="#D4B5B5", ls=":",  label="Lowest MD"),
    },
    "V3": {
        "best":   dict(color="#2166AC", ls="-",  label="Highest MD"),
        "middle": dict(color="#67A9CF", ls="--", label="Middle MD"),
        "worst":  dict(color="#B0C4DE", ls=":",  label="Lowest MD"),
    },
}

# ── load pickle files ──────────────────────────────────────────────────────
def load_rpca_pickles(pickle_dir):
    """Load all *_rpca.pkl.gz files.  Returns dict seg_key -> payload."""
    results = {}
    for fp in sorted(pickle_dir.glob("*_rpca.pkl.gz")):
        seg_key = fp.name.replace("_rpca.pkl.gz", "")
        with gzip.open(fp, "rb") as f:
            results[seg_key] = pickle.load(f)
    return results

print(f"Loading RPCA pickles from {PICKLE_DIR} ...")
rpca_results = load_rpca_pickles(PICKLE_DIR)
print(f"  Loaded {len(rpca_results)} segments.")

# ── read parameter table ──────────────────────────────────────────────────
df_params = pd.read_excel(PARAM_PATH)

df_rpca = df_params[
    (df_params["method"] == "rpca")
    & df_params["max_dilation"].notna()
].copy()
df_rpca["subject_id"] = df_rpca["subject_id"].astype(int)
df_rpca["cycle_index"] = df_rpca["cycle_index"].astype(int)

# ── rank cycles within each recording-segment ─────────────────────────────
def rank_cycles(group):
    """Rank 3 cycles: best=highest max_dilation, worst=lowest."""
    ranked = group.sort_values("max_dilation", ascending=False).reset_index(drop=True)
    labels = {0: "best", 1: "middle", 2: "worst"}
    ranked["rank"] = ranked.index.map(labels)
    return ranked

df_ranked = (
    df_rpca
    .groupby(["subject_id", "visit_id", "segment_label"], group_keys=False)
    .apply(rank_cycles)
)

# ── cycle index distribution per rank ────────────────────────────────────
# For each rank (highest/middle/lowest MD), count how often each cycle
# index (0, 1, 2) was selected, separately for artery and vein.
rank_cycle_rows = []
for seg_label in VESSEL_LABELS:
    vessel_type = VESSEL_LABELS[seg_label]
    sub = df_ranked[df_ranked["segment_label"] == seg_label]
    if RETAINED_SUBJECTS is not None:
        retained = RETAINED_SUBJECTS.get(seg_label, set())
        sub = sub[sub["subject_id"].isin(retained)]
    for rank in ["best", "middle", "worst"]:
        rank_sub = sub[sub["rank"] == rank]
        total = len(rank_sub)
        for ci in [0, 1, 2]:
            count = (rank_sub["cycle_index"] == ci).sum()
            frac = count / total if total > 0 else 0
            rank_cycle_rows.append({
                "vessel_type": vessel_type,
                "rank": {"best": "highest_MD", "middle": "middle_MD", "worst": "lowest_MD"}[rank],
                "cycle_index": ci + 1,  # 1-indexed for readability
                "count": int(count),
                "total": int(total),
                "fraction": round(frac, 3),
            })

df_rank_cycles = pd.DataFrame(rank_cycle_rows)
rank_csv_path = FIG_DIR / "cycle_rank_distribution.csv"
df_rank_cycles.to_csv(rank_csv_path, index=False)
print(f"\nCycle rank distribution saved: {rank_csv_path}")
print(df_rank_cycles.to_string(index=False))

# ── extract per-cycle traces ──────────────────────────────────────────────
def extract_cycle_trace(s_hat, t_abs, flicker_start):
    """Cut a single cycle, baseline-normalise, interpolate gaps, resample."""
    t_start = flicker_start - PRE_SEC
    t_end = flicker_start + POST_SEC

    mask = (t_abs >= t_start) & (t_abs < t_end)
    t_rel = t_abs[mask] - flicker_start
    y = np.asarray(s_hat[mask], dtype=float)

    # baseline normalise: subtract median of 30 s pre-flicker
    bl_mask = t_rel < 0
    bl_vals = y[bl_mask]
    bl_vals = bl_vals[np.isfinite(bl_vals)]
    if len(bl_vals) == 0:
        return None
    y -= np.median(bl_vals)

    # interpolate over gaps
    valid = np.isfinite(y)
    if valid.sum() < 3:
        return None
    if (~valid).any():
        f = interp1d(t_rel[valid], y[valid], kind="linear",
                     bounds_error=False, fill_value=np.nan)
        y = f(t_rel)

    # resample onto common time grid
    valid2 = np.isfinite(y)
    if valid2.sum() < 3:
        return None
    f2 = interp1d(t_rel[valid2], y[valid2], kind="linear",
                  bounds_error=False, fill_value=np.nan)
    return f2(T_COMMON)


# ── collect traces per vessel × rank ──────────────────────────────────────
traces = {}  # (seg_label, rank) -> list of 1-D arrays on T_COMMON

_seg_iter = (
    RETAINED_SUBJECTS.items()
    if RETAINED_SUBJECTS is not None
    else [(lbl, None) for lbl in VESSEL_LABELS]
)
for seg_label, retained in _seg_iter:
    for _, row in df_ranked[df_ranked["segment_label"] == seg_label].iterrows():
        sid = int(row["subject_id"])
        if retained is not None and sid not in retained:
            continue

        vid = row["visit_id"]
        ci = int(row["cycle_index"])
        rank = row["rank"]
        seg_key = f"{sid}_{vid}_{seg_label}"

        payload = rpca_results.get(seg_key)
        if payload is None:
            continue

        t_abs = np.asarray(payload["t"], dtype=float)
        s_hat = np.asarray(payload["s_hat"], dtype=float)

        if ci not in FLICKER_ONSETS:
            continue
        flicker_start = FLICKER_ONSETS[ci]

        tr = extract_cycle_trace(s_hat, t_abs, flicker_start)
        if tr is not None:
            traces.setdefault((seg_label, rank), []).append(tr)

# ── plot ──────────────────────────────────────────────────────────────────
for seg_label, vessel_name in VESSEL_LABELS.items():
    fig, ax = plt.subplots(figsize=(7, 4))

    # flicker shading
    ax.axvspan(0, FLICKER_DUR, alpha=0.08, color="#888888", zorder=0)
    ax.axhline(0, color="#cccccc", linewidth=0.5, linestyle=":", zorder=1)
    ax.axvline(0, color="#999999", linewidth=0.6, linestyle="--", zorder=2)
    ax.axvline(FLICKER_DUR, color="#999999", linewidth=0.6, linestyle="--", zorder=2)

    styles = RANK_STYLE[seg_label]
    for rank in ("best", "middle", "worst"):
        key = (seg_label, rank)
        arr_list = traces.get(key, [])
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
    ax.set_title(f"$\\bf{{{PANEL_LETTER[seg_label]}}}$  {VESSEL_PANEL[seg_label]}",
                 fontsize=10, loc="left")
    ax.text(10, Y_LIM[1] - (Y_LIM[1] - Y_LIM[0]) * 0.04, "Flicker",
            ha="center", va="top", fontsize=8, color="#888888", style="italic")

    plt.tight_layout()
    out_path = FIG_DIR / f"fig_cycle_quality_{vessel_name}.png"
    fig.savefig(out_path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

print("Done.")
