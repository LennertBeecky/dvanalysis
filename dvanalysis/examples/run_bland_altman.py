"""
Generate Bland-Altman figure for within-session cycle-to-cycle variability.

Reads dva_parameters.xlsx (pre-computed on server), filters to retained
RPCA segments, and plots MD, DA, tMAD and tMAD30 in a 4x2 grid.

If tMAD30 (time_to_max_dilation_30) is missing from the Excel file,
it is computed from RPCA pickles.

Usage:
    python dvanalysis/examples/run_bland_altman.py
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dvanalysis.visualization.bland_altman import (
    compute_bland_altman_data,
    plot_bland_altman,
)

# ── Configuration (paths and retained subjects supplied externally) ──
import argparse
import json

_DEF_SUBJECTS = Path(__file__).resolve().parent / "cycle_quality_subjects.json"


def _parse_args():
    p = argparse.ArgumentParser(
        description="Bland-Altman within-session cycle-to-cycle variability figure.")
    p.add_argument("--param-path", type=Path, required=True,
                   help="Precomputed biomarker table (dva_parameters.xlsx).")
    p.add_argument("--subjects-json", type=Path, default=_DEF_SUBJECTS,
                   help="JSON of retained subject IDs per segment label "
                        "(gitignored; see cycle_quality_subjects.example.json).")
    return p.parse_args()


_args = _parse_args()
PARAM_PATH = _args.param_path
OUT_DIR = PARAM_PATH.parent
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _load_retained(path):
    """Map the gitignored per-segment-label subject JSON to vessel-keyed sets."""
    with open(path) as f:
        raw = json.load(f)
    by_label = {k: {int(s) for s in v}
                for k, v in raw.items() if not k.startswith("_")}
    return {"artery": by_label.get("A1", set()), "vein": by_label.get("V3", set())}


RETAINED = _load_retained(_args.subjects_json)
VESSEL_SEGMENTS = {"artery": "A1", "vein": "V3"}

# ── Biomarkers ───────────────────────────────────────────────────────
BIOMARKERS = ["max_dilation", "max_constriction", "dilation_amplitude", "time_to_max_dilation", "time_to_max_dilation_30"]
BIO_LABELS = ["MD (%)", "MC (%)", "DA (%)", "tMAD (s)", "tMAD30 (s)"]
VESSELS = ["artery", "vein"]
# MC and DA are artery-only (veins lack consistent post-flicker constriction)
ARTERY_ONLY = {"max_constriction", "dilation_amplitude"}
VESSEL_COLOURS = {"artery": "#B2182B", "vein": "#2166AC"}
VESSEL_PANEL = {"artery": "Artery", "vein": "Vein"}
PANEL_LETTER = {"artery": "A", "vein": "B"}

# ── Load and filter ──────────────────────────────────────────────────
print(f"Loading {PARAM_PATH} ...")
df = pd.read_excel(PARAM_PATH)

# RPCA only, A1 and V3
df = df[(df["method"] == "rpca") & (df["segment_label"].isin(["A1", "V3"]))].copy()
df["subject_id"] = df["subject_id"].astype(int)

# Filter to retained subjects per vessel
dfs = []
for vessel, seg in VESSEL_SEGMENTS.items():
    sub = df[(df["vessel_type"] == vessel) & (df["segment_label"] == seg)]
    sub = sub[sub["subject_id"].isin(RETAINED[vessel])]
    dfs.append(sub)
df = pd.concat(dfs, ignore_index=True)

print(f"Filtered: {len(df)} rows")
for v in VESSELS:
    n = len(df[df["vessel_type"] == v])
    ns = df[df["vessel_type"] == v]["subject_id"].nunique()
    print(f"  {v.capitalize()}: {n} rows, {ns} subjects")

# ── Check if tMAD30 exists, compute from pickles if not ──────────────
if "time_to_max_dilation_30" not in df.columns:
    print("\ntMAD30 not in Excel, computing from RPCA pickles ...")
    import gzip, pickle
    from dvanalysis.domain import TimeWindow, StimulusCycle, StimulusProtocol
    from dvanalysis.biomarkers.library import calc_dilation_max_30_t

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

    PICKLE_DIR = OUT_DIR / "preprocessed_segment"
    rpca_results = {}
    for fp in sorted(PICKLE_DIR.glob("*_rpca.pkl.gz")):
        seg_key = fp.name.replace("_rpca.pkl.gz", "")
        with gzip.open(fp, "rb") as f:
            rpca_results[seg_key] = pickle.load(f)
    print(f"  Loaded {len(rpca_results)} RPCA pickles")

    tMAD30_values = []
    for _, row in df.iterrows():
        sid = int(row["subject_id"])
        vid = row["visit_id"]
        seg_label = row["segment_label"]
        ci = int(row["cycle_index"])
        seg_key = f"{sid}_{vid}_{seg_label}"

        payload = rpca_results.get(seg_key)
        if payload is None:
            tMAD30_values.append(np.nan)
            continue

        t = np.asarray(payload["t"], float)
        s_hat = np.asarray(payload["s_hat"], float)
        cycle = PROTOCOL.cycles[ci]

        try:
            val = calc_dilation_max_30_t(
                s_hat, t,
                w_search=TimeWindow("flicker_ext", cycle.flicker.start_sec, cycle.flicker.end_sec + 10.0),
                w_baseline=cycle.baseline,
            )
            tMAD30_values.append(val)
        except Exception:
            tMAD30_values.append(np.nan)

    df["time_to_max_dilation_30"] = tMAD30_values
    valid = np.isfinite(df["time_to_max_dilation_30"]).sum()
    print(f"  Computed tMAD30 for {valid}/{len(df)} rows")

# ── Compute Bland-Altman data ────────────────────────────────────────
ba_data = {}

for vessel in VESSELS:
    for bio in BIOMARKERS:
        sub = df[df["vessel_type"] == vessel]
        if bio not in sub.columns or sub[bio].isna().all():
            continue
        data = compute_bland_altman_data(sub, bio, subject_col="subject_id")
        ba_data[(vessel, bio)] = data
        print(f"{vessel:6s} | {bio:30s}: LoA = [{data['loa_lower']:+.2f}, {data['loa_upper']:+.2f}], "
              f"SD = {data['sd_dev']:.2f}, n = {len(data['means'])}")

# ── Plot: 4 rows x 2 columns ────────────────────────────────────────
plt.rcParams.update({
    "font.size": 10, "figure.dpi": 200,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})

n_rows = len(BIOMARKERS)
fig, axes = plt.subplots(n_rows, 2, figsize=(7.5, 9))

for row, (bio, bio_label) in enumerate(zip(BIOMARKERS, BIO_LABELS)):
    for col, vessel in enumerate(VESSELS):
        ax = axes[row, col]
        if bio in ARTERY_ONLY and vessel == "vein":
            ax.set_visible(False)
            continue
        data = ba_data.get((vessel, bio))
        if data is None:
            ax.set_visible(False)
            continue

        plot_bland_altman(
            data["means"], data["deviations"],
            data["loa_lower"], data["loa_upper"],
            colour=VESSEL_COLOURS[vessel], ax=ax,
            ylabel=bio_label if col == 0 else "",
            xlabel="Within-subject mean" if row == n_rows - 1 else "",
            title=f"$\\bf{{{PANEL_LETTER[vessel]}}}$  {VESSEL_PANEL[vessel]}" if row == 0 else "",
            show_loa_labels=True,
        )

plt.tight_layout(h_pad=0.8, w_pad=0.8)

out_path = FIG_DIR / "fig_bland_altman"
fig.savefig(f"{out_path}.png", dpi=200, facecolor="white", bbox_inches="tight")
fig.savefig(f"{out_path}.pdf", facecolor="white", bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out_path}.png and .pdf")

print("\nDone.")
