"""
Empirical effective rank of real multi-locus DVA recordings.

For each retained segment, computes the singular-value spectrum of the centred
locus matrix on the global baseline window and on the three flicker windows
separately, then reports the variance explained by the leading principal
component. Aggregates per vessel type and writes a results CSV plus a
publication-ready figure.

The output supports the spatial-coherence analysis in Appendix~G of the
manuscript (Spatial-Heterogeneity Sensitivity).

Usage
-----
    python dvanalysis/examples/run_effective_rank.py \
        --data-dir ./data/Healthy_Volunteers \
        --out-dir ./outputs/effective_rank
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from dvanalysis.domain import StimulusCycle, StimulusProtocol, TimeWindow
from dvanalysis.io import DataReaderConfig, ImedosReader


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

SEGMENT_LABELS = ("A1", "V3")
VESSEL_COLOURS = {"artery": "#B2182B", "vein": "#2166AC"}


def _load_retained_subjects(path: Optional[Path]) -> Optional[dict]:
    """Mirror of run_nonstationary_hybrid._load_subjects for consistency."""
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


def _window_indices(t: np.ndarray, w: TimeWindow) -> np.ndarray:
    return np.where((t >= float(w.start_sec)) & (t < float(w.end_sec)))[0]


def _effective_rank(X: np.ndarray) -> dict:
    """Singular-value-based effective-rank summary of a (T, P) matrix.

    Centres each locus by its window-mean before SVD. NaN-handling: any
    timepoint with at least one NaN locus is dropped (cluster-wise, since
    DVA missingness is contiguous).
    """
    if X.ndim != 2 or X.shape[0] < 5 or X.shape[1] < 2:
        return dict(pc1_var_frac=np.nan, pc2_var_frac=np.nan,
                    n_loci=int(X.shape[1]), n_samples=int(X.shape[0]))
    # Drop rows with any NaN (contiguous DVA gaps)
    keep = ~np.isnan(X).any(axis=1)
    Y = X[keep, :]
    if Y.shape[0] < 5:
        return dict(pc1_var_frac=np.nan, pc2_var_frac=np.nan,
                    n_loci=int(X.shape[1]), n_samples=int(keep.sum()))
    # Centre per locus
    Yc = Y - np.nanmean(Y, axis=0, keepdims=True)
    # Economy SVD on (samples, loci)
    try:
        s = np.linalg.svd(Yc, compute_uv=False)
    except np.linalg.LinAlgError:
        return dict(pc1_var_frac=np.nan, pc2_var_frac=np.nan,
                    n_loci=int(X.shape[1]), n_samples=int(Y.shape[0]))
    var = s ** 2
    total = float(var.sum())
    pc1 = float(var[0] / total) if total > 0 else np.nan
    pc2 = float(var[1] / total) if (total > 0 and len(var) > 1) else np.nan
    return dict(
        pc1_var_frac=pc1,
        pc2_var_frac=pc2,
        n_loci=int(X.shape[1]),
        n_samples=int(Y.shape[0]),
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Empirical effective rank of real multi-locus DVA recordings."
    )
    p.add_argument("--data-dir", type=Path, default=Path("./data/Healthy_Volunteers"))
    p.add_argument("--out-dir", type=Path, default=Path("./outputs/effective_rank"))
    p.add_argument("--subjects-json", type=Path, default=None,
                   help="JSON of retained subject IDs per segment label.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    retained = _load_retained_subjects(args.subjects_json)
    if retained is None:
        print("WARNING: cycle_quality_subjects.json not found; using all segments.")
    else:
        print(f"Filtering to retained subjects: "
              f"{sum(len(v) for v in retained.values())} per-segment-label IDs.")

    print(f"Loading dataset from {args.data_dir} ...")
    ds = ImedosReader(DataReaderConfig(protocol=PROTOCOL)).read(args.data_dir)
    print(f"  Loaded {len(ds.recordings)} recordings")

    rows = []
    try:
        from tqdm import tqdm
        rec_iter = tqdm(list(ds.recordings), desc="Effective rank", unit="rec")
    except ImportError:
        rec_iter = ds.recordings

    for rec in rec_iter:
        for seg_label in SEGMENT_LABELS:
            if seg_label not in rec.segments:
                continue
            if (
                retained is not None
                and str(rec.subject_id) not in retained.get(seg_label, set())
            ):
                continue
            seg = rec.segments[seg_label]
            sig = seg.signal
            t = np.asarray(sig.t, float)
            x = np.asarray(sig.x, float).copy()
            m = np.asarray(sig.m, bool)
            x[~m] = np.nan
            seg_key = f"{rec.subject_id}_{rec.visit_id}_{seg_label}"

            # Global baseline window
            bl_idx = _window_indices(t, PROTOCOL.global_baseline)
            bl_summary = _effective_rank(x[bl_idx, :]) if bl_idx.size else None
            if bl_summary is not None:
                rows.append(dict(
                    seg_key=seg_key, vessel_type=seg.vessel_type,
                    window="baseline", cycle_idx=-1,
                    **bl_summary,
                ))

            # Per-cycle flicker windows
            for ci, cyc in enumerate(PROTOCOL.cycles):
                fl_idx = _window_indices(t, cyc.flicker)
                if fl_idx.size == 0:
                    continue
                summary = _effective_rank(x[fl_idx, :])
                rows.append(dict(
                    seg_key=seg_key, vessel_type=seg.vessel_type,
                    window="flicker", cycle_idx=ci,
                    **summary,
                ))

    df = pd.DataFrame(rows)
    csv_path = args.out_dir / "effective_rank_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({len(df)} rows, {df['seg_key'].nunique()} segments)")

    # Summary
    summary = (
        df.dropna(subset=["pc1_var_frac"])
          .groupby(["vessel_type", "window"])["pc1_var_frac"]
          .agg(median="median", q25=lambda s: float(np.percentile(s, 25)),
               q75=lambda s: float(np.percentile(s, 75)),
               n="count")
          .round(3)
          .reset_index()
    )
    print("\nVariance explained by PC1 (median [IQR]):")
    print(summary.to_string(index=False))
    summary.to_csv(args.out_dir / "effective_rank_summary.csv", index=False)

    # Figure: histogram of PC1 variance fraction per vessel × window
    plt.rcParams.update({"font.size": 9, "savefig.dpi": 200})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), sharey=True)
    bins = np.linspace(0, 1, 26)
    for ax, window in zip(axes, ("baseline", "flicker")):
        for vessel, colour in VESSEL_COLOURS.items():
            sub = df[(df["window"] == window) & (df["vessel_type"] == vessel)]
            sub = sub.dropna(subset=["pc1_var_frac"])
            if sub.empty:
                continue
            ax.hist(sub["pc1_var_frac"], bins=bins, color=colour, alpha=0.65,
                    edgecolor=colour, linewidth=0.9, zorder=3,
                    label=f"{vessel.capitalize()} (median = {sub['pc1_var_frac'].median():.2f})")
        ax.set_xlabel("Variance explained by PC1", fontsize=8)
        ax.set_title(f"$\\bf{{{window[0].upper()}}}$  {window.capitalize()} window",
                     loc="left", fontsize=10)
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=7, loc="upper left",
                  framealpha=0.95, edgecolor="#dddddd", fancybox=False)
    axes[0].set_ylabel("Segments", fontsize=8)
    fig.tight_layout(w_pad=1.5)
    out_png = args.out_dir / "fig_effective_rank.png"
    out_pdf = args.out_dir / "fig_effective_rank.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
