"""
Generate per-cycle biomarker parameters across the clinical recordings.

For each segment in the dataset, runs RPCA preprocessing with the paper
configuration and extracts the eight biomarkers in `make_paper_definitions()`
(MD, MC, DA, tMAD, tMAD30, time_to_max_constriction, t_mdmc, AUC dilation,
AUC constriction). The result is written as `dva_parameters.xlsx`.

This is the generator for the file consumed by `run_bland_altman.py` and the
within-session Bland-Altman analysis (Section 3.2). Modernised port of
`old_code/00_extract_parameters_quality.py`, RPCA-only.

Usage
-----
    python dvanalysis/examples/run_extract_parameters.py \
        --data-dir ./data/Healthy_Volunteers \
        --out-dir ./outputs/parameters

Output: <out_dir>/dva_parameters.xlsx (and optionally
<out_dir>/preprocessed_segment/*_rpca.pkl.gz pickles).
"""
from __future__ import annotations

import argparse
import gzip
import pickle
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from dvanalysis.domain import StimulusCycle, StimulusProtocol, TimeWindow
from dvanalysis.io import DataReaderConfig, ImedosReader
from dvanalysis.preprocessing.rpca_denoise import MyMethodConfig, MyMethodRPCA
from dvanalysis.biomarkers import (
    ParameterExtractor,
    ParameterExtractorConfig,
)
from dvanalysis.biomarkers.definitions import make_paper_definitions


# ── Protocol (matches the rest of the paper) ──────────────────────────
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

# ── RPCA config (matches run_nonstationary_hybrid.py) ─────────────────
RPCA_CFG = MyMethodConfig(
    rpca_lmb=0.55,
    rpca_gamma=1000.0,
    rpca_max_iter=70,
    rpca_tol_rel=1e-3,
    standardize=True,
    harmonize_output=True,
    harmonize_percent_mode="delta_over_baseline",
    harmonize_baseline_source="protocol_global",
    harmonize_aggregation_order="percent_then_aggregate",
    harmonize_baseline_per_locus=True,
    support_min_valid_frac=0.75,
    support_min_valid_abs=12,
    hampel_enable=True,
)

SEGMENT_LABELS = ("A1", "V3")


def _save_rpca_pickle(result, filepath: Path) -> None:
    payload = {
        "s_hat": result.s_hat,
        "S_hat": result.S_hat,
        "A_hat": result.A_hat,
        "mask": result.m_used,
        "diagnostics": result.diagnostics,
        "config": result.config,
        "t": result.input_signal.t,
        "units": getattr(result.input_signal, "units", None),
    }
    with gzip.open(filepath, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run RPCA on clinical recordings and extract per-cycle biomarkers."
    )
    p.add_argument(
        "--data-dir", type=Path, default=Path("./data/Healthy_Volunteers"),
        help="Clinical DVA recordings directory.",
    )
    p.add_argument(
        "--out-dir", type=Path, default=Path("./outputs/parameters"),
        help="Output directory for the Excel and optional pickles.",
    )
    p.add_argument(
        "--save-pickles", action="store_true",
        help="Also save per-segment RPCA pickles under <out-dir>/preprocessed_segment/.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read dataset
    print(f"Loading dataset from {args.data_dir} ...")
    reader = ImedosReader(DataReaderConfig(protocol=PROTOCOL, units="a.u."))
    dataset = reader.read(args.data_dir)
    print(f"  Loaded {len(dataset.recordings)} recordings")

    # 2. Run RPCA on each retained segment label
    rpca = MyMethodRPCA(config=RPCA_CFG)
    pickle_dir = args.out_dir / "preprocessed_segment"
    if args.save_pickles:
        pickle_dir.mkdir(parents=True, exist_ok=True)

    methods_by_segment = {"rpca": {}}

    try:
        from tqdm import tqdm
        rec_iter = tqdm(list(dataset.recordings), desc="RPCA", unit="rec")
    except ImportError:
        rec_iter = dataset.recordings

    n_ok, n_fail = 0, 0
    for rec in rec_iter:
        for seg_label in SEGMENT_LABELS:
            if seg_label not in rec.segments:
                continue
            seg = rec.segments[seg_label]
            seg_key = f"{rec.subject_id}_{rec.visit_id}_{seg_label}"
            try:
                result = rpca.run(seg.signal)
                methods_by_segment["rpca"][seg_key] = result
                if args.save_pickles:
                    _save_rpca_pickle(result, pickle_dir / f"{seg_key}_rpca.pkl.gz")
                n_ok += 1
            except Exception as e:
                print(f"  [rpca] failed for {seg_key}: {e}")
                n_fail += 1

    print(f"  RPCA: {n_ok} succeeded, {n_fail} failed")

    # 3. Extract biomarkers (paper definitions, includes tMAD30)
    extractor = ParameterExtractor(
        config=ParameterExtractorConfig(
            strict=False, keep_failed_rows=True, debug=False,
        ),
        definitions=make_paper_definitions(),
    )
    df = extractor.analyze_dataset(dataset, methods_by_segment)
    print(f"  Extracted {len(df)} parameter rows ({df['cycle_index'].nunique()} cycles, "
          f"{df['method'].nunique()} method)")

    # 4. Write Excel
    out_path = args.out_dir / "dva_parameters.xlsx"
    extractor.to_excel(df, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
