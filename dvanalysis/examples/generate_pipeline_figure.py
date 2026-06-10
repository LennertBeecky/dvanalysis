"""
Generate the pipeline figure for the manuscript.

Usage:
    conda run -n dvanalysis python dvanalysis/examples/generate_pipeline_figure.py
"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from dvanalysis.domain import TimeWindow, StimulusCycle, StimulusProtocol
from dvanalysis.io import ImedosReader, DataReaderConfig
from dvanalysis.preprocessing.rpca_denoise import MyMethodRPCA, MyMethodConfig
from dvanalysis.visualization.pipeline_figure import plot_pipeline_figure

# Protocol
protocol = StimulusProtocol(
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

# Load data
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "Healthy_Volunteers"
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# RPCA config (same as paper)
rpca_config = MyMethodConfig(
    fill_missing=False,
    heartbeat_filter=False,
    standardize=True,
    rpca_lmb=0.55,
    rpca_gamma=1000.0,
    rpca_max_iter=70,
    rpca_tol_rel=1e-3,
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

print("Loading data...")
ds = ImedosReader(config=DataReaderConfig(protocol=protocol)).read(DATA_DIR)
rec = next(r for r in ds.recordings if "A1" in r.segments and "V3" in r.segments)

rpca = MyMethodRPCA(config=rpca_config)

# Artery (A1)
seg_label = "A1"
sig = rec.segments[seg_label].signal
print(f"Running RPCA on {seg_label} (T={sig.T}, P={sig.P})...")
result = rpca.run(sig)
print(f"Done: {result.diagnostics.get('n_iter', '?')} iterations")

FUNDUS_PATH = str(Path(__file__).resolve().parent.parent.parent / "manuscript" / "figures" / "fundus_segments.png")

fig = plot_pipeline_figure(
    sig, result,
    cycle_index=0,
    pre_sec=5.0,
    vessel_colour="#B2182B",
    fundus_path=FUNDUS_PATH,
    save_path=str(OUT_DIR / "fig_pipeline_artery"),
    save_pdf=True,
)

# Vein (V3)
seg_label = "V3"
sig_v = rec.segments[seg_label].signal
print(f"Running RPCA on {seg_label} (T={sig_v.T}, P={sig_v.P})...")
result_v = rpca.run(sig_v)

fig_v = plot_pipeline_figure(
    sig_v, result_v,
    cycle_index=0,
    pre_sec=5.0,
    vessel_colour="#2166AC",
    save_path=str(OUT_DIR / "fig_pipeline_vein"),
    save_pdf=True,
)

print("Done.")
