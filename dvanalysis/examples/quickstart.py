"""
DVAnalysis Quickstart
=====================

This script demonstrates the core workflow:
  1. Define a stimulus protocol
  2. Load DVA recordings from Imedos RTF exports
  3. Run RPCA denoising on one segment
  4. Run Kotliar SMS baseline for comparison
  5. Extract biomarkers per cycle
  6. Plot results

Usage:
    conda run -n dvanalysis python examples/quickstart.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ── 1. Define the stimulus protocol ──────────────────────────────────────────

from dvanalysis.domain import TimeWindow, StimulusCycle, StimulusProtocol

# Standard 3-cycle DVA protocol: 20s global baseline, then 3× (30s baseline + 20s flicker + 50s recovery)
protocol = StimulusProtocol(
    name="DVA_3cycle",
    fs=25.0,  # sampling rate in Hz
    global_baseline=TimeWindow("baseline", 0.0, 20.0),
    cycles=[
        StimulusCycle(index=0,
            baseline=TimeWindow("baseline", 20.0, 50.0),
            flicker=TimeWindow("flicker", 50.0, 70.0),
            recovery=TimeWindow("recovery", 70.0, 120.0)),
        StimulusCycle(index=1,
            baseline=TimeWindow("baseline", 120.0, 150.0),
            flicker=TimeWindow("flicker", 150.0, 170.0),
            recovery=TimeWindow("recovery", 170.0, 220.0)),
        StimulusCycle(index=2,
            baseline=TimeWindow("baseline", 220.0, 250.0),
            flicker=TimeWindow("flicker", 250.0, 270.0),
            recovery=TimeWindow("recovery", 270.0, 320.0)),
    ],
)

# ── 2. Load recordings ──────────────────────────────────────────────────────

from dvanalysis.io import ImedosReader, DataReaderConfig

data_dir = Path(__file__).resolve().parent.parent.parent / "data" / "Healthy_Volunteers"
reader = ImedosReader(config=DataReaderConfig(protocol=protocol))
dataset = reader.read(data_dir)

print(f"Loaded {len(dataset.recordings)} recordings")

# Pick the first recording with an arterial segment
rec = next(r for r in dataset.recordings if "A1" in r.segments)
seg = rec.segments["A1"]
sig = seg.signal
print(f"Segment {seg.segment_label} ({seg.vessel_type}): T={sig.T}, P={sig.P}, valid={sig.valid_fraction():.1%}")

# ── 3. Run RPCA denoising ───────────────────────────────────────────────────

from dvanalysis.preprocessing import MyMethodRPCA, MyMethodConfig

rpca_config = MyMethodConfig(
    fill_missing=False,
    heartbeat_filter=False,
    standardize=True,
    rpca_lmb=0.55,                                              # sparsity penalty (lambda)
    rpca_gamma=1000.0,                                          # temporal smoothness
    rpca_max_iter=70,
    rpca_tol_rel=1e-3,
    locus_agg="median",
    harmonize_output=True,                                      # convert to % change from baseline
    harmonize_percent_mode="delta_over_baseline",
    harmonize_baseline_source="protocol_global",
    harmonize_aggregation_order="percent_then_aggregate",
    harmonize_baseline_per_locus=True,
    support_min_valid_frac=0.75,
    support_min_valid_abs=12,
    hampel_enable=True,
)

rpca = MyMethodRPCA(config=rpca_config)
rpca_result = rpca.run(sig)

print(f"RPCA done: {rpca_result.diagnostics.get('n_iter', '?')} iterations")

# ── 4. Run Kotliar SMS baseline (for comparison) ────────────────────────────

from dvanalysis.preprocessing import KotliarPreprocessor, KotliarConfig

kotliar = KotliarPreprocessor(config=KotliarConfig(
    mode="smooth_only",
    smooth_only_keep_outside_cycles=True,
))
kot_result = kotliar.run(sig)

# ── 5. Extract biomarkers ────────────────────────────────────────────────────

from dvanalysis.biomarkers import ParameterExtractor, ParameterExtractorConfig, make_default_definitions

extractor = ParameterExtractor(
    config=ParameterExtractorConfig(trace_source="s_hat", strict=False),
    definitions=make_default_definitions(),
)

rows = extractor.analyze_segment(seg, rpca_result, method_name="rpca")

print("\nBiomarkers per cycle (RPCA):")
print(f"  {'Cycle':<8} {'Max dilation':>14} {'Max constriction':>18} {'Dil. amplitude':>16}")
for row in rows:
    print(f"  {row['cycle_index']:<8} {row['max_dilation']:>14.3f} {row['max_constriction']:>18.3f} {row['dilation_amplitude']:>16.3f}")

# ── 6. Plot ──────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

t = sig.t

# Top panel: raw median trace
y_raw = sig.aggregate_over_loci(agg="median")
axes[0].plot(t, y_raw, color="#999999", linewidth=0.5, alpha=0.7, label="Raw (median over loci)")
axes[0].set_ylabel("Diameter (a.u.)")
axes[0].set_title(f"Subject {rec.subject_id} — {seg.segment_label} ({seg.vessel_type})")
axes[0].legend(fontsize=8)

# Bottom panel: denoised traces
axes[1].plot(t, rpca_result.s_hat, color="#B2182B", linewidth=1.2, label="RPCA")
axes[1].plot(t, kot_result.s_hat, color="#2166AC", linewidth=1.0, alpha=0.7, label="Kotliar SMS")
axes[1].set_ylabel("% change from baseline")
axes[1].set_xlabel("Time (s)")
axes[1].legend(fontsize=8)

# Shade flicker windows
for ax in axes:
    for cyc in protocol.cycles:
        ax.axvspan(cyc.flicker.start_sec, cyc.flicker.end_sec,
                   alpha=0.08, color="#FFD700", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent / "quickstart_output.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved to examples/quickstart_output.png")
