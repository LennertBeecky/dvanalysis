"""One-off script to generate reference outputs for test_reference.py.

Usage:
    python -m dvanalysis.tests._gen_refs [SUBJECT_ID ...]

With no arguments it generates references for every recording found in the
data directory. Pass explicit subject IDs to restrict the set. No identifiers
are stored in this file.
"""
import sys
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
from dvanalysis.domain import TimeWindow, StimulusCycle, StimulusProtocol
from dvanalysis.io.readers import ImedosReader, DataReaderConfig
from dvanalysis.preprocessing.rpca_denoise import MyMethodRPCA, MyMethodConfig

gb = TimeWindow("baseline", 0.0, 20.0)
cycles = [StimulusCycle(index=i,
    baseline=TimeWindow("baseline", float(b), float(f)),
    flicker=TimeWindow("flicker", float(f), float(r)),
    recovery=TimeWindow("recovery", float(r), float(e)))
    for i, (b, f, r, e) in enumerate([(20,50,70,120),(120,150,170,220),(220,250,270,320)])]
protocol = StimulusProtocol(name="DVA_3cycle_custom", fs=25.0, global_baseline=gb, cycles=cycles)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ds = ImedosReader(config=DataReaderConfig(protocol=protocol)).read(
    _PROJECT_ROOT / "data" / "Healthy_Volunteers")
rpca_cfg = MyMethodConfig(rpca_lmb=0.01, rpca_mu=1.0, rpca_rho=5.0, rpca_gamma=0.0,
    rpca_max_iter=200, rpca_tol_rel=1e-4, standardize=True, harmonize_output=True,
    harmonize_percent_mode="delta_over_baseline", heartbeat_filter=False,
    fill_missing=False, support_min_valid_abs=12)
out = _PROJECT_ROOT / "data" / "reference_outputs"
out.mkdir(parents=True, exist_ok=True)
_requested = sys.argv[1:]
recs = ([ds.get_recording(s, "0") for s in _requested]
        if _requested else list(ds.recordings))
for rec in recs:
    sid, vid = rec.subject_id, rec.visit_id
    for sl in rec.segments:
        r = MyMethodRPCA(config=rpca_cfg).run(rec.segments[sl].signal)
        np.save(str(out / f"{sid}_{vid}_{sl}_rpca_trace.npy"), r.s_hat)
        np.save(str(out / f"{sid}_{vid}_{sl}_rpca_loci.npy"), r.S_hat)
        print(f"{sid}_{vid}_{sl}: trace={r.s_hat.shape} loci={r.S_hat.shape}")
