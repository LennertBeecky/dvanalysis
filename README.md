# DVAnalysis

Physiology-informed denoising and biomarker extraction for **Dynamic Retinal Vessel Analysis (DVA)**.

DVAnalysis is a Python library that implements a masked Robust PCA (RPCA) framework with temporal smoothness regularisation for the joint denoising of multichannel DVA recordings, along with a comprehensive flicker-response biomarker extraction pipeline.

## Reference

This library accompanies the manuscript:

> Beeckmans L., Van Eijgen J., Ventura-Abreu N., Stalmans I.\*, De Vos M.\*
> **Preserving Retinal Vessel Flicker Responses: Physiology-Informed Denoising for Dynamic Vessel Analysis.**
> \*Shared last authorship.

Affiliations:

1. Department of Electrical Engineering (ESAT), STADIUS Center for Dynamical Systems, Signal Processing and Data Analytics, KU Leuven, Leuven, Belgium
2. Research Group Ophthalmology, Department of Neurosciences, KU Leuven, Leuven, Belgium
3. Department of Ophthalmology, University Hospitals UZ Leuven, Leuven, Belgium
4. Department of Ophthalmology, Hospital Clínic de Barcelona, Universitat de Barcelona, Barcelona, Spain
5. Department of Development and Regeneration, KU Leuven, Leuven, Belgium

A `CITATION.cff` will be added once the manuscript is accepted.

## Installation

Requires Python ≥ 3.10.

```bash
git clone https://github.com/LennertBeecky/dvanalysis.git
cd dvanalysis
pip install -e .
```

For development (tests, linting):

```bash
pip install -e ".[dev]"
pytest
```

## Usage

The typical workflow is three steps: load a recording, denoise it with masked
RPCA, then extract the flicker-response biomarkers you care about.

```python
from dvanalysis.io import ImedosReader, DataReaderConfig
from dvanalysis.preprocessing import MyMethodRPCA, MyMethodConfig
from dvanalysis.biomarkers import (
    ParameterExtractor, ParameterExtractorConfig, make_paper_definitions,
)

# 1. Load recordings (protocol describes the stimulus timing; see quickstart.py)
reader = ImedosReader(config=DataReaderConfig(protocol=protocol))
dataset = reader.read("path/to/recordings/")
seg = dataset.recordings[0].segments["A1"]      # one vessel segment

# 2. Denoise with masked RPCA -> result.s_hat is the cleaned % change trace
rpca = MyMethodRPCA(config=MyMethodConfig(
    rpca_lmb=0.55, rpca_gamma=1000.0, harmonize_output=True,
))
result = rpca.run(seg.signal)

# 3. Select the biomarkers and extract them per flicker cycle
extractor = ParameterExtractor(
    config=ParameterExtractorConfig(trace_source="s_hat"),
    definitions=make_paper_definitions(),       # full clinical set (9 parameters)
    #                                             or make_default_definitions(),
    #                                             or a custom subset of ParameterDefinition
)
rows = extractor.analyze_segment(seg, result, method_name="rpca")
for row in rows:
    print(row["cycle_index"], row["max_dilation"], row["dilation_amplitude"])
```

Swap `make_paper_definitions()` for `make_default_definitions()` or your own
list of `ParameterDefinition` objects to choose exactly which biomarkers are
computed. A complete, runnable walkthrough (including how to build a
`StimulusProtocol`) is in
[`dvanalysis/examples/quickstart.ipynb`](dvanalysis/examples/quickstart.ipynb)
(also available as `quickstart.py`).

## Package layout

```
dvanalysis/
├── domain/         data model (Dataset > Recording > Segment > SegmentSignal)
├── io/             Imedos-format readers and CSV / Excel writers
├── preprocessing/  RPCA solver, group-RPCA, Kotliar SMS, Gherghel, filtering, centering
├── quality/        recording-level quality criteria and reporting
├── biomarkers/     flicker-response biomarker library and extractor
├── validation/     hybrid-sample validation framework with synthetic noise
├── visualization/  Plotly / matplotlib plots (triage, cycles, Bland–Altman, comparison)
├── examples/       runnable scripts that reproduce the paper figures and tables
└── tests/          pytest suite (synthetic data, no external dependencies)
```

## Reproducing the manuscript results

The scripts in [`dvanalysis/examples/`](dvanalysis/examples/) reproduce every
figure and table in the manuscript. Each takes a `--data-dir` pointing at a
directory of DVA recordings (not redistributable; see [Data](#data)) and
writes figures and CSV tables to an output directory. Run everything end to
end with:

```bash
python dvanalysis/examples/run_all_experiments.py --data-dir ./data --out-dir ./outputs
```

Statistical comparisons use fixed random seeds (pass `--seed` /
`--paired-stats-seed` to vary them) and a subject-level cluster bootstrap with
Holm correction over the primary comparison family.

## Data

This repository does **not** contain patient recordings. The clinical datasets
used in the manuscript cannot be redistributed for ethical reasons (IRB
approval S59677, University Hospitals Leuven). All committed example outputs
are de-identified (subjects relabelled `S01`–`S52`). The `tests/` suite is
fully synthetic and requires no external data, and the example scripts include
instructions for pointing the pipeline at your own DVA data.

## Roadmap

- A user-friendly GUI for importing data, running preprocessing, inspecting raw and denoised traces, extracting biomarkers and exporting figures and tables
- Readers for more DVA export formats and a generic long-format CSV importer
- Interactive denoised-pipeline visualisations from raw to denoised to biomarkers, with method-comparison plots and export-ready figure presets

## License

MIT — see [`LICENSE`](LICENSE).

## Contact

Corresponding author: Ingeborg Stalmans — ingeborg.stalmans@mac.com
