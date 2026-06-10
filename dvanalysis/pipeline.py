"""End-to-end DVA processing pipeline: raw data -> denoised -> biomarkers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import pandas as pd

from dvanalysis.domain import Dataset, Recording, Segment, SegmentSignal
from dvanalysis.preprocessing.result import PreprocessResult


class Pipeline:
    """High-level pipeline composing IO, preprocessing, and biomarker extraction.

    Example
    -------
    >>> from dvanalysis import Pipeline
    >>> results = Pipeline().run("recording_dir/")
    """

    def __init__(
        self,
        *,
        preprocessor: str = "rpca",
        preprocessor_config: Optional[Any] = None,
        parameter_definitions: Optional[Mapping] = None,
        reader_settings: Optional[Any] = None,
    ) -> None:
        self.preprocessor_name = preprocessor
        self.preprocessor_config = preprocessor_config
        self.parameter_definitions = parameter_definitions
        self.reader_settings = reader_settings

    def run(
        self,
        path: Union[str, Path],
        protocol: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Run the full pipeline on a directory of DVA recordings.

        Parameters
        ----------
        path : str or Path
            Root directory containing DVA export files.
        protocol : StimulusProtocol, optional
            Stimulus protocol. Required for loading data.

        Returns
        -------
        dict with keys:
            "dataset" : Dataset
            "results" : dict mapping (subject_id, visit_id, segment_label) -> PreprocessResult
            "parameters" : pd.DataFrame (one row per cycle per segment)
        """
        from dvanalysis.io.readers import ImedosReader, ImedosReaderSettings, DataReaderConfig

        if protocol is None:
            raise ValueError(
                "protocol must be provided. Example:\n"
                "  from dvanalysis.domain import StimulusProtocol\n"
                "  protocol = StimulusProtocol(name='my_protocol', fs=25.0, ...)"
            )

        # --- Load data ---
        reader_settings = self.reader_settings or ImedosReaderSettings()
        reader_config = DataReaderConfig(protocol=protocol)
        reader = ImedosReader(config=reader_config, settings=reader_settings)
        dataset = reader.read(Path(path))

        # --- Preprocess ---
        all_results: Dict[str, PreprocessResult] = {}

        if self.preprocessor_name == "rpca":
            from dvanalysis.preprocessing.rpca_denoise import MyMethodRPCA, MyMethodConfig
            cfg = self.preprocessor_config or MyMethodConfig()
            proc = MyMethodRPCA(config=cfg)

            for rec in dataset.recordings:
                for seg in rec.iter_segments():
                    key = f"{rec.subject_id}_{rec.visit_id}_{seg.segment_label}"
                    result = proc.run(seg.signal)
                    all_results[key] = result

        elif self.preprocessor_name == "group_rpca":
            from dvanalysis.preprocessing.group_rpca import RPCARecordingPreprocessor, RPCARecordingConfig
            cfg = self.preprocessor_config or RPCARecordingConfig()
            proc = RPCARecordingPreprocessor(config=cfg)

            for rec in dataset.recordings:
                rec_results = proc.run(rec)
                for seg_label, result in rec_results.items():
                    key = f"{rec.subject_id}_{rec.visit_id}_{seg_label}"
                    all_results[key] = result

        elif self.preprocessor_name == "kotliar":
            from dvanalysis.preprocessing.kotliar import KotliarPreprocessor, KotliarConfig
            cfg = self.preprocessor_config or KotliarConfig()
            proc = KotliarPreprocessor(config=cfg)

            for rec in dataset.recordings:
                for seg in rec.iter_segments():
                    key = f"{rec.subject_id}_{rec.visit_id}_{seg.segment_label}"
                    result = proc.run(seg.signal)
                    all_results[key] = result

        elif self.preprocessor_name == "gherghel":
            from dvanalysis.preprocessing.gherghel import GherghelPreprocessor, GherghelConfig
            cfg = self.preprocessor_config or GherghelConfig()
            proc = GherghelPreprocessor(config=cfg)

            for rec in dataset.recordings:
                for seg in rec.iter_segments():
                    key = f"{rec.subject_id}_{rec.visit_id}_{seg.segment_label}"
                    result = proc.run(seg.signal)
                    all_results[key] = result

        else:
            raise ValueError(f"Unknown preprocessor '{self.preprocessor_name}'.")

        # --- Extract biomarkers ---
        from dvanalysis.biomarkers.extraction import ParameterExtractor, ParameterExtractorConfig
        from dvanalysis.biomarkers.definitions import make_default_definitions

        definitions = self.parameter_definitions or make_default_definitions()
        extractor = ParameterExtractor(
            config=ParameterExtractorConfig(trace_source="s_hat", strict=False),
            definitions=definitions,
        )
        parameters_df = extractor.analyze_dataset(dataset, {self.preprocessor_name: all_results})

        return {
            "dataset": dataset,
            "results": all_results,
            "parameters": parameters_df,
        }
