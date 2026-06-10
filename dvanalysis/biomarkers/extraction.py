"""ParameterExtractor — compute defined parameters per cycle across segments."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np
import pandas as pd

from dvanalysis.domain import Dataset, Recording, Segment, SegmentSignal
from dvanalysis.preprocessing.result import PreprocessResult
from .definitions import ParameterDefinition, make_default_definitions
from .primitives import masked_trace_mean, masked_trace_median


@dataclass(frozen=True)
class ParameterExtractorConfig:
    trace_source: str = "raw"
    raw_agg: str = "median"
    strict: bool = True
    debug: bool = False
    keep_failed_rows: bool = True


class ParameterExtractor:
    def __init__(self, config: ParameterExtractorConfig, definitions: Optional[Mapping[str, ParameterDefinition]] = None):
        self.config = config
        self.definitions = dict(definitions) if definitions is not None else make_default_definitions()

    def _trace_from_segment_signal(self, sig: SegmentSignal):
        if self.config.raw_agg == "mean":
            y = masked_trace_mean(sig)
        elif self.config.raw_agg == "median":
            y = masked_trace_median(sig)
        else:
            raise ValueError(f"Unknown raw_agg '{self.config.raw_agg}'.")
        return sig.t, y, sig.units

    def _trace_from_result(self, res: PreprocessResult):
        return res.input_signal.t, res.s_hat, "percent"

    def _coerce_trace(self, obj):
        if isinstance(obj, SegmentSignal):
            t, y, units = self._trace_from_segment_signal(obj)
            return t, y, units, obj.protocol
        if isinstance(obj, PreprocessResult):
            t, y, units = self._trace_from_result(obj)
            return t, y, units, obj.input_signal.protocol
        raise TypeError(f"Unsupported type: {type(obj)}")

    def _base_row(self, segment, *, method_name, cycle_index, units, target_id):
        meta = segment.signal.meta if isinstance(segment.signal.meta, dict) else {}
        return {
            "target_id": target_id, "subject_id": meta.get("subject_id"),
            "visit_id": meta.get("visit_id"), "segment_label": segment.segment_label,
            "vessel_type": segment.vessel_type, "method": method_name,
            "cycle_index": int(cycle_index), "units": units,
            "ok": True, "error": "", "error_stage": "",
        }

    def _failed_row(self, segment, *, method_name, cycle_index, units, target_id, error, stage):
        row = self._base_row(segment, method_name=method_name, cycle_index=cycle_index, units=units, target_id=target_id)
        row["ok"] = False
        row["error"] = f"{type(error).__name__}: {error}"
        row["error_stage"] = stage
        for key in self.definitions:
            row[key] = float("nan")
        return row

    def analyze_segment(self, segment, obj, method_name, target_id=None) -> List[Dict[str, Any]]:
        try:
            t, y, units, protocol = self._coerce_trace(obj)
        except Exception as e:
            if self.config.strict:
                raise
            if not self.config.keep_failed_rows:
                return []
            return [self._failed_row(segment, method_name=method_name, cycle_index=-1,
                                     units="", target_id=target_id, error=e, stage="coerce_trace")]

        try:
            cycles = getattr(protocol, "cycles", None)
            if cycles is None or len(cycles) == 0:
                raise ValueError("Protocol has no cycles.")
        except Exception as e:
            if self.config.strict:
                raise
            if not self.config.keep_failed_rows:
                return []
            return [self._failed_row(segment, method_name=method_name, cycle_index=-1,
                                     units=units, target_id=target_id, error=e, stage="protocol_cycles")]

        rows: List[Dict[str, Any]] = []
        for cyc in cycles:
            cyc_index = int(getattr(cyc, "index", -1))
            base = self._base_row(segment, method_name=method_name, cycle_index=cyc_index, units=units, target_id=target_id)

            any_failed = False
            first_error = None
            first_error_stage = ""

            for key, definition in self.definitions.items():
                try:
                    value = float(definition.compute(y, t, cyc))
                except Exception as e:
                    any_failed = True
                    if first_error is None:
                        first_error = e
                        first_error_stage = f"compute:{key}"
                    if self.config.strict:
                        raise
                    value = float("nan")
                base[key] = value

            if any_failed and first_error is not None:
                base["ok"] = False
                base["error"] = f"{type(first_error).__name__}: {first_error}"
                base["error_stage"] = first_error_stage
                if not self.config.keep_failed_rows:
                    continue

            rows.append(base)
        return rows

    def analyze_recording(self, recording, per_segment, method_name, target_id_prefix=None):
        all_rows = []
        for seg_label, obj in per_segment.items():
            if seg_label not in recording.segments:
                if self.config.strict:
                    raise KeyError(f"Missing segment {seg_label}")
                continue
            segment = recording.segments[seg_label]
            target_id = f"{target_id_prefix}:{recording.subject_id}_{recording.visit_id}_{seg_label}" if target_id_prefix else None
            try:
                all_rows.extend(self.analyze_segment(segment, obj, method_name=method_name, target_id=target_id))
            except Exception as e:
                if self.config.strict:
                    raise
                if self.config.keep_failed_rows:
                    all_rows.append(self._failed_row(segment, method_name=method_name, cycle_index=-1,
                                                     units="", target_id=target_id, error=e, stage="analyze_segment"))
        return pd.DataFrame(all_rows)

    def analyze_dataset(self, dataset, methods_by_segment):
        rows = []
        rec_index = {}
        for rec in dataset.recordings:
            for seg_label in rec.segments:
                rec_index[f"{rec.subject_id}_{rec.visit_id}_{seg_label}"] = rec

        for method_name, mapping in methods_by_segment.items():
            for seg_key, obj in mapping.items():
                if seg_key not in rec_index:
                    if self.config.strict:
                        raise KeyError(f"Missing key: {seg_key}")
                    continue
                rec = rec_index[seg_key]
                seg_label = seg_key.split("_")[-1]
                if seg_label not in rec.segments:
                    if self.config.strict:
                        raise KeyError(f"Missing segment: {seg_label}")
                    continue
                seg = rec.segments[seg_label]
                try:
                    rows.extend(self.analyze_segment(seg, obj, method_name=method_name, target_id=seg_key))
                except Exception as e:
                    if self.config.strict:
                        raise
                    if self.config.keep_failed_rows:
                        rows.append(self._failed_row(seg, method_name=method_name, cycle_index=-1,
                                                     units="", target_id=seg_key, error=e, stage="analyze_segment"))
        return pd.DataFrame(rows)

    def to_excel(self, df, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(path, index=False)
