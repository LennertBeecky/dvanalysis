"""Quality reporting data structures and export utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# QualityReport
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QualityReport:
    """Stores computed quality metrics and decision flags.

    Parameters
    ----------
    target_type : str
        "raw" for SegmentSignal, "processed" for PreprocessResult.
    target_id : str or None
        Optional identifier for the analysed object (file path, hash, etc.).
    """

    analyzer_name: str
    analyzer_version: str
    target_type: str
    target_id: Optional[str] = None

    metrics: Dict[str, Any] = field(default_factory=dict)
    flags: Dict[str, bool] = field(default_factory=dict)
    notes: str = ""

    def passed(self) -> bool:
        if not self.flags:
            return True
        return not any(self.flags.values())

    def __str__(self) -> str:
        lines = [
            f"QualityReport(analyzer={self.analyzer_name} v{self.analyzer_version}, target={self.target_type})",
            f"passed={self.passed()}",
        ]
        if self.flags:
            bad = [k for k, v in self.flags.items() if v]
            if bad:
                lines.append("flags_failed=" + ", ".join(bad))
        if "missing_fraction_total" in self.metrics:
            lines.append(f"missing_total={self.metrics['missing_fraction_total']:.3f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# QualityItem / QualityRun
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QualityItem:
    subject_id: str
    visit_id: str
    segment_label: str
    vessel_type: str
    stage: str
    target_id: Optional[str]
    report: QualityReport


@dataclass(frozen=True)
class QualityRun:
    """A run over a chosen scope (Recording or Dataset)."""

    items: List[QualityItem]
    meta: Dict[str, Any]


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def quality_run_to_dataframe(run: QualityRun) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for item in run.items:
        r = item.report
        row: Dict[str, Any] = {
            "subject_id": item.subject_id,
            "visit_id": item.visit_id,
            "segment_label": item.segment_label,
            "vessel_type": item.vessel_type,
            "stage": item.stage,
            "target_id": item.target_id,
            "passed": r.passed(),
            "notes": r.notes,
        }
        for k, v in r.metrics.items():
            if isinstance(v, (dict, list, tuple)):
                row[k] = str(v)
            else:
                row[k] = v
        for k, v in r.flags.items():
            row[f"flag_{k}"] = bool(v)
        rows.append(row)

    df = pd.DataFrame(rows)
    for k, v in run.meta.items():
        df[f"run_{k}"] = v
    return df


def save_quality_run_excel(run: QualityRun, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = quality_run_to_dataframe(run)
    df.to_excel(out_path, index=False)
    return out_path
