"""DVA data readers — abstract interface and Imedos implementation."""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from dvanalysis.domain import Dataset, Recording, Segment, SegmentSignal, StimulusProtocol


# ---------------------------------------------------------------------------
# Base reader
# ---------------------------------------------------------------------------

@dataclass
class DataReaderConfig:
    protocol: StimulusProtocol
    units: str = "a.u."
    meta: Dict[str, Any] = field(default_factory=dict)


class DataReader(ABC):
    """Reader interface: parse raw exports -> return Dataset."""

    name: str = "DataReader"
    version: str = "0.1"

    def __init__(self, config: DataReaderConfig) -> None:
        self.config = config

    @abstractmethod
    def read(self, root: Path) -> Dataset:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Imedos reader
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ImedosReaderSettings:
    """Reader-specific settings."""

    header_lines_to_skip: int = 4
    strict_T: bool = False
    debug: bool = False


class ImedosReader(DataReader):
    """Reads Imedos locus exports and returns Dataset -> Recording -> Segment -> SegmentSignal.

    Handles common Imedos "Copy data" exports where column separators may be
    literal '\\tab' tokens OR real tab characters, and row endings may be
    literal '\\par' tokens OR actual newlines.

    Time vector is generated as t = arange(T)/fs (no reliance on an explicit time column).
    """

    name: str = "ImedosReader"
    version: str = "0.4"

    def __init__(self, config: DataReaderConfig, settings: Optional[ImedosReaderSettings] = None) -> None:
        super().__init__(config)
        self.settings = settings or ImedosReaderSettings()
        self._filename_re = re.compile(
            r"(?P<subject>[^_]+)_(?P<visit>[^_]+)_(?P<segment>[A-Za-z]\d+)"
        )
        self._extensions = {".rtf", ".txt", ".csv"}
        self._encoding = "utf-8"

    def read(self, root: Path) -> Dataset:
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"Root path does not exist: {root}")

        ds = Dataset(
            name=f"imedos_export:{root.name}",
            description="Parsed Imedos locus exports",
            root_path=root,
            config={"reader": self.name, "reader_version": self.version},
        )

        recordings: Dict[Tuple[str, str], Recording] = {}

        for path in self._iter_files(root):
            info = self._parse_filename(path.name)
            if info is None:
                continue

            subject_id = info["subject"]
            visit_id = info["visit"]
            segment_label = info["segment"]
            vessel_type = self._infer_vessel_type(segment_label)

            try:
                t, x = self._load_matrix_from_file(path)
            except Exception as e:
                if self.settings.debug:
                    print(f"[ImedosReader] Failed parsing: {path}")
                    print(f"[ImedosReader] Reason: {type(e).__name__}: {e}")
                raise

            m = np.isfinite(x)

            signal = SegmentSignal(
                t=t, x=x, m=m,
                protocol=self.config.protocol,
                units=self.config.units,
                meta={
                    "source_path": str(path),
                    "subject_id": subject_id,
                    "visit_id": visit_id,
                    "segment_label": segment_label,
                    "vessel_type": vessel_type,
                },
            )

            seg = Segment(
                segment_label=segment_label,
                vessel_type=vessel_type,
                signal=signal,
            )

            key = (subject_id, visit_id)
            if key not in recordings:
                recordings[key] = Recording(
                    subject_id=subject_id,
                    visit_id=visit_id,
                    recording_datetime=None,
                    meta={"source_root": str(root)},
                )
            recordings[key].add_segment(seg)

        for rec in recordings.values():
            ds.add_recording(rec)

        return ds

    # -------------------------
    # File discovery & naming
    # -------------------------

    def _iter_files(self, root: Path) -> Iterable[Path]:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in self._extensions:
                yield p

    def _parse_filename(self, filename: str) -> Optional[Dict[str, str]]:
        m = self._filename_re.search(filename)
        if m is None:
            return None
        return {
            "subject": m.group("subject"),
            "visit": m.group("visit"),
            "segment": m.group("segment"),
        }

    def _infer_vessel_type(self, segment_label: str) -> str:
        """Infer vessel type from first character of the segment label.

        Works for any naming convention: A1, Art1, V3, Ven2, etc.
        """
        c = segment_label.strip()[0].upper()
        if c == "A":
            return "artery"
        if c == "V":
            return "vein"
        return "unknown"

    # -------------------------
    # Core parsing
    # -------------------------

    def _load_matrix_from_file(self, path: Path) -> Tuple[np.ndarray, np.ndarray]:
        text = path.read_text(encoding=self._encoding, errors="ignore")
        text = self._normalize_imedos_text(text)

        # Fast path: try pandas CSV parser on the normalised text
        x = self._try_fast_parse(text, path)

        if x is None:
            # Fall back to the robust line-by-line parser
            rows = self._parse_numeric_table(
                text=text,
                header_lines_to_skip=self.settings.header_lines_to_skip,
                debug=self.settings.debug,
                source_path=path,
            )
            x = np.asarray(rows, dtype=float)

        if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
            raise ValueError(f"Parsed invalid matrix shape {x.shape} from: {path}")

        x = self._maybe_drop_time_column(x)

        if self.settings.strict_T:
            expected_T = self._expected_T_from_protocol()
            if x.shape[0] != expected_T:
                raise ValueError(
                    f"Unexpected number of timepoints for {path.name}: got T={x.shape[0]} "
                    f"but expected T={expected_T} from protocol."
                )

        fs = float(self.config.protocol.fs)
        t = np.arange(x.shape[0], dtype=float) / fs
        return t, x

    def _try_fast_parse(self, text: str, path: Path) -> Optional[np.ndarray]:
        """Fast path: skip headers, parse with pandas. Returns None on failure."""
        import io as _io
        import pandas as _pd

        try:
            lines = text.split("\n")

            # Find the first numeric line (skip all header/metadata/empty)
            skip = 0
            for i, line in enumerate(lines):
                stripped = line.strip()
                if not stripped:
                    continue
                # Check if the line starts with a number (possibly negative)
                first_token = stripped.split("\t")[0].strip() if "\t" in stripped else stripped.split()[0].strip()
                first_token = first_token.replace(",", ".")
                try:
                    float(first_token)
                    skip = i
                    break
                except (ValueError, IndexError):
                    continue
            else:
                return None

            table_text = "\n".join(lines[skip:])
            if not table_text.strip():
                return None

            sep = "\t" if "\t" in table_text[:500] else r"\s+"

            df = _pd.read_csv(
                _io.StringIO(table_text),
                sep=sep,
                header=None,
                na_values=["", " ", "nan", "NaN"],
                on_bad_lines="skip",
                engine="c",
                dtype=float,
            )

            if df.shape[0] < 2 or df.shape[1] < 1:
                return None

            x = df.values.astype(float)
            return x

        except Exception:
            if self.settings.debug:
                print(f"[ImedosReader] Fast parse failed for {path}, falling back to slow parser")
            return None

    def _normalize_imedos_text(self, text: str) -> str:
        s = text
        s = s.replace(r"\tab", "\t")
        s = s.replace(r"\cell", "\t")
        s = s.replace(r"\par", "\n")
        s = s.replace(r"\line", "\n")
        s = re.sub(r"\n{2,}", "\n", s)
        return s

    def _looks_like_header_or_metadata(self, line: str) -> bool:
        s = line.strip()
        if s == "":
            return False
        s_low = s.lower()
        if s.startswith("{") or s.startswith("\\"):
            return True
        if "diameters complete" in s_low:
            return True
        if "-> position" in s_low:
            return True
        if "time" in s_low and re.search(r"[a-zA-Z]", s):
            return True
        return False

    def _parse_numeric_table(self, text, header_lines_to_skip, debug, source_path) -> List[List[float]]:
        skip = int(header_lines_to_skip)
        raw_lines: List[str] = []
        parsed_rows: List[List[float]] = []
        empty_row_flags: List[bool] = []
        started = False

        for line_no, raw_line in enumerate(text.splitlines(), start=1):
            if skip > 0:
                skip -= 1
                continue
            line = raw_line.rstrip("\r\n")
            line = self._normalize_line_tokens(line)

            if (not started) and self._looks_like_header_or_metadata(line):
                continue

            if self._is_effectively_empty_row(line):
                if not started:
                    continue
                raw_lines.append(raw_line)
                parsed_rows.append([])
                empty_row_flags.append(True)
                continue

            nums = self._parse_line_to_numbers(line)
            if len(nums) > 0:
                started = True
                raw_lines.append(raw_line)
                parsed_rows.append(nums)
                empty_row_flags.append(False)
                continue

            if not started:
                continue
            raw_lines.append(raw_line)
            parsed_rows.append([])
            empty_row_flags.append(True)

        if len(parsed_rows) < 2:
            raise ValueError(f"No rows parsed from: {source_path}")

        nonempty_lens = [
            len(r) for r, is_empty in zip(parsed_rows, empty_row_flags)
            if (not is_empty) and len(r) > 0
        ]
        if not nonempty_lens:
            raise ValueError(f"All parsed rows are empty after header skip in: {source_path}")

        expected_cols = int(np.bincount(np.array(nonempty_lens, dtype=int)).argmax())
        if expected_cols < 1:
            raise ValueError(f"Invalid expected column count inferred in: {source_path}")

        rows: List[List[float]] = []
        for i, nums in enumerate(parsed_rows):
            if len(nums) == 0:
                rows.append([float("nan")] * expected_cols)
                continue
            if len(nums) > expected_cols:
                extra = nums[expected_cols:]
                if all((not np.isfinite(v)) for v in extra):
                    nums = nums[:expected_cols]
            if len(nums) < expected_cols:
                nums = nums + [float("nan")] * (expected_cols - len(nums))
            if len(nums) == expected_cols:
                rows.append(nums)
                continue

            preview = raw_lines[i].rstrip("\r\n")
            if len(preview) > 260:
                preview = preview[:260] + " ..."
            msg = (
                "Row has inconsistent number of columns.\n"
                f"  File:     {source_path}\n"
                f"  Expected: {expected_cols}\n"
                f"  Got:      {len(parsed_rows[i])}\n"
                f"  Row idx:  {i} (after header skip)\n"
                f"  Line:     {preview}"
            )
            raise ValueError(msg)

        return rows

    def _normalize_line_tokens(self, line: str) -> str:
        line = re.sub(r"\\tab\b", "\t", line)
        line = re.sub(r"\\cell\b", "\t", line)
        line = re.sub(r"\\par\b", "", line)
        line = re.sub(r"\\line\b", "", line)
        return line

    def _is_effectively_empty_row(self, line: str) -> bool:
        if line == "":
            return True
        stripped = line.replace("\t", "").strip()
        return stripped == ""

    def _parse_line_to_numbers(self, line: str) -> List[float]:
        if "\t" in line:
            cells = line.split("\t")
            return [self._safe_float(c) for c in cells]
        if re.search(r"\s{2,}", line) or " " in line:
            parts = [p for p in re.split(r"\s+", line.strip())]
            return [self._safe_float(p) for p in parts]
        line2 = line.replace(",", ".")
        tokens = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line2)
        return [float(t) for t in tokens]

    def _safe_float(self, value: str) -> float:
        v = value.strip()
        if v == "":
            return float("nan")
        v = v.replace(",", ".")
        v = re.sub(r"[^\d\.\+\-eE]", "", v)
        if v == "" or v in {"+", "-", ".", "e", "E", "+e", "-e", "+E", "-E"}:
            return float("nan")
        try:
            return float(v)
        except ValueError:
            return float("nan")

    def _maybe_drop_time_column(self, x: np.ndarray) -> np.ndarray:
        if x.shape[1] < 2:
            return x
        col0 = x[:, 0]
        if float(np.mean(~np.isfinite(col0))) > 0.2:
            return x
        diffs = np.diff(col0)
        diffs = diffs[np.isfinite(diffs)]
        if diffs.size < 10:
            return x
        frac_pos = float(np.mean(diffs > 0))
        if frac_pos < 0.95:
            return x
        med = float(np.median(diffs))
        if med <= 0:
            return x
        mad = float(np.median(np.abs(diffs - med)))
        if mad < 0.05 * abs(med):
            return x[:, 1:]
        return x

    def _expected_T_from_protocol(self) -> int:
        end_sec = float(self.config.protocol.end_time_sec())
        fs = float(self.config.protocol.fs)
        return int(np.ceil(end_sec * fs))
