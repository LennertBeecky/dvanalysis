"""Parameter definitions — bind names to compute callables."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
from dvanalysis.domain import StimulusCycle, TimeWindow
from .library import (
    calc_dilation_max, calc_dilation_max_bc, calc_constr_max, calc_constr_max_bc,
    calc_da, calc_da_bc, calc_dilation_max_t, calc_dilation_max_t_bc,
    calc_dilation_max_30_t,
    calc_constr_max_t, calc_auc_dilation, calc_auc_constriction, calc_fic, calc_t_mdmc,
)


@dataclass(frozen=True)
class ParameterDefinition:
    """Defines one scalar parameter computed per cycle from a 1-D trace.

    Parameters
    ----------
    key : str
        Machine-readable identifier.
    label : str
        Human-readable label.
    description : str
        What this parameter measures.
    units_mode : str
        "absolute" or "percent".
    compute : callable
        (y, t, cycle) -> float
    """

    key: str
    label: str
    description: str
    units_mode: str
    compute: Callable[[np.ndarray, np.ndarray, StimulusCycle], float]


def make_default_definitions() -> Dict[str, ParameterDefinition]:
    return {
        "max_dilation": ParameterDefinition(
            key="max_dilation", label="Max dilation",
            description="Maximum value during flicker window",
            units_mode="absolute",
            compute=lambda y, t, cycle: calc_dilation_max(y, t, cycle.flicker),
        ),
        "max_constriction": ParameterDefinition(
            key="max_constriction", label="Max constriction",
            description="Minimum value during recovery window",
            units_mode="absolute",
            compute=lambda y, t, cycle: calc_constr_max(y, t, cycle.recovery),
        ),
        "dilation_amplitude": ParameterDefinition(
            key="dilation_amplitude", label="Dilation amplitude",
            description="Max during flicker minus min during recovery",
            units_mode="absolute",
            compute=lambda y, t, cycle: calc_da(y, t, cycle.flicker, cycle.recovery),
        ),
    }


def make_paper_definitions() -> Dict[str, ParameterDefinition]:
    """Feature set for clinical validation (8 parameters)."""
    return {
        "max_dilation": ParameterDefinition(
            key="max_dilation", label="Max dilation",
            description="Maximum value during flicker+10s window, baseline-corrected per cycle",
            units_mode="absolute",
            compute=lambda y, t, cycle: calc_dilation_max_bc(
                y, t,
                w_flicker=TimeWindow("flicker_ext", cycle.flicker.start_sec, cycle.flicker.end_sec + 10.0),
                w_baseline=cycle.baseline,
            ),
        ),
        "max_constriction": ParameterDefinition(
            key="max_constriction", label="Max constriction",
            description="Minimum value during recovery window, baseline-corrected per cycle",
            units_mode="absolute",
            compute=lambda y, t, cycle: calc_constr_max_bc(y, t, w_recovery=cycle.recovery, w_baseline=cycle.baseline),
        ),
        "dilation_amplitude": ParameterDefinition(
            key="dilation_amplitude", label="Dilation amplitude",
            description="Max during flicker+10s minus min during recovery, baseline-corrected",
            units_mode="absolute",
            compute=lambda y, t, cycle: calc_da_bc(
                y, t,
                TimeWindow("flicker_ext", cycle.flicker.start_sec, cycle.flicker.end_sec + 10.0),
                cycle.recovery, w_baseline=cycle.baseline,
            ),
        ),
        "time_to_max_dilation": ParameterDefinition(
            key="time_to_max_dilation", label="Time to max dilation",
            description="Time from flicker onset to maximum dilation, baseline-corrected",
            units_mode="absolute",
            compute=lambda y, t, cycle: calc_dilation_max_t_bc(
                y, t,
                w_flicker=TimeWindow("flicker_ext", cycle.flicker.start_sec, cycle.flicker.end_sec + 10.0),
                w_baseline=cycle.baseline, relative_to_start=True,
            ),
        ),
        "time_to_max_constriction": ParameterDefinition(
            key="time_to_max_constriction", label="Time to max constriction",
            description="Time from flicker onset to minimum diameter in recovery",
            units_mode="absolute",
            compute=lambda y, t, cycle: calc_constr_max_t(y, t, cycle.recovery, ref_time_sec=cycle.flicker.start_sec),
        ),
        "t_mdmc": ParameterDefinition(
            key="t_mdmc", label="t(MaxC) - t(MaxD)",
            description="Time difference between max constriction and max dilation",
            units_mode="absolute",
            compute=lambda y, t, cycle: calc_t_mdmc(y, t, cycle.flicker, cycle.recovery),
        ),
        "time_to_max_dilation_30": ParameterDefinition(
            key="time_to_max_dilation_30", label="tMAD30",
            description="Time from flicker onset to 30% of max dilation on rising phase",
            units_mode="absolute",
            compute=lambda y, t, cycle: calc_dilation_max_30_t(
                y, t,
                w_search=TimeWindow("flicker_ext", cycle.flicker.start_sec, cycle.flicker.end_sec + 10.0),
                w_baseline=cycle.baseline,
            ),
        ),
        "auc_dilation": ParameterDefinition(
            key="auc_dilation", label="AUC dilation",
            description="Integral of max(y - baseline, 0) over flicker window",
            units_mode="absolute",
            compute=lambda y, t, cycle: calc_auc_dilation(y, t, w_baseline=cycle.baseline, w_flicker=cycle.flicker),
        ),
        "auc_constriction": ParameterDefinition(
            key="auc_constriction", label="AUC constriction",
            description="Integral of max(baseline - y, 0) over recovery window",
            units_mode="absolute",
            compute=lambda y, t, cycle: calc_auc_constriction(y, t, w_baseline=cycle.baseline, w_recovery=cycle.recovery),
        ),
    }
