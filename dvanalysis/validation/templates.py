"""Synthetic response templates for DVA hybrid ground-truth construction.

Templates use PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)
interpolation on control points digitised from real DVA flicker responses.
The normalised shape is fixed per vessel type; only amplitude, constriction
depth and small timing perturbations are varied across configurations.

Arterial: gradual curved rise to plateau around 15-18 s, decay after flicker
          offset, constriction undershoot below baseline, gradual recovery.
Venous:   biphasic rise (fast initial, slower approach to peak), peak near
          flicker cessation (~22 s), gradual decay. No constriction.

Parameter ranges from Streese et al. 2021 normative data:
  - Arterial aFID: 3.74 +/- 2.17% -> range 1-6%
  - Venous vFID: 4.64 +/- 1.85% -> range 2-7%
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.interpolate import PchipInterpolator

from dvanalysis.domain import StimulusProtocol


# ---------------------------------------------------------------------------
# Normalised response shapes (digitised from real DVA population means)
# ---------------------------------------------------------------------------
# Time is relative to flicker onset (s). Amplitude is normalised so that the
# dilation peak equals 1.0.
#
# The arterial response is decomposed into two smooth components:
#   - dilation envelope: 0 -> 1 -> 0 (always >= 0)
#   - constriction envelope: 0 -> 1 -> 0 (always >= 0, peaks during recovery)
# The full response is: dilation_amplitude * dilation - constriction_depth * constriction
# This avoids derivative discontinuities at y=0.

_ART_DIL_T = np.array(
    [-15, -10, -5, -1, 0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 20,
     21, 22, 23, 24, 25, 26, 27, 28, 30, 33, 37, 42, 50, 55],
    dtype=float,
)
_ART_DIL_Y = np.array(
    [0, 0, 0, 0, 0, 0.04, 0.14, 0.30, 0.50, 0.68, 0.82, 0.91, 0.96, 0.99, 1.0, 1.0,
     0.98, 0.93, 0.85, 0.72, 0.55, 0.38, 0.22, 0.10, 0.02, 0, 0, 0, 0, 0],
    dtype=float,
)

_ART_CON_T = np.array(
    [-15, -10, 0, 20, 23, 25, 27, 29, 31, 33, 35, 37, 40, 44, 48, 50, 55],
    dtype=float,
)
_ART_CON_Y = np.array(
    [0, 0, 0, 0, 0.02, 0.10, 0.30, 0.60, 0.85, 1.0, 0.95, 0.82, 0.55, 0.25, 0.08, 0.02, 0],
    dtype=float,
)

_VEN_SHAPE_T = np.array(
    [-15, -10, -5, -1, 0, 1, 3, 5, 8, 10, 13, 16, 18, 20, 22,
     24, 26, 28, 30, 35, 40, 45, 50, 55],
    dtype=float,
)
_VEN_SHAPE_Y = np.array(
    [0, 0, 0, 0, 0, 0.10, 0.25, 0.38, 0.53, 0.63, 0.75, 0.85, 0.93, 0.98, 1.0,
     0.88, 0.70, 0.53, 0.35, 0.10, 0.01, 0, 0, 0],
    dtype=float,
)

# Pre-compute the normalised interpolators
_ART_DIL_SPLINE = PchipInterpolator(_ART_DIL_T, _ART_DIL_Y)
_ART_CON_SPLINE = PchipInterpolator(_ART_CON_T, _ART_CON_Y)
_VEN_SPLINE = PchipInterpolator(_VEN_SHAPE_T, _VEN_SHAPE_Y)


# ---------------------------------------------------------------------------
# Parameter dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ArterialParams:
    """Parameters for one arterial flicker response cycle."""

    dilation_amplitude: float    # % above baseline (1-6)
    constriction_depth: float    # % below baseline (0.5-3)
    time_shift: float = 0.0     # seconds to shift the whole response (small jitter)


@dataclass
class VenousParams:
    """Parameters for one venous flicker response cycle."""

    dilation_amplitude: float    # % above baseline (2-7)
    time_shift: float = 0.0     # seconds to shift the whole response (small jitter)


# ---------------------------------------------------------------------------
# Arterial template
# ---------------------------------------------------------------------------

def arterial_template(
    t: np.ndarray,
    protocol: StimulusProtocol,
    params: ArterialParams,
    cycle_params: Optional[list[ArterialParams]] = None,
) -> np.ndarray:
    """Generate an arterial flicker response trace in percent change from baseline.

    The normalised shape is derived from population-mean responses in real
    DVA recordings. Only amplitude, constriction depth and timing are varied.

    Parameters
    ----------
    t : (T,) time vector in seconds.
    protocol : stimulus protocol with cycles.
    params : default parameters (used if cycle_params is None).
    cycle_params : optional per-cycle parameter overrides.

    Returns
    -------
    (T,) response trace in % change from baseline.
    """
    y = np.zeros_like(t, dtype=float)

    for i, cyc in enumerate(protocol.cycles):
        p = cycle_params[i] if cycle_params is not None else params
        fl_start = cyc.flicker.start_sec

        t_rel = t - fl_start - p.time_shift

        # Clamp to spline domain to prevent extrapolation artifacts
        t_dil = np.clip(t_rel, _ART_DIL_T[0], _ART_DIL_T[-1])
        t_con = np.clip(t_rel, _ART_CON_T[0], _ART_CON_T[-1])
        dil = _ART_DIL_SPLINE(t_dil)
        con = _ART_CON_SPLINE(t_con)
        # Zero outside domains
        outside = (t_rel < _ART_DIL_T[0]) | (t_rel > _ART_DIL_T[-1])
        dil[outside] = 0.0
        con[outside] = 0.0
        # Ensure non-negative envelopes
        dil = np.clip(dil, 0, None)
        con = np.clip(con, 0, None)

        cycle_response = p.dilation_amplitude * dil - p.constriction_depth * con

        # Window
        window_start = fl_start - 2.0
        window_end = cyc.recovery.end_sec
        active = (t >= window_start) & (t < window_end)
        y[active] += cycle_response[active]

    return y


# ---------------------------------------------------------------------------
# Venous template
# ---------------------------------------------------------------------------

def venous_template(
    t: np.ndarray,
    protocol: StimulusProtocol,
    params: VenousParams,
    cycle_params: Optional[list[VenousParams]] = None,
) -> np.ndarray:
    """Generate a venous flicker response trace in percent change from baseline.

    The normalised shape is derived from population-mean responses in real
    DVA recordings. Only amplitude and timing are varied.

    Parameters
    ----------
    t : (T,) time vector in seconds.
    protocol : stimulus protocol with cycles.
    params : default parameters.
    cycle_params : optional per-cycle overrides.

    Returns
    -------
    (T,) response trace in % change from baseline.
    """
    y = np.zeros_like(t, dtype=float)

    for i, cyc in enumerate(protocol.cycles):
        p = cycle_params[i] if cycle_params is not None else params
        fl_start = cyc.flicker.start_sec

        t_rel = t - fl_start - p.time_shift

        # Clamp to spline domain to prevent extrapolation artifacts
        t_clamped = np.clip(t_rel, _VEN_SHAPE_T[0], _VEN_SHAPE_T[-1])
        cycle_response = _VEN_SPLINE(t_clamped) * p.dilation_amplitude
        cycle_response[(t_rel < _VEN_SHAPE_T[0]) | (t_rel > _VEN_SHAPE_T[-1])] = 0.0
        cycle_response = np.clip(cycle_response, 0, None)

        # Window
        window_start = fl_start - 2.0
        window_end = cyc.recovery.end_sec
        active = (t >= window_start) & (t < window_end)
        y[active] += cycle_response[active]

    return y


# ---------------------------------------------------------------------------
# Parameter sampling
# ---------------------------------------------------------------------------

def sample_arterial_params(rng: np.random.Generator) -> ArterialParams:
    """Sample arterial response parameters from physiologically plausible ranges.

    Constriction depth (absolute, in %) is inversely correlated with
    dilation amplitude: weak dilators show deep constriction, strong
    dilators show shallow constriction.
    """
    amp = rng.uniform(1.0, 6.0)
    # Inverse relationship in absolute terms:
    # At amp=1: constriction ~2.5-3.5%
    # At amp=6: constriction ~0.5-1.5%
    constr_mean = 3.5 - 0.5 * (amp - 1.0)  # 3.5 at amp=1, 1.0 at amp=6
    constr = max(0.3, constr_mean + rng.normal(0, 0.3))
    return ArterialParams(
        dilation_amplitude=amp,
        constriction_depth=constr,
        time_shift=rng.uniform(-2.0, 2.0),
    )


def sample_venous_params(rng: np.random.Generator) -> VenousParams:
    """Sample venous response parameters from physiologically plausible ranges."""
    return VenousParams(
        dilation_amplitude=rng.uniform(2.0, 7.0),
        time_shift=rng.uniform(-2.0, 2.0),
    )


def apply_intercycle_jitter(
    params: ArterialParams | VenousParams,
    rng: np.random.Generator,
    amplitude_cv: float = 0.05,
    timing_cv: float = 0.05,
) -> ArterialParams | VenousParams:
    """Apply per-cycle jitter to response parameters.

    Parameters
    ----------
    params : base parameters.
    rng : random number generator.
    amplitude_cv : coefficient of variation for amplitude jitter.
    timing_cv : coefficient of variation for timing jitter.

    Returns
    -------
    New parameter instance with jittered values.
    """
    def _jitter_amp(val: float) -> float:
        return max(0.01, val * (1.0 + rng.normal(0, amplitude_cv)))

    if isinstance(params, ArterialParams):
        return ArterialParams(
            dilation_amplitude=_jitter_amp(params.dilation_amplitude),
            constriction_depth=_jitter_amp(params.constriction_depth),
            time_shift=params.time_shift + rng.normal(0, 0.5),
        )
    elif isinstance(params, VenousParams):
        return VenousParams(
            dilation_amplitude=_jitter_amp(params.dilation_amplitude),
            time_shift=params.time_shift + rng.normal(0, 0.5),
        )
    else:
        raise TypeError(f"Unknown params type: {type(params)}")
