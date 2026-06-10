"""Hybrid ground-truth sample assembly for DVA denoising validation.

Combines a known synthetic response with real baseline residual
(cardiac pulsatility + noise) and a constructed observation mask
from an actual DVA recording.

The baseline residual is extracted from all non-flicker baseline epochs,
preserving the full temporal structure of real DVA noise without separating
cardiac from non-cardiac components. The observation mask is constructed
by tiling the baseline missingness pattern and adding deterministic
flicker dropout (every other frame during 12.5 Hz flicker at 25 Hz sampling).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np

from dvanalysis.domain import SegmentSignal, TimeWindow
from dvanalysis.validation.templates import (
    ArterialParams,
    VenousParams,
    arterial_template,
    venous_template,
    sample_arterial_params,
    sample_venous_params,
    apply_intercycle_jitter,
)
from dvanalysis.validation.noise import (
    extract_baseline_residual,
    tile_residual,
    build_flicker_mask,
)


@dataclass
class SparseArtefactConfig:
    """Configuration for synthetic sparse artefact injection.

    Artefacts are column-sparse: each event affects ALL loci at the same
    timepoint(s), matching real DVA where blinks and saccades displace
    the entire fundus image simultaneously.

    Two types:
    1. Impulses: single-frame spikes (microsaccades). All loci get the
       same base amplitude with per-locus jitter.
    2. Bursts: contiguous multi-frame events (blinks, fixation loss).
       All loci affected for the same duration.

    Amplitudes calibrated from empirical RPCA sparse component on real
    DVA recordings (>3sigma entries: median 19 MU, 95th 37 MU).
    """

    enabled: bool = True

    # Impulse events (microsaccades): single frame, all loci
    impulse_rate_per_min: float = 2.0     # events per minute (not per locus)
    impulse_amp_mean: float = 15.0        # base amplitude in MU (~12% baseline)
    impulse_amp_sd: float = 8.0
    impulse_sign_prob_pos: float = 0.6
    impulse_locus_jitter: float = 0.15    # per-locus amplitude jitter (relative)

    # Burst events (blinks, fixation loss): multi-frame, all loci
    burst_rate_per_min: float = 0.8       # events per minute
    burst_amp_mean: float = 12.0          # base amplitude in MU
    burst_amp_sd: float = 6.0
    burst_dur_min_frames: int = 3         # minimum duration in frames
    burst_dur_max_frames: int = 15        # maximum duration (~0.6s at 25 Hz)
    burst_sign_prob_pos: float = 0.6
    burst_locus_jitter: float = 0.10


def inject_sparse_artefacts(
    t: np.ndarray,
    P: int,
    rng: np.random.Generator,
    cfg: SparseArtefactConfig,
) -> np.ndarray:
    """Generate column-sparse artefact matrix A(T, P).

    Each artefact event affects ALL loci at the same timepoint(s),
    with small per-locus amplitude jitter.

    Returns
    -------
    (T, P) artefact matrix. Zero where no artefact.
    """
    T = len(t)
    if not cfg.enabled or P <= 0 or T <= 0:
        return np.zeros((T, P), dtype=float)

    dur_sec = float(t[-1] - t[0]) if T > 1 else 0.0
    dur_min = max(1e-9, dur_sec / 60.0)

    A = np.zeros((T, P), dtype=float)

    # Impulse events: single frame, all loci
    n_imp = int(rng.poisson(cfg.impulse_rate_per_min * dur_min))
    for _ in range(n_imp):
        idx = int(rng.integers(0, T))
        base_amp = max(0.0, float(rng.normal(cfg.impulse_amp_mean, cfg.impulse_amp_sd)))
        sign = 1.0 if float(rng.random()) < cfg.impulse_sign_prob_pos else -1.0
        # Per-locus jitter around the base amplitude
        locus_amps = base_amp * (1.0 + rng.normal(0, cfg.impulse_locus_jitter, size=P))
        locus_amps = np.maximum(0.0, locus_amps)
        A[idx, :] += sign * locus_amps

    # Burst events: contiguous frames, all loci
    n_burst = int(rng.poisson(cfg.burst_rate_per_min * dur_min))
    for _ in range(n_burst):
        start = int(rng.integers(0, T))
        dur_frames = int(rng.integers(cfg.burst_dur_min_frames, cfg.burst_dur_max_frames + 1))
        end = min(T, start + dur_frames)
        base_amp = max(0.0, float(rng.normal(cfg.burst_amp_mean, cfg.burst_amp_sd)))
        sign = 1.0 if float(rng.random()) < cfg.burst_sign_prob_pos else -1.0
        # Per-locus jitter
        locus_amps = base_amp * (1.0 + rng.normal(0, cfg.burst_locus_jitter, size=P))
        locus_amps = np.maximum(0.0, locus_amps)
        A[start:end, :] += sign * locus_amps[np.newaxis, :]

    return A


@dataclass
class HybridSample:
    """One hybrid ground-truth sample.

    Attributes
    ----------
    signal : SegmentSignal
        The assembled hybrid signal (response + residual, with constructed mask).
    ground_truth : (T,) ndarray
        The known vessel-level response trace in percent change from baseline.
    params : dict
        Parameters used to generate this sample.
    """

    signal: SegmentSignal
    ground_truth: np.ndarray
    params: Dict[str, Any] = field(default_factory=dict)


def build_hybrid(
    sig: SegmentSignal,
    vessel_type: str,
    params: Union[ArterialParams, VenousParams],
    rng: np.random.Generator,
    cycle_jitter: bool = True,
    locus_variability: float = 0.0,
    locus_jitter_samples: int = 2,
    sparse_cfg: Optional[SparseArtefactConfig] = None,
) -> HybridSample:
    """Build one hybrid sample from a real recording and synthetic parameters.

    Parameters
    ----------
    sig : SegmentSignal
        The real recording (used for baseline residual and resting diameter).
    vessel_type : "artery" or "vein"
    params : response template parameters.
    rng : random number generator.
    cycle_jitter : apply per-cycle amplitude/timing jitter.
    locus_variability : float, default 0.0
        Std of per-locus amplitude scaling (0 = rank-1).
    locus_jitter_samples : int, default 2
        Max per-locus onset jitter in samples (0 = no jitter).
    sparse_cfg : SparseArtefactConfig, optional
        Configuration for synthetic sparse artefact injection.
        Default: SparseArtefactConfig() (enabled with default rates).

    Returns
    -------
    HybridSample with the assembled signal and known ground truth.
    """
    t = np.asarray(sig.t, float)
    T, P = sig.T, sig.P
    protocol = sig.protocol

    # Collect all baseline windows (global + pre-flicker per cycle)
    baseline_windows = []
    if protocol.global_baseline is not None:
        baseline_windows.append(protocol.global_baseline)
    for cyc in protocol.cycles:
        if cyc.baseline is not None:
            baseline_windows.append(cyc.baseline)

    # 1. Generate synthetic response (percent change from baseline)
    if cycle_jitter and protocol.cycles:
        cycle_params = [apply_intercycle_jitter(params, rng) for _ in protocol.cycles]
    else:
        cycle_params = None

    if vessel_type.lower().startswith("a"):
        gt_pct = arterial_template(t, protocol, params, cycle_params=cycle_params)
    else:
        gt_pct = venous_template(t, protocol, params, cycle_params=cycle_params)

    # 2. Extract real baseline residual (cardiac + noise, no separation)
    residual_block = extract_baseline_residual(sig, baseline_windows)

    # Per-locus resting diameter (median across all baselines)
    x_real = np.asarray(sig.x, float).copy()
    m_real = np.asarray(sig.m, bool)
    x_real[~m_real] = np.nan
    all_bl_idx = []
    for w in baseline_windows:
        all_bl_idx.append(np.where(w.contains(t))[0])
    all_bl_idx = np.concatenate(all_bl_idx)
    baselines = np.nanmedian(x_real[all_bl_idx, :], axis=0)  # (P,)

    # Tile residual to full recording length (NaN preserved)
    residual_tiled = tile_residual(residual_block, target_length=T, rng=rng)

    # 3. Per-locus response variation
    if locus_variability > 0.0:
        locus_scale = 1.0 + rng.normal(0, locus_variability, size=P)
    else:
        locus_scale = np.ones(P)

    gt_pct_loci = gt_pct[:, np.newaxis] * locus_scale[np.newaxis, :]

    locus_shifts = np.zeros(P, dtype=int)
    if locus_jitter_samples > 0:
        locus_shifts = rng.integers(
            -locus_jitter_samples, locus_jitter_samples + 1, size=P
        )
        for p in range(P):
            s = int(locus_shifts[p])
            if s != 0:
                gt_pct_loci[:, p] = np.roll(gt_pct_loci[:, p], s)
                # Zero the wrapped positions to prevent edge artifacts
                if s > 0:
                    gt_pct_loci[:s, p] = 0.0
                else:
                    gt_pct_loci[s:, p] = 0.0

    # 4. Assemble hybrid signal
    response_factor = 1.0 + gt_pct_loci / 100.0  # (T, P)
    x_hybrid = baselines[np.newaxis, :] * response_factor + residual_tiled

    # 5. Construct observation mask (before adding sparse artefacts,
    #    so baseline estimation for ground truth is clean)
    # Residual mask: True where the tiled residual has a valid (non-NaN) value
    residual_mask = np.isfinite(residual_tiled)  # (T, P)

    # Flicker mask: False at dark frames during flicker periods
    flicker_mask_1d = build_flicker_mask(t, protocol, fs=protocol.fs)
    # Broadcast to (T, P)
    flicker_mask = flicker_mask_1d[:, np.newaxis] & np.ones(P, dtype=bool)[np.newaxis, :]

    # Final mask = residual has data AND not a flicker dark frame
    m_hybrid = residual_mask & flicker_mask

    # Apply mask (before artefacts, so baseline estimation is clean)
    x_hybrid[~m_hybrid] = np.nan

    # 6. Ground truth: the clean template s(t) in percent change from baseline.
    # With locus_variability=0 (default), all loci share the same response,
    # so s(t) is the unambiguous ground truth regardless of reference frame
    # or which loci are observed. NaN only at timepoints with no observed loci.
    if locus_variability > 0.0 or locus_jitter_samples > 0:
        ground_truth = np.full(T, np.nan)
        for i_t in range(T):
            valid = m_hybrid[i_t, :]
            if valid.sum() > 0:
                ground_truth[i_t] = np.nanmedian(gt_pct_loci[i_t, valid])
    else:
        ground_truth = gt_pct.copy()
        no_obs = m_hybrid.sum(axis=1) == 0
        ground_truth[no_obs] = np.nan

    # 7. Optionally inject synthetic sparse artefacts AFTER ground truth computation
    if sparse_cfg is None:
        sparse_cfg = SparseArtefactConfig(enabled=False)
    A_true = inject_sparse_artefacts(t, P, rng, sparse_cfg)
    if sparse_cfg.enabled:
        x_hybrid[m_hybrid] += A_true[m_hybrid]

    # 8. Create SegmentSignal
    hybrid_sig = SegmentSignal(
        t=t, x=x_hybrid, m=m_hybrid, protocol=protocol,
        units=sig.units, meta={**sig.meta, "hybrid": True},
    )

    return HybridSample(
        signal=hybrid_sig,
        ground_truth=ground_truth,
        params={
            "vessel_type": vessel_type,
            "template_params": params.__dict__,
            "cycle_jitter": cycle_jitter,
            "gt_pct_clean": gt_pct.copy(),
            "locus_variability": locus_variability,
            "locus_scale": locus_scale,
            "locus_jitter_samples": locus_jitter_samples,
            "locus_shifts": locus_shifts,
            "sparse_cfg": sparse_cfg.__dict__ if sparse_cfg.enabled else {"enabled": False},
            "sparse_n_nonzero": int(np.count_nonzero(A_true)),
        },
    )


class HybridBuilder:
    """Generate multiple hybrid samples per segment."""

    def __init__(self, n_configs: int = 10, seed: int = 42,
                 locus_variability: float = 0.0):
        self.n_configs = n_configs
        self.locus_variability = locus_variability
        self.rng = np.random.default_rng(seed)

    def build_for_segment(
        self, sig: SegmentSignal, vessel_type: str
    ) -> List[HybridSample]:
        samples = []
        for config_idx in range(self.n_configs):
            if vessel_type.lower().startswith("a"):
                params = sample_arterial_params(self.rng)
            else:
                params = sample_venous_params(self.rng)

            hs = build_hybrid(
                sig, vessel_type, params, rng=self.rng,
                cycle_jitter=True, locus_variability=self.locus_variability,
                locus_jitter_samples=2,
            )
            hs.params["config_index"] = config_idx
            samples.append(hs)

        return samples
