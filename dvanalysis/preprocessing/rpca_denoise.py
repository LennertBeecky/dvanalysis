# preprocessing/rpca_denoise.py
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np

from dvanalysis.domain import SegmentSignal
from .base import BasePreprocessor
from .configs import PreprocessorConfig
from .result import PreprocessResult

from .rpca import RobustPCA
from .filtering import extract_heartbeat_psd, lowpass_filter_masked
from .centering import baseline_center_and_scale
from .aggregation import aggregate_loci

from dvanalysis.utils.math import fill_missing_interp, hampel_filter_1d, to_percent
from dvanalysis.utils.masks import support_good_timepoints, resolve_min_valid_abs


@dataclass(frozen=True)
class MyMethodConfig(PreprocessorConfig):
    """Physiology informed denoising for DVA segment signals.

    Pipeline
      1) Optional missing value fill per locus (linear interpolation)
      2) Optional heartbeat low pass filtering per locus (mask aware)
      3) Baseline aware centering and robust scaling (standardization)
      4) Masked smooth RPCA: low rank physiology plus sparse artefacts
      5) Vessel trace via robust locus aggregation
      6) Optional harmonisation to percent change from baseline

    Important:
      - Standardization is applied BEFORE RPCA when `standardize=True`.
      - By default we DESTANDARDIZE after RPCA (return to a.u.).
      - If you want to KEEP the standardized outputs (unitless z-ish),
        set `return_standardized_outputs=True`. In that mode:
          * S_hat / A_hat are NOT reconverted back
          * s_hat is the aggregated standardized physiology trace
          * harmonize_output is forcibly disabled (percent needs absolute baseline)
    """

    # General
    strict: bool = False
    debug: bool = False

    # Timing
    timestep_ms: float = 40.0

    # Missing value handling
    fill_missing: bool = False  # fill raw X before any processing

    # Heartbeat filtering (mask aware low pass)
    heartbeat_filter: bool = False
    extra_cutoff_hz: float = 1.1 / 2.0
    lp_order: int = 4

    # Heartbeat filter stability knobs
    lp_min_block_len: Optional[int] = None
    lp_padlen: Optional[int] = None
    lp_taper_len: int = 0

    # Baseline aware standardization
    standardize: bool = True
    standardize_scale_mode: str = "global"  # global or per_locus
    standardize_scale_floor: float = 1e-6

    # >>> NEW: keep standardized space outputs (do NOT reconvert to absolute units)
    return_standardized_outputs: bool = False

    # RPCA hyperparameters
    rpca_lmb: float = 1e-2
    rpca_mu: float = 1.0
    rpca_rho: float = 5.0
    rpca_gamma: float = 0.0
    rpca_max_iter: int = 200
    rpca_tol_rel: float = 1e-4

    # Optional saving of intermediate matrices
    rpca_path: str = ""
    rpca_save_every: int = 0

    # Aggregation to produce a vessel trace from loci
    locus_agg: str = "median"  # mean or median

    # Harmonisation for benchmarking against normalized vessel diameter curves
    harmonize_output: bool = True
    harmonize_percent_mode: str = "delta_over_baseline"  # delta_over_baseline or ratio

    harmonize_baseline_source: str = "protocol_global"  # protocol_global | cycle0_baseline | custom
    harmonize_custom_baseline_start_sec: float = 0.0
    harmonize_custom_baseline_end_sec: float = 30.0

    harmonize_aggregation_order: str = "aggregate_then_percent"  # aggregate_then_percent | percent_then_aggregate
    harmonize_baseline_per_locus: bool = True

    # Against the peaks in the averaged trace
        # Against peaks in the averaged trace (support gate)
    support_min_valid_abs: int = 12          # absolute minimum loci
    support_min_valid_frac: Optional[float] = None  # e.g. 0.75 means 75% of loci

    # Hint for RPCA logging and saved matrices naming
    artery_hint: bool = True

    # 1D safety net (after aggregation + support gate)
    hampel_enable: bool = False
    hampel_window: int = 9          # half-window size k; total window = 2k+1
    hampel_nsigmas: float = 3.0     # threshold in MAD units
    hampel_replace: str = "median"  # "median" or "nan"


class MyMethodRPCA(BasePreprocessor[MyMethodConfig]):
    name: str = "RPCA"
    version: str = "0.9"  # added return_standardized_outputs

    def run(self, signal: SegmentSignal) -> PreprocessResult:
        cfg = self.config

        X = signal.x.astype(float).copy()  # (T,P)
        M = signal.m.astype(bool).copy()   # (T,P)

        # Optional fill (legacy convenience)
        if cfg.fill_missing:
            X = self._fill_missing(X, M)
            M = np.isfinite(X)

        # Baseline indices from protocol, used for scaling and harmonisation
        b0_idx, baseline_info = self._get_baseline_indices(signal, cfg)

        # Optional heartbeat driven low pass
        hb_bpm, hb_amp = self._estimate_heartbeat(X, M, timestep_ms=cfg.timestep_ms)
        if not cfg.heartbeat_filter:
            X_lp = X.copy()
            X_hi = np.full_like(X, np.nan, dtype=float)
            cutoff_hz = float("nan")
        else:
            if not np.isfinite(hb_bpm):
                msg = "Heartbeat bpm not finite, cannot derive low pass cutoff"
                if cfg.strict:
                    raise ValueError(msg)
                if cfg.debug:
                    print(f"[{self.name}] {msg}. Continuing without heartbeat filtering.")
                X_lp = X.copy()
                X_hi = np.full_like(X, np.nan, dtype=float)
                cutoff_hz = float("nan")
            else:
                X_lp, X_hi, cutoff_hz = self._apply_lowpass(
                    X, M, hb_bpm,
                    timestep_ms=cfg.timestep_ms,
                    extra_cutoff_hz=cfg.extra_cutoff_hz,
                    lp_order=cfg.lp_order,
                    min_block_len=cfg.lp_min_block_len,
                    padlen=cfg.lp_padlen,
                    taper_len=cfg.lp_taper_len,
                )

        # Masked smooth RPCA
        S_hat, A_hat, rpca_diag = self._run_rpca_masked_smooth(
            X_lp,
            M,
            standardize=cfg.standardize,
            destandardize=not bool(cfg.return_standardized_outputs),
            rpca_kwargs={
                "lmb": cfg.rpca_lmb,
                "mu": cfg.rpca_mu,
                "rho": cfg.rpca_rho,
                "gamma": cfg.rpca_gamma,
                "max_iter": cfg.rpca_max_iter,
                "tol_rel": cfg.rpca_tol_rel,
                "path": cfg.rpca_path,
                "artery": bool(cfg.artery_hint),
                "save_every": int(cfg.rpca_save_every),
                "baseline_idx": b0_idx,
                "scale_mode": str(cfg.standardize_scale_mode),
                "scale_floor": float(cfg.standardize_scale_floor),
            },
        )

        # confidence is computed in the SAME space as S_hat/A_hat
        conf_diag = self._confidence_per_cycle(signal=signal, S_hat=S_hat, A_hat=A_hat, M=M)

        # Vessel level trace from physiology matrix
        s_hat_abs = self._aggregate_loci(S_hat, M, mode=cfg.locus_agg)

        good_t = np.ones(S_hat.shape[0], dtype=bool)

        min_valid_abs = self._resolve_min_valid_abs(
            M,
            min_abs=getattr(cfg, "support_min_valid_abs", 0),
            min_frac=getattr(cfg, "support_min_valid_frac", None),
        )

        if min_valid_abs > 0:
            good_t = self._support_good_timepoints(M, min_valid_abs)
            s_hat_abs = np.asarray(s_hat_abs, float).copy()
            s_hat_abs[~good_t] = np.nan

        # If we keep standardized outputs: do NOT do percent harmonization (needs absolute baseline)
        if cfg.return_standardized_outputs:
            s_hat = s_hat_abs
            harm_diag = {
                "harmonize_output": False,
                "units_out": "standardized",
                "return_standardized_outputs": True,
                **baseline_info,
            }
        else:
            if cfg.harmonize_output:
                s_hat, harm_diag = self._harmonize_s_hat(
                    signal=signal, L=S_hat, M=M, s_hat_abs=s_hat_abs, cfg=cfg
                )
            else:
                s_hat = s_hat_abs
                harm_diag = {
                    "harmonize_output": False,
                    "units_out": getattr(signal, "units", "a.u."),
                    "return_standardized_outputs": False,
                    **baseline_info,
                }

        if good_t is not None:
            s_hat = np.asarray(s_hat, float).copy()
            s_hat[~good_t] = np.nan

        hampel_diag = {}
        if getattr(cfg, "hampel_enable", False):
            s_hat_f, is_out = self._hampel_filter_1d(
                s_hat,
                k=int(cfg.hampel_window),
                nsigmas=float(cfg.hampel_nsigmas),
                replace=str(cfg.hampel_replace),
            )
            hampel_diag = {
                "hampel_enable": True,
                "hampel_window": int(cfg.hampel_window),
                "hampel_nsigmas": float(cfg.hampel_nsigmas),
                "hampel_replace": str(cfg.hampel_replace),
                "hampel_n_outliers": int(np.count_nonzero(is_out)),
            }
            s_hat = s_hat_f
        else:
            hampel_diag = {"hampel_enable": False}

        diagnostics: Dict[str, object] = {
            "heartbeat_bpm": float(hb_bpm),
            "heartbeat_amp": float(hb_amp),
            "heartbeat_filter": bool(cfg.heartbeat_filter),
            "cutoff_hz": float(cutoff_hz),
            "fill_missing": bool(cfg.fill_missing),
            "standardize": bool(cfg.standardize),
            "return_standardized_outputs": bool(cfg.return_standardized_outputs),
            "standardize_scale_mode": str(cfg.standardize_scale_mode),
            "standardize_scale_floor": float(cfg.standardize_scale_floor),
            "lp_order": int(cfg.lp_order),
            "lp_min_block_len": cfg.lp_min_block_len,
            "lp_padlen": cfg.lp_padlen,
            "lp_taper_len": int(cfg.lp_taper_len),
            "locus_agg": str(cfg.locus_agg),
            "rpca_max_iter": int(cfg.rpca_max_iter),
            "rpca_lmb": float(cfg.rpca_lmb),
            "rpca_mu": float(cfg.rpca_mu),
            "rpca_rho": float(cfg.rpca_rho),
            "rpca_gamma": float(cfg.rpca_gamma),
            **baseline_info,
            **rpca_diag,
            **conf_diag,
            **harm_diag,
            **hampel_diag
        }

        return PreprocessResult(
            preprocessor_name=self.name,
            preprocessor_version=self.version,
            config=asdict(cfg),
            input_signal=signal,
            s_hat=s_hat,
            m_used=M,
            S_hat=S_hat,
            A_hat=A_hat,
            H_hat=X_hi,
            diagnostics=diagnostics,
        )

    # ------------------------------------------------------------------
    # Harmonisation helpers
    # ------------------------------------------------------------------
    def _harmonize_s_hat(
        self,
        *,
        signal: SegmentSignal,
        L: np.ndarray,
        M: np.ndarray,
        s_hat_abs: np.ndarray,
        cfg: MyMethodConfig,
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        b0_idx, baseline_info = self._get_baseline_indices(signal, cfg)

        order = str(cfg.harmonize_aggregation_order).lower().strip()
        if order not in {"aggregate_then_percent", "percent_then_aggregate"}:
            raise ValueError("harmonize_aggregation_order must be 'aggregate_then_percent' or 'percent_then_aggregate'.")

        percent_mode = str(cfg.harmonize_percent_mode).lower().strip()
        if percent_mode not in {"delta_over_baseline", "ratio"}:
            raise ValueError("harmonize_percent_mode must be 'delta_over_baseline' or 'ratio'.")

        if order == "aggregate_then_percent":
            b0 = float(np.nanmedian(s_hat_abs[b0_idx])) if b0_idx.size else float("nan")
            if (not np.isfinite(b0)) or b0 == 0.0:
                raise ValueError(f"Invalid baseline for harmonisation: median baseline={b0}.")
            s_hat_pct = self._to_percent_1d(s_hat_abs, b0, mode=percent_mode)
            diag = {
                "harmonize_output": True,
                "harmonize_aggregation_order": order,
                "harmonize_percent_mode": percent_mode,
                "harmonize_baseline_source": cfg.harmonize_baseline_source,
                "harmonize_baseline_b0": float(b0),
                "units_out": "percent",
                **baseline_info,
            }
            return s_hat_pct, diag

        # percent then aggregate
        if cfg.harmonize_baseline_per_locus:
            Lm = L.copy()
            Lm[~M] = np.nan
            b0_loci = np.nanmedian(Lm[b0_idx, :], axis=0)
            b0_loci = np.where(np.isfinite(b0_loci) & (b0_loci != 0.0), b0_loci, np.nan)

            L_pct = np.full_like(L, np.nan, dtype=float)
            for p in range(L.shape[1]):
                if np.isfinite(b0_loci[p]):
                    L_pct[:, p] = self._to_percent_1d(L[:, p], float(b0_loci[p]), mode=percent_mode)

            s_hat_pct = self._aggregate_loci(L_pct, M, mode=cfg.locus_agg)
            diag = {
                "harmonize_output": True,
                "harmonize_aggregation_order": order,
                "harmonize_percent_mode": percent_mode,
                "harmonize_baseline_source": cfg.harmonize_baseline_source,
                "harmonize_baseline_per_locus": True,
                "units_out": "percent",
                **baseline_info,
            }
            return s_hat_pct, diag

        b0 = float(np.nanmedian(s_hat_abs[b0_idx])) if b0_idx.size else float("nan")
        if (not np.isfinite(b0)) or b0 == 0.0:
            raise ValueError(f"Invalid baseline for harmonisation: median baseline={b0}.")
        L_pct = self._to_percent_2d(L, b0, mode=percent_mode)
        s_hat_pct = self._aggregate_loci(L_pct, M, mode=cfg.locus_agg)
        diag = {
            "harmonize_output": True,
            "harmonize_aggregation_order": order,
            "harmonize_percent_mode": percent_mode,
            "harmonize_baseline_source": cfg.harmonize_baseline_source,
            "harmonize_baseline_per_locus": False,
            "harmonize_baseline_b0": float(b0),
            "units_out": "percent",
            **baseline_info,
        }
        return s_hat_pct, diag

    @staticmethod
    def _get_baseline_indices(signal: SegmentSignal, cfg: MyMethodConfig) -> Tuple[np.ndarray, Dict[str, object]]:
        t = signal.t.astype(float)
        prot = signal.protocol
        src = str(cfg.harmonize_baseline_source).lower().strip()

        if src == "protocol_global":
            if getattr(prot, "global_baseline", None) is not None:
                b = prot.global_baseline
                start_sec = float(b.start_sec)
                end_sec = float(b.end_sec)
            else:
                src = "cycle0_baseline"

        if src == "cycle0_baseline":
            if prot.cycles is None or len(prot.cycles) == 0:
                raise ValueError("Cannot use cycle0_baseline: protocol has no cycles.")
            c0 = prot.cycles[0]
            if c0.baseline is None:
                raise ValueError("Cannot use cycle0_baseline: cycle 0 has no baseline window.")
            start_sec = float(c0.baseline.start_sec)
            end_sec = float(c0.baseline.end_sec)

        elif src == "custom":
            start_sec = float(cfg.harmonize_custom_baseline_start_sec)
            end_sec = float(cfg.harmonize_custom_baseline_end_sec)

        elif src not in {"protocol_global", "cycle0_baseline"}:
            raise ValueError("harmonize_baseline_source must be 'protocol_global', 'cycle0_baseline', or 'custom'.")

        idx = np.where((t >= start_sec) & (t < end_sec))[0]
        if idx.size == 0:
            raise ValueError(
                f"Baseline window empty. Window [{start_sec:.3f}, {end_sec:.3f}) within t=[{float(t[0]):.3f}, {float(t[-1]):.3f}]."
            )
        info = {"baseline_start_sec": float(start_sec), "baseline_end_sec": float(end_sec)}
        return idx, info

    @staticmethod
    def _to_percent_1d(y: np.ndarray, baseline: float, mode: str) -> np.ndarray:
        if (not np.isfinite(baseline)) or baseline == 0.0:
            return np.full_like(y, np.nan, dtype=float)
        if mode == "ratio":
            return 100.0 * (y / baseline)
        if mode == "delta_over_baseline":
            return 100.0 * ((y - baseline) / baseline)
        raise ValueError("mode must be 'delta_over_baseline' or 'ratio'.")

    @staticmethod
    def _to_percent_2d(X: np.ndarray, baseline: float, mode: str) -> np.ndarray:
        out = X.astype(float).copy()
        if (not np.isfinite(baseline)) or baseline == 0.0:
            out[:] = np.nan
            return out
        if mode == "ratio":
            return 100.0 * (out / baseline)
        if mode == "delta_over_baseline":
            return 100.0 * ((out - baseline) / baseline)
        raise ValueError("mode must be 'delta_over_baseline' or 'ratio'.")

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _fill_missing(X: np.ndarray, M: np.ndarray) -> np.ndarray:
        """Linear interpolation per locus, only if at least 2 valid points exist."""
        return fill_missing_interp(X, M)

    @staticmethod
    def _estimate_heartbeat(X: np.ndarray, M: np.ndarray, *, timestep_ms: float) -> Tuple[float, float]:
        x_masked = X.copy()
        x_masked[~M] = np.nan
        y = np.nanmean(x_masked, axis=1)
        m_y = np.any(M, axis=1) & np.isfinite(y)
        hb_bpm, hb_amp = extract_heartbeat_psd(y, timestep_ms=timestep_ms, mask=m_y)
        return float(hb_bpm), float(hb_amp)

    @staticmethod
    def _apply_lowpass(
        X: np.ndarray,
        M: np.ndarray,
        heartbeat_bpm: float,
        *,
        timestep_ms: float,
        extra_cutoff_hz: float,
        lp_order: int,
        min_block_len: Optional[int],
        padlen: Optional[int],
        taper_len: int,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        fs = 1000.0 / float(timestep_ms)
        cutoff_hz = heartbeat_bpm / 60.0 + float(extra_cutoff_hz)

        X_lp = np.full_like(X, np.nan, dtype=float)
        X_hi = np.full_like(X, np.nan, dtype=float)

        for p in range(X.shape[1]):
            y = X[:, p]
            m = M[:, p]
            y_lp, y_hi = lowpass_filter_masked(
                y,
                m,
                cutoff=cutoff_hz,
                sampling_rate=fs,
                order=int(lp_order),
                min_block_len=min_block_len,
                padlen=padlen,
                taper_len=int(taper_len),
            )
            X_lp[:, p] = y_lp
            X_hi[:, p] = y_hi

        return X_lp, X_hi, float(cutoff_hz)

    @staticmethod
    def _run_rpca_masked_smooth(
        X_lp: np.ndarray,
        M: np.ndarray,
        *,
        standardize: bool,
        destandardize: bool,
        rpca_kwargs: Dict[str, object],
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
        """Masked smooth RPCA on X_lp with mask M.

        If standardize is True:
          - centre by baseline median per locus
          - scale by baseline MAD (global pooled or per-locus)
        If destandardize is False:
          - returns S_hat/A_hat in standardized space (NOT reconverted)
        """
        X = X_lp.astype(float)
        M = M.astype(bool)

        M_eff = M & np.isfinite(X)

        mu = np.zeros((1, X.shape[1]), dtype=float)
        sd = np.ones((1, X.shape[1]), dtype=float)

        if standardize:
            baseline_idx = rpca_kwargs.pop("baseline_idx", None)
            if baseline_idx is None:
                raise ValueError("rpca_kwargs must contain baseline_idx for baseline centring and scaling.")
            bidx = np.asarray(baseline_idx, dtype=int).ravel()
            if bidx.size == 0:
                raise ValueError("baseline_idx is empty.")
            if np.any((bidx < 0) | (bidx >= X.shape[0])):
                raise ValueError("baseline_idx contains indices outside [0, T).")

            scale_mode = str(rpca_kwargs.pop("scale_mode", "global")).lower().strip()
            scale_floor = float(rpca_kwargs.pop("scale_floor", 1e-6))

            Xm = X.copy()
            Xm[~M_eff] = np.nan
            Xb = Xm[bidx, :]
            mu = np.nanmedian(Xb, axis=0, keepdims=True)

            rb = Xb - mu

            if scale_mode == "global":
                pool = rb[np.isfinite(rb)]
                if pool.size == 0:
                    sd_global = 1.0
                else:
                    med = float(np.median(pool))
                    mad = float(np.median(np.abs(pool - med)))
                    sd_global = 1.4826 * mad
                    if (not np.isfinite(sd_global)) or sd_global <= 0.0:
                        sd_global = float(np.std(pool)) if pool.size > 1 else 1.0
                        if (not np.isfinite(sd_global)) or sd_global <= 0.0:
                            sd_global = 1.0
                sd = np.full((1, X.shape[1]), max(float(sd_global), float(scale_floor)), dtype=float)

            elif scale_mode == "per_locus":
                sd = np.empty((1, X.shape[1]), dtype=float)
                for p in range(X.shape[1]):
                    r = rb[:, p]
                    r = r[np.isfinite(r)]
                    if r.size == 0:
                        s = 1.0
                    else:
                        med = float(np.median(r))
                        mad = float(np.median(np.abs(r - med)))
                        s = 1.4826 * mad
                        if (not np.isfinite(s)) or s <= 0.0:
                            s = float(np.std(r)) if r.size > 1 else 1.0
                            if (not np.isfinite(s)) or s <= 0.0:
                                s = 1.0
                    sd[0, p] = max(float(s), float(scale_floor))

            else:
                raise ValueError("standardize_scale_mode must be 'global' or 'per_locus'.")

            Z = (X - mu) / sd
        else:
            Z = X

        Z = Z.astype(float, copy=True)
        Z[~M_eff] = 0.0
        Z[~np.isfinite(Z)] = 0.0

        rpca = RobustPCA(**rpca_kwargs)
        S_pt, A_pt, diag = rpca.fit(Z.T, M_eff.T)  # (P,T)
        S = S_pt.T
        A = A_pt.T

        # reconvert to absolute units ONLY if requested
        if standardize and destandardize:
            S = S * sd + mu
            A = A * sd

        diag_out: Dict[str, object] = {}
        if isinstance(diag, dict):
            for kk, vv in diag.items():
                diag_out[kk] = vv.tolist() if isinstance(vv, np.ndarray) else vv

        # add standardization info (small summaries only)
        if standardize:
            diag_out["standardize_destandardize"] = bool(destandardize)
            diag_out["standardize_mu_median"] = float(np.nanmedian(mu))
            diag_out["standardize_sd_median"] = float(np.nanmedian(sd))
            diag_out["standardize_sd_min"] = float(np.nanmin(sd))
            diag_out["standardize_sd_max"] = float(np.nanmax(sd))

        return S, A, diag_out

    @staticmethod
    def _aggregate_loci(x: np.ndarray, m: np.ndarray, mode: str) -> np.ndarray:
        return aggregate_loci(x, m, mode=mode)

    @staticmethod
    def _cycle_window_indices(signal: SegmentSignal, cycle_idx: int) -> np.ndarray:
        """Return indices for flicker + recovery of a given cycle."""
        t = signal.t.astype(float)
        prot = signal.protocol

        if getattr(prot, "cycles", None) is None or len(prot.cycles) <= cycle_idx:
            return np.arange(t.shape[0], dtype=int)

        cyc = prot.cycles[cycle_idx]

        if getattr(cyc, "flicker", None) is None or getattr(cyc, "recovery", None) is None:
            return np.arange(t.shape[0], dtype=int)

        start_sec = float(cyc.flicker.start_sec)
        end_sec = float(cyc.recovery.end_sec)

        idx = np.where((t >= start_sec) & (t < end_sec))[0]
        if idx.size == 0:
            idx = np.arange(t.shape[0], dtype=int)
        return idx

    @staticmethod
    def _confidence_per_cycle(
        *,
        signal: SegmentSignal,
        S_hat: np.ndarray,
        A_hat: np.ndarray,
        M: np.ndarray,
        eps: float = 1e-8,
    ) -> Dict[str, object]:
        """Compute artefact to physiology energy ratio per cycle on observed support."""
        prot = signal.protocol
        tlen = S_hat.shape[0]

        if getattr(prot, "cycles", None) is None:
            n_cycles = 1
        else:
            n_cycles = max(1, len(prot.cycles))

        out: Dict[str, object] = {"conf_n_cycles": int(n_cycles)}

        c_list: List[float] = []
        obs_list: List[int] = []

        for j in range(n_cycles):
            idx = MyMethodRPCA._cycle_window_indices(signal, j)
            idx = idx[(idx >= 0) & (idx < tlen)]
            if idx.size == 0:
                idx = np.arange(tlen, dtype=int)

            Mw = M[idx, :]
            Sw = S_hat[idx, :]
            Aw = A_hat[idx, :]

            MA = Mw * Aw
            MS = Mw * Sw

            n_obs = int(np.count_nonzero(Mw))
            num = float(np.linalg.norm(MA, ord="fro"))
            den = float(np.linalg.norm(MS, ord="fro")) + float(eps)
            c = float(num / den)

            out[f"conf_c_cycle{j}"] = c
            out[f"conf_nobs_cycle{j}"] = n_obs

            c_list.append(c)
            obs_list.append(n_obs)

        out["conf_c_median"] = float(np.median(c_list)) if len(c_list) else float("nan")
        out["conf_c_min"] = float(np.min(c_list)) if len(c_list) else float("nan")
        out["conf_c_max"] = float(np.max(c_list)) if len(c_list) else float("nan")
        out["conf_nobs_total"] = int(np.sum(obs_list)) if len(obs_list) else 0

        return out

    @staticmethod
    def _support_good_timepoints(M: np.ndarray, min_valid_abs: int) -> np.ndarray:
        """Return good_t (T,) where at least min_valid_abs loci are valid."""
        return support_good_timepoints(M, min_valid_abs)

    @staticmethod
    def _hampel_filter_1d(
        y: np.ndarray,
        *,
        k: int = 9,
        nsigmas: float = 3.0,
        replace: str = "median",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Hampel filter (NaN-aware) for 1D arrays.

        Parameters
        ----------
        y : (T,) array
        k : half-window size
        nsigmas : threshold in MAD units
        replace : "median" to replace outliers with window median, or "nan"

        Returns
        -------
        y_out : filtered signal
        is_outlier : boolean mask of outliers (T,)
        """
        return hampel_filter_1d(y, k=k, nsigmas=nsigmas, replace=replace)

    @staticmethod
    def _resolve_min_valid_abs(M: np.ndarray, min_abs: int, min_frac: Optional[float]) -> int:
        """
        Choose an absolute min-valid threshold from either:
          - fraction of total loci (preferred if provided), or
          - absolute min count.
        """
        return resolve_min_valid_abs(M, min_abs=min_abs, min_frac=min_frac)
