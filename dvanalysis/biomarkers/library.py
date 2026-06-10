"""Comprehensive biomarker computation library for DVA signals.

Contains ~40 calc_* functions covering baseline, dilation, constriction,
AUC, and combined/derived biomarkers.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

from dvanalysis.domain import TimeWindow, StimulusCycle
from dvanalysis.utils.math import nanstat, safe_nanargmax, safe_nanargmin
from .windows import WindowLike, as_window, idx_for_window


def _dt_from_t(t: np.ndarray) -> float:
    if t.size < 2:
        return float("nan")
    d = np.diff(t)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan")
    return float(np.median(d))


# -------------------------
# Baseline family
# -------------------------

def calc_baseline_dynamic(y: np.ndarray, t: np.ndarray, end_time_sec: float, duration_sec: float = 20.0, *, strict: bool = True) -> float:
    end_time_sec = float(end_time_sec)
    w = TimeWindow("baseline_dynamic", end_time_sec - float(duration_sec), end_time_sec)
    idx = idx_for_window(t, w, strict=strict, label="baseline_dynamic")
    return float(np.nanmean(y[idx]))


def calc_baseline_diameter(y: np.ndarray, t: np.ndarray, w: WindowLike, *, strict: bool = True, stat: str = "median") -> float:
    idx = idx_for_window(t, w, strict=strict, label="baseline_diameter")
    return nanstat(y[idx], stat)


def calc_baseline_max(y: np.ndarray, t: np.ndarray, w: WindowLike, *, strict: bool = True) -> float:
    idx = idx_for_window(t, w, strict=strict, label="baseline_max")
    return float(np.nanmax(y[idx]))


def calc_baseline_min(y: np.ndarray, t: np.ndarray, w: WindowLike, *, strict: bool = True) -> float:
    idx = idx_for_window(t, w, strict=strict, label="baseline_min")
    return float(np.nanmin(y[idx]))


def calc_baseline_fluct(y: np.ndarray, t: np.ndarray, w: WindowLike, *, strict: bool = True) -> float:
    bmax = calc_baseline_max(y, t, w, strict=strict)
    bmin = calc_baseline_min(y, t, w, strict=strict)
    return float(bmax - bmin)


# -------------------------
# Dilation family
# -------------------------

def calc_dilation_max(y: np.ndarray, t: np.ndarray, w: WindowLike, *, strict: bool = True) -> float:
    idx = idx_for_window(t, w, strict=strict, label="dilation_max")
    return float(np.nanmax(y[idx]))


def calc_dilation_max_bc(y, t, w_flicker, w_baseline, *, strict=True, baseline_stat="median"):
    bd = calc_baseline_diameter(y, t, w_baseline, strict=strict, stat=baseline_stat)
    idx = idx_for_window(t, w_flicker, strict=strict, label="dilation_max_bc")
    return float(np.nanmax(y[idx] - bd))


def calc_dilation_max_perc(y, t, w_dilation, w_baseline, *, strict=True, baseline_stat="median"):
    max_d = calc_dilation_max(y, t, w_dilation, strict=strict)
    bd = calc_baseline_diameter(y, t, w_baseline, strict=strict, stat=baseline_stat)
    if not np.isfinite(bd) or bd == 0.0:
        return float("nan")
    return float((max_d - bd) / bd * 100.0)


def calc_dilation_max_t(y, t, w, *, strict=True, relative_to_start=True):
    ww = as_window(w, phase="dilation_window")
    idx = idx_for_window(t, ww, strict=strict, label="dilation_max_t")
    yy = y[idx]
    j = safe_nanargmax(yy)
    if j is None:
        return float("nan")
    tmax = float(t[idx[j]])
    return float(tmax - ww.start_sec) if relative_to_start else tmax


def calc_dilation_max_t_bc(y, t, w_flicker, w_baseline, *, strict=True, relative_to_start=True, baseline_stat="median"):
    bd = calc_baseline_diameter(y, t, w_baseline, strict=strict, stat=baseline_stat)
    ww = as_window(w_flicker, phase="dilation_window")
    idx = idx_for_window(t, ww, strict=strict, label="dilation_max_t_bc")
    yy = y[idx] - bd
    j = safe_nanargmax(yy)
    if j is None:
        return float("nan")
    tmax = float(t[idx[j]])
    return float(tmax - ww.start_sec) if relative_to_start else tmax


def calc_dilation_max_I(y, t, ref_time_sec, *, window=(-9.0, 3.0), strict=True):
    w = TimeWindow("maxd_I", float(ref_time_sec) + float(window[0]), float(ref_time_sec) + float(window[1]))
    idx = idx_for_window(t, w, strict=strict, label="dilation_max_I")
    return float(np.nanmax(y[idx]))


def calc_dilation_max_II(y, t, ref_time_sec, *, window=(-2.0, 3.0), strict=True):
    w = TimeWindow("maxd_II", float(ref_time_sec) + float(window[0]), float(ref_time_sec) + float(window[1]))
    idx = idx_for_window(t, w, strict=strict, label="dilation_max_II")
    return float(np.nanmax(y[idx]))


def calc_dilation_max_III(y, t, ref_time_sec, *, window=(-3.0, 3.0), strict=True):
    w = TimeWindow("maxd_III", float(ref_time_sec) + float(window[0]), float(ref_time_sec) + float(window[1]))
    idx = idx_for_window(t, w, strict=strict, label="dilation_max_III")
    return float(np.nanmean(y[idx]))


def calc_dilation_max_30_t(y, t, w_search, w_baseline, *, strict=True, baseline_stat="median"):
    ww = as_window(w_search, phase="dilation_search")
    idx = idx_for_window(t, ww, strict=strict, label="dilation_max_30_t")
    yy = y[idx]
    max_d = float(np.nanmax(yy))
    bd = calc_baseline_diameter(y, t, w_baseline, strict=strict, stat=baseline_stat)
    if not np.isfinite(bd) or not np.isfinite(max_d):
        return float("nan")
    thr = bd + 0.3 * (max_d - bd)
    ok = np.isfinite(yy)
    if not np.any(ok):
        return float("nan")
    hit = np.where(ok & (yy >= thr))[0]
    if hit.size == 0:
        return float("nan")
    t_hit = float(t[idx[int(hit[0])]])
    return float(t_hit - ww.start_sec)


def calc_dilation_mmd(y, t, t_max_sec, *, window=(-2.0, 2.0), strict=True):
    w = TimeWindow("mmd", float(t_max_sec) + float(window[0]), float(t_max_sec) + float(window[1]))
    idx = idx_for_window(t, w, strict=strict, label="dilation_mmd")
    return float(np.nanmedian(y[idx]))


def calc_dilation_mmd_perc(y, t, t_max_sec, baseline_value, *, window=(-2.0, 2.0), strict=True):
    mmd = calc_dilation_mmd(y, t, t_max_sec, window=window, strict=strict)
    if not np.isfinite(baseline_value) or baseline_value == 0.0:
        return float("nan")
    return float((mmd - baseline_value) / baseline_value * 100.0)


def calc_dilation_flicker(y, t, w_flicker, *, strict=True):
    idx = idx_for_window(t, w_flicker, strict=strict, label="dilation_flicker")
    return float(np.nanmean(y[idx]))


def calc_dilation_slope(y, t, w_search, w_baseline, *, strict=True, baseline_stat="median"):
    ww = as_window(w_search, phase="dilation_search")
    max_d = calc_dilation_max(y, t, ww, strict=strict)
    bd = calc_baseline_diameter(y, t, w_baseline, strict=strict, stat=baseline_stat)
    tmax_rel = calc_dilation_max_t(y, t, ww, strict=strict, relative_to_start=True)
    if not np.isfinite(max_d) or not np.isfinite(bd) or not np.isfinite(tmax_rel) or tmax_rel <= 0.0:
        return float("nan")
    return float((max_d - bd) / tmax_rel)


def calc_dilation_dfc(y, t, t_end_sec, *, strict=True):
    idx = np.where(t >= float(t_end_sec))[0]
    if idx.size == 0:
        if strict:
            raise ValueError(f"t_end_sec={t_end_sec} exceeds recording end {float(t[-1]):.3f}")
        return float("nan")
    return float(y[int(idx[0])])


# -------------------------
# Constriction family
# -------------------------

def calc_constr_max(y, t, w, *, strict=True):
    idx = idx_for_window(t, w, strict=strict, label="constr_max")
    return float(np.nanmin(y[idx]))


def calc_constr_max_bc(y, t, w_recovery, w_baseline, *, strict=True, baseline_stat="median"):
    bd = calc_baseline_diameter(y, t, w_baseline, strict=strict, stat=baseline_stat)
    idx = idx_for_window(t, w_recovery, strict=strict, label="constr_max_bc")
    return float(np.nanmin(y[idx] - bd))


def calc_constr_max_perc(y, t, w_constr, w_baseline, *, strict=True, baseline_stat="median"):
    max_c = calc_constr_max(y, t, w_constr, strict=strict)
    bd = calc_baseline_diameter(y, t, w_baseline, strict=strict, stat=baseline_stat)
    if not np.isfinite(bd) or bd == 0.0:
        return float("nan")
    return float((bd - max_c) / bd * 100.0)


def calc_constr_max_t(y, t, w_constr, ref_time_sec, *, strict=True):
    ww = as_window(w_constr, phase="constr_window")
    idx = idx_for_window(t, ww, strict=strict, label="constr_max_t")
    yy = y[idx]
    j = safe_nanargmin(yy)
    if j is None:
        return float("nan")
    tmin = float(t[idx[j]])
    return float(tmin - float(ref_time_sec))


def calc_constr_mmc(y, t, t_min_sec, *, window=(-2.0, 2.0), strict=True):
    w = TimeWindow("mmc", float(t_min_sec) + float(window[0]), float(t_min_sec) + float(window[1]))
    idx = idx_for_window(t, w, strict=strict, label="constr_mmc")
    return float(np.nanmedian(y[idx]))


def calc_constr_max_median_perc(y, t, t_min_sec, w_baseline, *, window=(-2.0, 2.0), strict=True, baseline_stat="median"):
    bd = calc_baseline_diameter(y, t, w_baseline, strict=strict, stat=baseline_stat)
    mmc = calc_constr_mmc(y, t, t_min_sec, window=window, strict=strict)
    if not np.isfinite(bd) or bd == 0.0:
        return float("nan")
    return float((bd - mmc) / bd * 100.0)


def calc_constr_avg(y, t, start_time_sec, duration_sec=40.0, *, strict=True):
    w = TimeWindow("acon", float(start_time_sec), float(start_time_sec) + float(duration_sec))
    idx = idx_for_window(t, w, strict=strict, label="constr_avg")
    return float(np.nanmean(y[idx]))


def calc_constr_slope(y, t, t_maxd_sec, max_dilation_value, t_maxc_sec, max_constr_value):
    dt = float(t_maxc_sec - t_maxd_sec)
    if not np.isfinite(dt) or dt <= 0.0:
        return float("nan")
    return float((max_constr_value - max_dilation_value) / dt)


# -------------------------
# AUC family
# -------------------------

def calc_auc(y, t, w, *, strict=True):
    idx = idx_for_window(t, w, strict=strict, label="auc")
    if idx.size < 2:
        return float("nan")
    yy = y[idx]
    tt = t[idx]
    ok = np.isfinite(yy) & np.isfinite(tt)
    if int(ok.sum()) < 2:
        return float("nan")
    return float(np.trapz(yy[ok], x=tt[ok]))


def calc_aucaf(y, t, recovery_start_sec, *, strict=True):
    w = TimeWindow("aucaf", float(recovery_start_sec) + 50.0, float(recovery_start_sec) + 80.0)
    return calc_auc(y, t, w, strict=strict)


def calc_aucef(y, t, recovery_start_sec, *, strict=True):
    w = TimeWindow("aucef", float(recovery_start_sec) + 15.0, float(recovery_start_sec) + 45.0)
    return calc_auc(y, t, w, strict=strict)


def calc_auc_dilation(y, t, w_baseline, w_flicker, *, strict=True, baseline_stat="median"):
    bd = calc_baseline_diameter(y, t, w_baseline, strict=strict, stat=baseline_stat)
    idx = idx_for_window(t, w_flicker, strict=strict, label="auc_dilation")
    if idx.size < 2 or not np.isfinite(bd):
        return float("nan")
    yy = np.maximum(y[idx] - bd, 0.0)
    tt = t[idx]
    ok = np.isfinite(yy) & np.isfinite(tt)
    if int(ok.sum()) < 2:
        return float("nan")
    return float(np.trapz(yy[ok], x=tt[ok]))


def calc_auc_constriction(y, t, w_baseline, w_recovery, *, strict=True, baseline_stat="median"):
    bd = calc_baseline_diameter(y, t, w_baseline, strict=strict, stat=baseline_stat)
    idx = idx_for_window(t, w_recovery, strict=strict, label="auc_constriction")
    if idx.size < 2 or not np.isfinite(bd):
        return float("nan")
    yy = np.maximum(bd - y[idx], 0.0)
    tt = t[idx]
    ok = np.isfinite(yy) & np.isfinite(tt)
    if int(ok.sum()) < 2:
        return float("nan")
    return float(np.trapz(yy[ok], x=tt[ok]))


# -------------------------
# Combined / derived
# -------------------------

def calc_fic(y, t, w_baseline, flicker_end_sec, *, strict=True, baseline_stat="median"):
    bd = calc_baseline_diameter(y, t, w_baseline, strict=strict, stat=baseline_stat)
    fl = calc_dilation_dfc(y, t, flicker_end_sec, strict=strict)
    if not np.isfinite(bd) or bd == 0.0:
        return float("nan")
    return float((fl - bd) / bd * 100.0)


def calc_da(y, t, w_flicker, w_recovery, *, strict=True):
    max_d = calc_dilation_max(y, t, w_flicker, strict=strict)
    max_c = calc_constr_max(y, t, w_recovery, strict=strict)
    return float(max_d - max_c)


def calc_da_bc(y, t, w_flicker, w_recovery, w_baseline, *, strict=True, baseline_stat="median"):
    max_d = calc_dilation_max_bc(y, t, w_flicker, w_baseline, strict=strict, baseline_stat=baseline_stat)
    max_c = calc_constr_max_bc(y, t, w_recovery, w_baseline, strict=strict, baseline_stat=baseline_stat)
    return float(max_d - max_c)


def calc_da_perc(y, t, w_flicker, w_recovery, w_baseline, *, strict=True, baseline_stat="median"):
    da = calc_da(y, t, w_flicker, w_recovery, strict=strict)
    bd = calc_baseline_diameter(y, t, w_baseline, strict=strict, stat=baseline_stat)
    if not np.isfinite(bd) or bd == 0.0:
        return float("nan")
    return float(da / bd * 100.0)


def calc_bcfr(y, t, w_baseline, w_flicker, w_recovery, *, strict=True, baseline_stat="median"):
    da = calc_da(y, t, w_flicker, w_recovery, strict=strict)
    bdf = calc_baseline_fluct(y, t, w_baseline, strict=strict)
    return float(da - bdf)


def calc_t_mdmc(y, t, w_flicker, w_recovery, *, strict=True):
    wf = as_window(w_flicker, phase="flicker")
    wr = as_window(w_recovery, phase="recovery")
    idx_f = idx_for_window(t, wf, strict=strict, label="t_mdmc_flicker")
    idx_r = idx_for_window(t, wr, strict=strict, label="t_mdmc_recovery")
    jf = safe_nanargmax(y[idx_f])
    jr = safe_nanargmin(y[idx_r])
    if jf is None or jr is None:
        return float("nan")
    t_maxd = float(t[idx_f[jf]])
    t_maxc = float(t[idx_r[jr]])
    return float(t_maxc - t_maxd)


def calc_auc_center_baseline(y, t, w_baseline, start_time_sec, *, strict=True, baseline_stat="median"):
    bd = calc_baseline_diameter(y, t, w_baseline, strict=strict, stat=baseline_stat)
    if not np.isfinite(bd):
        return float("nan")
    idx = np.where(t >= float(start_time_sec))[0]
    if idx.size < 2:
        if strict:
            raise ValueError("Not enough samples after start_time_sec for CoG.")
        return float("nan")
    yy = np.maximum(y[idx] - bd, 0.0)
    tt = t[idx]
    ok = np.isfinite(yy) & np.isfinite(tt)
    if int(ok.sum()) < 2:
        return float("nan")
    tt_ok = tt[ok]
    yy_ok = yy[ok]
    auc_total = float(np.trapz(yy_ok, x=tt_ok))
    if not np.isfinite(auc_total) or auc_total <= 0.0:
        return float("nan")
    target = 0.5 * auc_total
    dt = np.diff(tt_ok)
    area_steps = 0.5 * (yy_ok[:-1] + yy_ok[1:]) * dt
    cum = np.cumsum(area_steps)
    k = int(np.searchsorted(cum, target))
    if k >= (tt_ok.size - 1):
        return float(tt_ok[-1])
    return float(tt_ok[k])
