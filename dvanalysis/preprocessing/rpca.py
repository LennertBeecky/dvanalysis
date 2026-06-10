"""Masked Smooth RPCA solver (ADMM) with temporal Tikhonov prior.

Objective:
    min_{S,A}  ||S||_* + lmb*||A||_1 + (gamma/2)*||D_t S||_F^2
    s.t.       M . X = M . (S + A)

This module contains the core optimisation algorithm.
Do NOT modify without verifying correctness against reference outputs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.linalg as la
from scipy.linalg import solve_banded


def soft_thresholding(y: np.ndarray, tau: float) -> np.ndarray:
    """Elementwise soft threshold."""
    return np.sign(y) * np.clip(np.abs(y) - tau, a_min=0.0, a_max=None)


def svd_shrinkage(y: np.ndarray, tau: float) -> np.ndarray:
    """Proximal operator of the nuclear norm via singular value thresholding."""
    U, s, Vh = np.linalg.svd(y, full_matrices=False)
    s_t = soft_thresholding(s, tau)
    return U @ np.diag(s_t) @ Vh


def _solve_tridiagonal(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Thomas algorithm for tridiagonal system.

    Convention: a[0] = 0, c[-1] = 0.
    Adds a tiny eps to pivots to avoid hard crashes in pathological mask patterns.
    """
    n = int(d.size)
    if n == 0:
        return d.copy()

    cp = np.empty(n, dtype=float)
    dp = np.empty(n, dtype=float)

    denom = b[0]
    if not np.isfinite(denom):
        denom = 0.0
    denom = denom + eps if abs(denom) < eps else denom
    cp[0] = c[0] / denom
    dp[0] = d[0] / denom

    for i in range(1, n):
        denom = b[i] - a[i] * cp[i - 1]
        if not np.isfinite(denom):
            denom = 0.0
        denom = denom + eps if abs(denom) < eps else denom
        cp[i] = c[i] / denom if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom

    x = np.empty(n, dtype=float)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


@dataclass
class RobustPCA:
    """Masked Smooth RPCA (ADMM) with temporal Tikhonov prior.

    Parameters
    ----------
    lmb : float
        Sparsity penalty weight (lambda).
    mu : float
        Augmented Lagrangian parameter for data fidelity.
    rho : float
        Augmented Lagrangian parameter for S=Z split.
    gamma : float
        Temporal smoothness weight (0 = no smoothness).
    max_iter : int
        Maximum ADMM iterations.
    tol_rel : float
        Relative convergence tolerance.
    path : str
        Optional directory for saving intermediate results.
    artery : bool
        Hint for naming saved files.
    save_every : int
        Save interval (0 = never).
    tri_eps : float
        Numerical stabiliser for the tridiagonal solve.

    Input shapes: X (P, T), M (P, T).
    """

    lmb: float
    mu: float = 1.0
    rho: float = 5.0
    gamma: float = 0.0
    max_iter: int = 200
    tol_rel: float = 1e-4

    path: str = ""
    artery: bool = True
    save_every: int = 0
    tri_eps: float = 1e-12

    def fit(self, X: np.ndarray, M: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
        """Run ADMM to decompose X = S + A subject to mask M.

        Parameters
        ----------
        X : (P, T) observation matrix
        M : (P, T) boolean mask (True = observed). Defaults to all True.

        Returns
        -------
        S : (P, T) low-rank temporally smooth component
        A : (P, T) sparse component (zero outside mask)
        diagnostics : dict with convergence traces
        """
        if X.ndim != 2:
            raise ValueError("X must be 2D, shape (P,T).")

        P, T = X.shape

        if M is None:
            M = np.ones_like(X, dtype=bool)
        else:
            if M.shape != X.shape:
                raise ValueError("Mask M must have same shape as X.")
            M = M.astype(bool)

        if self.lmb <= 0.0:
            raise ValueError("lmb must be > 0.")
        if self.mu <= 0.0:
            raise ValueError("mu must be > 0.")
        if self.rho <= 0.0:
            raise ValueError("rho must be > 0.")
        if self.gamma < 0.0:
            raise ValueError("gamma must be >= 0.")

        Xf = X.astype(float).copy()
        Xf[~np.isfinite(Xf)] = 0.0

        S = np.zeros_like(Xf)
        A = np.zeros_like(Xf)
        Z = np.zeros_like(Xf)
        Y = np.zeros_like(Xf)
        U = np.zeros_like(Xf)

        if T >= 2:
            dt_diag = np.ones(T, dtype=float) * 2.0
            dt_diag[0] = 1.0
            dt_diag[-1] = 1.0
        else:
            dt_diag = np.zeros(T, dtype=float)

        r_data_hist: List[float] = []
        r_split_hist: List[float] = []
        obj_proxy_hist: List[float] = []

        def maybe_save(k: int, S_: np.ndarray, A_: np.ndarray) -> None:
            if self.save_every <= 0 or (k % self.save_every) != 0 or not self.path:
                return
            try:
                tag = "Artery" if self.artery else "Vein"
                np.save(os.path.join(self.path, f"S_{tag}_iter_{k}.npy"), S_)
                np.save(os.path.join(self.path, f"A_{tag}_iter_{k}.npy"), A_)
            except Exception:
                pass

        norm_X = float(la.norm(Xf[M].ravel(), ord=2)) if np.any(M) else 1.0
        last_k = 0

        for k in range(self.max_iter):
            last_k = k

            # A update
            R = Xf - S + Y
            A_new = np.zeros_like(A)
            if np.any(M):
                A_new[M] = soft_thresholding(R[M], self.lmb / self.mu)

            # Z update
            Z_new = svd_shrinkage(S + U, 1.0 / self.rho)

            # S update — vectorised tridiagonal solve via scipy.linalg.solve_banded
            M_float = M.astype(float)  # (P, T)
            rhs_all = (self.rho * (Z_new - U)
                       + self.mu * M_float * (Xf - A_new + Y))  # (P, T)

            if self.gamma <= 0.0 or T < 2:
                # No temporal smoothness: pointwise division
                denom_all = self.rho + self.mu * M_float  # (P, T)
                S_new = np.where(denom_all > 0.0, rhs_all / denom_all, 0.0)
            else:
                # Tridiagonal solve per locus using scipy.linalg.solve_banded
                # Band matrix ab has shape (3, T): [upper, main, lower]
                # The off-diagonals are the same for all loci; only the main diagonal varies
                S_new = np.empty_like(S)
                off_diag = -self.gamma

                for p in range(P):
                    ab = np.empty((3, T), dtype=float)
                    ab[0, 1:] = off_diag   # upper diagonal (c)
                    ab[0, 0] = 0.0
                    ab[1, :] = self.rho + self.mu * M_float[p, :] + self.gamma * dt_diag  # main diagonal (b)
                    ab[2, :-1] = off_diag   # lower diagonal (a)
                    ab[2, -1] = 0.0

                    S_new[p, :] = solve_banded((1, 1), ab, rhs_all[p, :])

            # Dual updates
            r_data = np.zeros_like(Xf)
            if np.any(M):
                r_data[M] = Xf[M] - S_new[M] - A_new[M]
            Y_new = Y + r_data

            r_split = S_new - Z_new
            U_new = U + r_split

            # Residuals
            r1 = float(la.norm(r_data[M].ravel(), ord=2) / (norm_X + 1e-12)) if np.any(M) else 0.0
            denom2 = float(la.norm(S_new.ravel(), ord=2) + 1e-12)
            r2 = float(la.norm(r_split.ravel(), ord=2) / denom2)

            r_data_hist.append(r1)
            r_split_hist.append(r2)

            # Objective proxy
            nuc = float(np.sum(la.svd(Z_new, compute_uv=False)))
            l1 = float(np.sum(np.abs(A_new[M]))) if np.any(M) else 0.0
            if self.gamma > 0.0 and T >= 2:
                dS = S_new[:, 1:] - S_new[:, :-1]
                tikh = 0.5 * self.gamma * float(np.sum(dS * dS))
            else:
                tikh = 0.0
            obj_proxy_hist.append(nuc + self.lmb * l1 + tikh)

            S, A, Z, Y, U = S_new, A_new, Z_new, Y_new, U_new

            maybe_save(k, S, A)

            if max(r1, r2) < self.tol_rel:
                break

        A = A.copy()
        A[~M] = 0.0

        diagnostics: Dict[str, object] = {
            "n_iter": int(last_k + 1),
            "r_data_hist": np.asarray(r_data_hist, dtype=float),
            "r_split_hist": np.asarray(r_split_hist, dtype=float),
            "obj_proxy_hist": np.asarray(obj_proxy_hist, dtype=float),
            "final_r_data": float(r_data_hist[-1]) if r_data_hist else float("nan"),
            "final_r_split": float(r_split_hist[-1]) if r_split_hist else float("nan"),
            "mu": float(self.mu),
            "rho": float(self.rho),
            "gamma": float(self.gamma),
            "lmb": float(self.lmb),
            "tri_eps": float(self.tri_eps),
        }

        return S, A, diagnostics
